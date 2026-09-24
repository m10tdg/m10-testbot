#!/usr/bin/env python3
"""
Evaluation of the fine-tuned Llama-3.3-70B-Instruct LoRA adapter.

Mirrors evaluate_deepseek.py, adapted for Llama:
- Base model (BF16, ~140 GB) is sharded across all 8 GCDs with
  device_map="balanced"; the LoRA adapter is attached on top.
- Uses attn_implementation="sdpa" (no flash_attn patch needed).
- Loads the tokenizer from the adapter directory (saved at end of training).
- Generation stops on <|eot_id|> as well as the tokenizer EOS.
- Batched greedy decoding (left padding) for throughput; decoding is
  memory-bandwidth bound on a 70B model, so batching helps a lot.
- Saves per-example predictions + a summary with syntax-valid, Playwright-call
  and exact-match rates.
"""

import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path("/project/project_465003167/m10-testbot/finetuning")

BASE_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR   = PROJECT_DIR / "llama-finetuned-final"
VAL_FILE      = PROJECT_DIR / "dataset" / "validation_data.jsonl"
OUTPUT_DIR    = PROJECT_DIR / "training_results"
OUTPUT_FILE   = OUTPUT_DIR / "evaluation_llama_finetuned.json"
SUMMARY_FILE  = OUTPUT_DIR / "evaluation_llama_summary.json"

# Maximum new tokens per example (responses are ~350 tokens at most).
MAX_NEW_TOKENS = 512

# Prompts per generate() call. Lower this if you hit OOM.
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))

MAX_MEMORY_PER_GPU = "56GiB"


# ============================================================================
# SYSTEM PROMPT  (identical to training)
# ============================================================================

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a Playwright test automation expert. "
        "Given a test instruction, URL, and DOM structure, "
        "generate a concise executable Playwright Python script. "
        "Output ONLY Python code using page.locator(), "
        "page.fill(), page.click(), page.wait_for_url(), "
        "and similar Playwright methods. "
        "Use the exact selector IDs from the DOM structure. "
        "Include try/except error handling. "
        "No explanations, no markdown, only Python code."
    ),
}


# ============================================================================
# HELPERS
# ============================================================================

def normalize_code(text: str) -> str:
    """
    Strip leading/trailing whitespace and normalize internal whitespace so
    that minor formatting differences don't affect exact-match scoring.
    """
    text = re.sub(r"^```(?:python)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_syntactically_valid(code: str) -> bool:
    """Return True if the generated string is valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def contains_playwright_calls(code: str) -> bool:
    """Return True if the code references at least one Playwright method."""
    playwright_patterns = [
        r"page\.",
        r"playwright",
        r"async_playwright",
        r"sync_playwright",
    ]
    return any(re.search(p, code, re.IGNORECASE) for p in playwright_patterns)


def build_prompt(tokenizer, example) -> str:
    user_content = str(example.get("instruction") or "").strip()
    input_text   = str(example.get("input")       or "").strip()

    if input_text:
        user_content = f"{user_content}\n\n{input_text}"

    messages = [
        SYSTEM_MESSAGE,
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("LLAMA-3.3-70B-INSTRUCT  —  FINE-TUNED EVALUATION")
    print("=" * 80)
    print(f"Base model:   {BASE_MODEL}")
    print(f"Adapter:      {ADAPTER_DIR}")
    print(f"Val file:     {VAL_FILE}")
    print(f"Output:       {OUTPUT_FILE}")
    print(f"Batch size:   {EVAL_BATCH_SIZE}")

    # ------------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------------

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found:\n  {ADAPTER_DIR}\n"
            "Run finetune_llama.py first."
        )

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file not found:\n  {VAL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # GPU check
    # ------------------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Run inside a LUMI GPU job.")

    n_gpus = torch.cuda.device_count()
    total_gib = sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)
    ) / 1024**3

    print(f"\nGPU count: {n_gpus}  ({torch.cuda.get_device_name(0)})")
    print(f"Total GPU memory: {total_gib:.1f} GiB")

    if total_gib < 300:
        raise RuntimeError(
            "Llama-3.3-70B in BF16 needs a full LUMI-G node (8 GCDs). "
            "Request --gpus-per-node=8."
        )

    # ------------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------------

    print("\nLoading tokenizer from adapter directory...")

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"   # Left-pad for batched generation.

    # Stop on both the tokenizer EOS and <|eot_id|> (end of assistant turn).
    eos_ids = {tokenizer.eos_token_id}
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        eos_ids.add(eot_id)
    eos_ids = sorted(i for i in eos_ids if i is not None)

    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token:  {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"Stop ids:   {eos_ids}")
    print("✓ Tokenizer loaded")

    # ------------------------------------------------------------------------
    # Base model + LoRA adapter
    # ------------------------------------------------------------------------

    print("\nLoading base model (sharded across GPUs)...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="balanced",
        max_memory={i: MAX_MEMORY_PER_GPU for i in range(n_gpus)},
        attn_implementation="sdpa",
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params / 1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print("✓ Model ready")

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    print("\nLoading validation dataset...")

    dataset = load_dataset("json", data_files=str(VAL_FILE))["train"]
    examples = [dataset[i] for i in range(len(dataset))]

    print(f"✓ {len(examples)} validation examples")

    # ------------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------------

    predictions = []

    syntax_ok     = 0
    playwright_ok = 0
    exact_matches = 0

    print(f"\nGenerating predictions (max_new_tokens={MAX_NEW_TOKENS})...\n")

    with torch.inference_mode():

        for start in tqdm(
            range(0, len(examples), EVAL_BATCH_SIZE),
            desc="Evaluating",
            unit="batch",
        ):
            batch_examples = examples[start:start + EVAL_BATCH_SIZE]
            prompts = [build_prompt(tokenizer, ex) for ex in batch_examples]

            # The chat template already contains <|begin_of_text|>.
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to("cuda:0")

            # Greedy decoding. temperature/top_p are cleared because the
            # model's generation_config ships with sampling values.
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )

            prompt_len = inputs["input_ids"].shape[1]

            for j, ex in enumerate(batch_examples):
                i = start + j

                gen_tokens = outputs[j][prompt_len:]
                pred_text  = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                pred_normalized = normalize_code(pred_text)
                ref_normalized  = normalize_code(str(ex.get("output") or ""))

                valid_syntax   = is_syntactically_valid(pred_normalized)
                has_playwright = contains_playwright_calls(pred_normalized)
                exact_match    = pred_normalized == ref_normalized

                if valid_syntax:
                    syntax_ok += 1
                if has_playwright:
                    playwright_ok += 1
                if exact_match:
                    exact_matches += 1

                predictions.append({
                    "index":          i,
                    "instruction":    ex.get("instruction", ""),
                    "input":          ex.get("input", ""),
                    "reference":      ex.get("output", ""),
                    "prediction":     pred_text,
                    "valid_syntax":   valid_syntax,
                    "has_playwright": has_playwright,
                    "exact_match":    exact_match,
                })

    # ------------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------------

    n = len(predictions)

    summary = {
        "model":                 BASE_MODEL,
        "adapter":               str(ADAPTER_DIR),
        "validation_examples":   n,
        "max_new_tokens":        MAX_NEW_TOKENS,
        "eval_batch_size":       EVAL_BATCH_SIZE,
        "syntax_valid_count":    syntax_ok,
        "syntax_valid_rate":     round(syntax_ok / n, 4) if n else 0,
        "playwright_call_count": playwright_ok,
        "playwright_call_rate":  round(playwright_ok / n, 4) if n else 0,
        "exact_match_count":     exact_matches,
        "exact_match_rate":      round(exact_matches / n, 4) if n else 0,
    }

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2)

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Examples evaluated:    {n}")
    print(f"  Syntax valid:          {syntax_ok}/{n}  ({summary['syntax_valid_rate']*100:.1f}%)")
    print(f"  Contains Playwright:   {playwright_ok}/{n}  ({summary['playwright_call_rate']*100:.1f}%)")
    print(f"  Exact match:           {exact_matches}/{n}  ({summary['exact_match_rate']*100:.1f}%)")
    print("=" * 80)
    print(f"\n✓ Predictions saved to:\n  {OUTPUT_FILE}")
    print(f"✓ Summary saved to:\n  {SUMMARY_FILE}")


if __name__ == "__main__":
    main()