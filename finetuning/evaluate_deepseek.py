#!/usr/bin/env python3
"""
Evaluation of the fine-tuned DeepSeek-Coder-V2-Lite-Instruct LoRA adapter.

Mirrors evaluate_qwen.py but adapted for DeepSeek:
- Applies the get_imports patch (same as finetune_deepseek.py) so the model
  loads without flash_attn installed.
- Uses attn_implementation="eager" (only supported value in transformers 4.41.2
  for DeepseekV2ForCausalLM).
- Loads the tokenizer from the adapter directory (which contains a saved copy).
- Generates predictions on the full validation set with greedy decoding.
- Saves per-example predictions + a summary with pass-rate and exact-match stats.
"""

import sys
import json
import re
from pathlib import Path

# ============================================================================
# FLASH-ATTN IMPORT PATCH  (identical to finetune_deepseek.py)
# Must run before any transformers import.
# ============================================================================

import transformers.dynamic_module_utils as _dmu

_orig_get_imports = _dmu.get_imports


def _patched_get_imports(filename):
    imports = _orig_get_imports(filename)
    if str(filename).endswith("modeling_deepseek.py"):
        imports = [imp for imp in imports if imp != "flash_attn"]
    return imports


_dmu.get_imports = _patched_get_imports

# ============================================================================
# NOW safe to import everything else
# ============================================================================

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path("/project/project_465003167/m10-testbot/finetuning")

BASE_MODEL    = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
ADAPTER_DIR   = PROJECT_DIR / "deepseek-finetuned-final"
VAL_FILE      = PROJECT_DIR / "dataset" / "validation_data.jsonl"
OUTPUT_DIR    = PROJECT_DIR / "training_results"
OUTPUT_FILE   = OUTPUT_DIR / "evaluation_deepseek_finetuned.json"
SUMMARY_FILE  = OUTPUT_DIR / "evaluation_deepseek_summary.json"

# Maximum new tokens to generate per example.
# 512 covers all outputs in the training set (max length ~773 total tokens,
# prompt is ~420, so response headroom is ~350 — 512 is a safe ceiling).
MAX_NEW_TOKENS = 512


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
    # Remove markdown code fences if the model accidentally emits them.
    text = re.sub(r"^```(?:python)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    # Collapse runs of blank lines to a single blank line.
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


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("DEEPSEEK-CODER-V2-LITE-INSTRUCT  —  FINE-TUNED EVALUATION")
    print("=" * 80)
    print(f"Base model:   {BASE_MODEL}")
    print(f"Adapter:      {ADAPTER_DIR}")
    print(f"Val file:     {VAL_FILE}")
    print(f"Output:       {OUTPUT_FILE}")

    # ------------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------------

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found:\n  {ADAPTER_DIR}\n"
            "Run finetune_deepseek.py first."
        )

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n  {VAL_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # GPU check
    # ------------------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Run inside a LUMI GPU job.")

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
    )

    # ------------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------------

    print("\nLoading tokenizer from adapter directory...")

    # The adapter directory contains a saved copy of the tokenizer
    # (tokenizer.save_pretrained() was called at the end of fine-tuning).
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"   # Left-pad for generation (not training).

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token:  {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print("✓ Tokenizer loaded")

    # ------------------------------------------------------------------------
    # Base model + LoRA adapter
    # ------------------------------------------------------------------------

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params / 1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print("✓ Model ready")

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    print("\nLoading validation dataset...")

    dataset = load_dataset("json", data_files=str(VAL_FILE))["train"]

    print(f"✓ {len(dataset)} validation examples")

    # ------------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------------

    predictions = []

    syntax_ok      = 0
    playwright_ok  = 0
    exact_matches  = 0

    print(f"\nGenerating predictions (max_new_tokens={MAX_NEW_TOKENS})...\n")

    with torch.inference_mode():

        for i, example in enumerate(
            tqdm(dataset, desc="Evaluating", unit="example")
        ):

            # Build prompt (same format as training).
            user_content = str(example.get("instruction") or "").strip()
            input_text   = str(example.get("input")       or "").strip()

            if input_text:
                user_content = f"{user_content}\n\n{input_text}"

            messages = [
                SYSTEM_MESSAGE,
                {"role": "user", "content": user_content},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda")

            # Greedy decoding — deterministic, matches evaluate_qwen.py.
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Slice off the prompt tokens to get only the generated response.
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            pred_text  = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # Per-example quality signals.
            pred_normalized = normalize_code(pred_text)
            ref_normalized  = normalize_code(str(example.get("output") or ""))

            valid_syntax  = is_syntactically_valid(pred_normalized)
            has_playwright = contains_playwright_calls(pred_normalized)
            exact_match   = pred_normalized == ref_normalized

            if valid_syntax:
                syntax_ok += 1
            if has_playwright:
                playwright_ok += 1
            if exact_match:
                exact_matches += 1

            predictions.append({
                "index":            i,
                "instruction":      example.get("instruction", ""),
                "input":            example.get("input", ""),
                "reference":        example.get("output", ""),
                "prediction":       pred_text,
                "valid_syntax":     valid_syntax,
                "has_playwright":   has_playwright,
                "exact_match":      exact_match,
            })

    # ------------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------------

    n = len(predictions)

    summary = {
        "model":                  BASE_MODEL,
        "adapter":                str(ADAPTER_DIR),
        "validation_examples":    n,
        "max_new_tokens":         MAX_NEW_TOKENS,
        "syntax_valid_count":     syntax_ok,
        "syntax_valid_rate":      round(syntax_ok / n, 4) if n else 0,
        "playwright_call_count":  playwright_ok,
        "playwright_call_rate":   round(playwright_ok / n, 4) if n else 0,
        "exact_match_count":      exact_matches,
        "exact_match_rate":       round(exact_matches / n, 4) if n else 0,
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