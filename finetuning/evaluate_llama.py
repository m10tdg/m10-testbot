#!/usr/bin/env python3
"""
Evaluation of a fine-tuned Llama-family Instruct LoRA adapter.

Uses the same validation dataset and task metrics as the DeepSeek evaluator:
  - Python syntax validity
  - Playwright-call presence
  - exact match against reference output

The base model is loaded in BF16 and the saved LoRA adapter is attached.

IMPORTANT:
BASE_MODEL must be the exact same repository used in finetune_llama.py
(i.e. the same value you passed as LLAMA_MODEL_NAME during training),
otherwise the LoRA adapter's target modules / tokenizer will not line up
with the base weights.
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

PROJECT_DIR = Path(
    "/project/project_465003167/m10-testbot/finetuning"
)

# This is a PLACEHOLDER value. Override with:
#   export LLAMA_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# It must match whatever LLAMA_MODEL_NAME was set to during training.
_PLACEHOLDER_BASE_MODEL = "your-org/Llama-3.1-8B-Instruct"
BASE_MODEL = os.environ.get(
    "LLAMA_MODEL_NAME",
    _PLACEHOLDER_BASE_MODEL,
)

ADAPTER_DIR = PROJECT_DIR / "llama-finetuned-final"
VAL_FILE = PROJECT_DIR / "dataset" / "validation_data.jsonl"
OUTPUT_DIR = PROJECT_DIR / "training_results"

OUTPUT_FILE = (
    OUTPUT_DIR / "evaluation_llama_finetuned.json"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "evaluation_llama_summary.json"
)

MAX_NEW_TOKENS = 512


# ============================================================================
# SYSTEM PROMPT
# Must remain identical to training.
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
    Normalize generated code for exact-match comparison.
    """

    text = re.sub(
        r"^```(?:python)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text.strip(),
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def is_syntactically_valid(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def contains_playwright_calls(code: str) -> bool:
    playwright_patterns = [
        r"page\.",
        r"playwright",
        r"async_playwright",
        r"sync_playwright",
    ]

    return any(
        re.search(pattern, code, re.IGNORECASE)
        for pattern in playwright_patterns
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("LLAMA — FINE-TUNED EVALUATION")
    print("=" * 80)

    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter:    {ADAPTER_DIR}")
    print(f"Val file:   {VAL_FILE}")
    print(f"Output:     {OUTPUT_FILE}")

    if BASE_MODEL == _PLACEHOLDER_BASE_MODEL:
        raise RuntimeError(
            "BASE_MODEL is still a placeholder. Set the LLAMA_MODEL_NAME "
            "environment variable to the exact model repository used "
            "for fine-tuning, e.g. "
            "'export LLAMA_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct'."
        )

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found:\n  {ADAPTER_DIR}\n"
            "Run finetune_llama.py first."
        )

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n  {VAL_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. Run inside a LUMI GPU job."
        )

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
    )

    # ------------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------------

    print("\nLoading tokenizer...")

    # The fine-tuning script saves the tokenizer in the adapter directory.
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "Tokenizer does not contain a chat template."
        )

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token:  {tokenizer.pad_token!r}")
    print(f"EOS token:  {tokenizer.eos_token!r}")
    print("Tokenizer loaded")

    # ------------------------------------------------------------------------
    # Base model + adapter
    # ------------------------------------------------------------------------

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        str(ADAPTER_DIR),
    )

    model.eval()

    total_params = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    print(
        f"Total parameters:     "
        f"{total_params / 1e9:.2f}B"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params / 1e6:.2f}M"
    )

    print("Model ready")

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    print("\nLoading validation dataset...")

    dataset = load_dataset(
        "json",
        data_files=str(VAL_FILE),
    )["train"]

    print(
        f"{len(dataset)} validation examples"
    )

    # ------------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------------

    predictions = []

    syntax_ok = 0
    playwright_ok = 0
    exact_matches = 0

    print(
        f"\nGenerating predictions "
        f"(max_new_tokens={MAX_NEW_TOKENS})...\n"
    )

    with torch.inference_mode():

        for i, example in enumerate(
            tqdm(
                dataset,
                desc="Evaluating",
                unit="example",
            )
        ):

            instruction = str(
                example.get("instruction") or ""
            ).strip()

            input_text = str(
                example.get("input") or ""
            ).strip()

            if input_text:
                user_content = (
                    f"{instruction}\n\n{input_text}"
                )
            else:
                user_content = instruction

            messages = [
                SYSTEM_MESSAGE,
                {
                    "role": "user",
                    "content": user_content,
                },
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

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_tokens = outputs[0][
                inputs["input_ids"].shape[1]:
            ]

            pred_text = tokenizer.decode(
                gen_tokens,
                skip_special_tokens=True,
            )

            pred_normalized = normalize_code(
                pred_text
            )

            ref_normalized = normalize_code(
                str(example.get("output") or "")
            )

            valid_syntax = is_syntactically_valid(
                pred_normalized
            )

            has_playwright = contains_playwright_calls(
                pred_normalized
            )

            exact_match = (
                pred_normalized == ref_normalized
            )

            if valid_syntax:
                syntax_ok += 1

            if has_playwright:
                playwright_ok += 1

            if exact_match:
                exact_matches += 1

            predictions.append({
                "index": i,
                "instruction": example.get(
                    "instruction",
                    "",
                ),
                "input": example.get(
                    "input",
                    "",
                ),
                "reference": example.get(
                    "output",
                    "",
                ),
                "prediction": pred_text,
                "valid_syntax": valid_syntax,
                "has_playwright": has_playwright,
                "exact_match": exact_match,
            })

    # ------------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------------

    n = len(predictions)

    summary = {
        "model": BASE_MODEL,
        "adapter": str(ADAPTER_DIR),
        "validation_examples": n,
        "max_new_tokens": MAX_NEW_TOKENS,

        "syntax_valid_count": syntax_ok,
        "syntax_valid_rate": (
            round(syntax_ok / n, 4)
            if n
            else 0
        ),

        "playwright_call_count": playwright_ok,
        "playwright_call_rate": (
            round(playwright_ok / n, 4)
            if n
            else 0
        ),

        "exact_match_count": exact_matches,
        "exact_match_rate": (
            round(exact_matches / n, 4)
            if n
            else 0
        ),
    }

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "predictions": predictions,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"  Examples evaluated:  "
        f"{n}"
    )

    print(
        f"  Syntax valid:        "
        f"{syntax_ok}/{n} "
        f"({summary['syntax_valid_rate'] * 100:.1f}%)"
    )

    print(
        f"  Contains Playwright: "
        f"{playwright_ok}/{n} "
        f"({summary['playwright_call_rate'] * 100:.1f}%)"
    )

    print(
        f"  Exact match:         "
        f"{exact_matches}/{n} "
        f"({summary['exact_match_rate'] * 100:.1f}%)"
    )

    print("=" * 80)

    print(
        f"\nPredictions saved to:\n"
        f"  {OUTPUT_FILE}"
    )

    print(
        f"Summary saved to:\n"
        f"  {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()