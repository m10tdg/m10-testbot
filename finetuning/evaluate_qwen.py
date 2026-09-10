#!/usr/bin/env python3

"""
Standalone evaluation of Qwen2.5-14B-Instruct + LoRA.

Metrics:
    - Validation loss
    - Perplexity
    - BLEU
    - ROUGE-L
    - Exact match
    - Whitespace-token positional accuracy
    - Python syntax validity

Optional:
    --compare_base

This evaluates the same validation dataset using:

    BASE:
        Qwen/Qwen2.5-14B-Instruct

    FINE-TUNED:
        Qwen/Qwen2.5-14B-Instruct
        +
        qwen-finetuned-final/

Generation uses Qwen's official chat template.
"""


import argparse
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset

from nltk.translate.bleu_score import (
    SmoothingFunction,
    sentence_bleu,
)

from peft import PeftModel

from rouge_score import rouge_scorer

from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path(
    "/project/project_465003167/m10-testbot/finetuning"
)

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

ADAPTER_DIR = (
    PROJECT_DIR
    / "qwen-finetuned-final"
)

VALIDATION_FILE = (
    PROJECT_DIR
    / "dataset"
    / "validation_data.jsonl"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "training_results"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_SEQ_LENGTH = 2048

MAX_NEW_TOKENS = 1024

SEED = 42


# ============================================================================
# DATASET
# ============================================================================

def load_validation_dataset():

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    dataset = load_dataset(
        "json",
        data_files=str(
            VALIDATION_FILE
        ),
    )["train"]

    return dataset


def build_user_content(
    instruction: str,
    input_text: str,
) -> str:

    instruction = str(
        instruction or ""
    ).strip()

    input_text = str(
        input_text or ""
    ).strip()

    if input_text:

        return (
            f"{instruction}\n\n"
            f"{input_text}"
        )

    return instruction


# ============================================================================
# VALIDATION LOSS
# ============================================================================

def prepare_validation_examples(
    raw_dataset,
    tokenizer,
):

    items = []

    for example in raw_dataset:

        user_content = build_user_content(
            example["instruction"],
            example["input"],
        )

        output = str(
            example["output"] or ""
        ).strip()

        user_messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": output,
            }
        ]

        prompt_text = (
            tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        full_text = (
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        if len(full_ids) > MAX_SEQ_LENGTH:

            raise ValueError(
                f"Validation example exceeds "
                f"{MAX_SEQ_LENGTH} tokens."
            )

        labels = (
            [-100] * len(prompt_ids)
            + full_ids[len(prompt_ids):]
        )

        items.append(
            {
                "input_ids": full_ids,

                "attention_mask": [
                    1
                ] * len(full_ids),

                "labels": labels,
            }
        )

    return items


def calculate_validation_loss(
    model,
    tokenizer,
    raw_dataset,
):

    tokenized = (
        prepare_validation_examples(
            raw_dataset,
            tokenizer,
        )
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    loader = DataLoader(
        tokenized,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    losses = []

    model.eval()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        batch = {
            key: value.to("cuda")
            for key, value in batch.items()
        }

        with torch.inference_mode():

            outputs = model(
                **batch
            )

        loss = outputs.loss

        if not torch.isfinite(loss):

            return float("nan")

        losses.append(
            float(loss.item())
        )

    if not losses:

        return float("nan")

    return float(
        np.mean(losses)
    )


# ============================================================================
# TEXT METRICS
# ============================================================================

def calculate_bleu(
    reference: str,
    candidate: str,
) -> float:

    reference_tokens = reference.split()

    candidate_tokens = candidate.split()

    if (
        not reference_tokens
        or not candidate_tokens
    ):

        return 0.0

    score = sentence_bleu(

        [reference_tokens],

        candidate_tokens,

        weights=(
            0.25,
            0.25,
            0.25,
            0.25,
        ),

        smoothing_function=(
            SmoothingFunction().method1
        ),
    )

    return float(
        score * 100.0
    )


def calculate_rouge_l(
    reference: str,
    candidate: str,
) -> float:

    if (
        not reference
        or not candidate
    ):

        return 0.0

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
    )

    score = scorer.score(
        reference,
        candidate,
    )

    return float(
        score["rougeL"].fmeasure
        * 100.0
    )


def calculate_exact_match(
    reference: str,
    candidate: str,
) -> float:

    return (
        100.0
        if reference.strip()
        == candidate.strip()
        else 0.0
    )


def calculate_token_accuracy(
    reference: str,
    candidate: str,
) -> float:
    """
    Positional whitespace-token accuracy.

    This metric is intentionally retained for comparison
    with the original experiment, but it is NOT a semantic
    code-quality metric.
    """

    reference_tokens = reference.split()

    candidate_tokens = candidate.split()

    if not reference_tokens:
        return 0.0

    if not candidate_tokens:
        return 0.0

    matches = 0

    for i in range(
        min(
            len(reference_tokens),
            len(candidate_tokens),
        )
    ):

        if (
            reference_tokens[i]
            == candidate_tokens[i]
        ):

            matches += 1

    return float(
        matches
        / len(reference_tokens)
        * 100.0
    )


# ============================================================================
# CODE METRICS
# ============================================================================

def normalize_code(
    text: str,
) -> str:
    """
    Removes common Markdown code fences.
    """

    text = text.strip()

    fenced = re.search(

        r"```(?:python|py)?\s*(.*?)```",

        text,

        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    if fenced:

        return fenced.group(1).strip()

    return text


def valid_python(
    code: str,
) -> bool:

    code = normalize_code(
        code
    )

    if not code.strip():

        return False

    try:

        ast.parse(code)

        return True

    except SyntaxError:

        return False


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    model = (
        AutoModelForCausalLM.from_pretrained(

            MODEL_NAME,

            torch_dtype=torch.bfloat16,

            low_cpu_mem_usage=True,

            trust_remote_code=True,
        )
        .to("cuda")
    )

    model.config.use_cache = True

    model.eval()

    return (
        tokenizer,
        model,
    )


def load_finetuned_model():

    if not ADAPTER_DIR.exists():

        raise FileNotFoundError(
            f"LoRA adapter not found:\n"
            f"{ADAPTER_DIR}"
        )

    tokenizer, base_model = (
        load_base_model()
    )

    model = PeftModel.from_pretrained(

        base_model,

        str(ADAPTER_DIR),

        is_trainable=False,
    )

    # Merge LoRA into the model for evaluation.
    model = model.merge_and_unload()

    model.eval()

    return (
        tokenizer,
        model,
    )


# ============================================================================
# GENERATION
# ============================================================================

def generate_candidate(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
) -> str:

    user_content = build_user_content(
        instruction,
        input_text,
    )

    messages = [

        {
            "role": "user",
            "content": user_content,
        }

    ]

    inputs = tokenizer.apply_chat_template(

        messages,

        tokenize=True,

        add_generation_prompt=True,

        return_dict=True,

        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.inference_mode():

        generated = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            num_beams=1,

            eos_token_id=(
                tokenizer.eos_token_id
            ),

            pad_token_id=(
                tokenizer.pad_token_id
            ),

            use_cache=True,
        )

    # IMPORTANT:
    # Keep only newly generated tokens.
    generated_tokens = (
        generated[
            0,
            input_length:
        ]
    )

    candidate = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True,
    ).strip()

    return candidate


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model,
    tokenizer,
    raw_dataset,
    model_label: str,
    output_file: Path,
):

    bleu_scores = []

    rouge_scores = []

    exact_scores = []

    token_accuracy_scores = []

    syntax_scores = []

    predictions = []

    total = len(
        raw_dataset
    )

    for index, example in enumerate(
        raw_dataset,
        start=1,
    ):

        reference = str(
            example["output"] or ""
        ).strip()

        candidate = generate_candidate(

            model=model,

            tokenizer=tokenizer,

            instruction=example[
                "instruction"
            ],

            input_text=example[
                "input"
            ],
        )

        bleu = calculate_bleu(
            reference,
            candidate,
        )

        rouge = calculate_rouge_l(
            reference,
            candidate,
        )

        exact = calculate_exact_match(
            reference,
            candidate,
        )

        token_accuracy = (
            calculate_token_accuracy(
                reference,
                candidate,
            )
        )

        syntax_ok = valid_python(
            candidate
        )

        bleu_scores.append(bleu)

        rouge_scores.append(rouge)

        exact_scores.append(exact)

        token_accuracy_scores.append(
            token_accuracy
        )

        syntax_scores.append(
            100.0
            if syntax_ok
            else 0.0
        )

        predictions.append(

            {
                "index": index - 1,

                "instruction": (
                    example["instruction"]
                ),

                "input": (
                    example["input"]
                ),

                "reference": reference,

                "prediction": candidate,

                "bleu": bleu,

                "rouge_l": rouge,

                "exact_match": exact,

                "token_accuracy": (
                    token_accuracy
                ),

                "python_syntax_valid": (
                    syntax_ok
                ),
            }
        )

        print(
            f"[{index}/{total}] "
            f"BLEU={bleu:.2f} "
            f"ROUGE-L={rouge:.2f} "
            f"Syntax={syntax_ok}"
        )

    validation_loss = (
        calculate_validation_loss(
            model=model,
            tokenizer=tokenizer,
            raw_dataset=raw_dataset,
        )
    )

    perplexity = (

        math.exp(validation_loss)

        if math.isfinite(
            validation_loss
        )

        else float("nan")
    )

    metrics = {

        "model": model_label,

        "examples": total,

        "validation_loss": (
            validation_loss
        ),

        "perplexity": (
            perplexity
        ),

        "bleu": float(
            np.mean(
                bleu_scores
            )
        ),

        "rouge_l": float(
            np.mean(
                rouge_scores
            )
        ),

        "exact_match": float(
            np.mean(
                exact_scores
            )
        ),

        "token_accuracy": float(
            np.mean(
                token_accuracy_scores
            )
        ),

        "python_syntax_validity": float(
            np.mean(
                syntax_scores
            )
        ),
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            {
                "metrics": metrics,

                "predictions": predictions,
            },

            file,

            indent=2,

            ensure_ascii=False,
        )

    return metrics


# ============================================================================
# PRINT
# ============================================================================

def print_metrics(
    title: str,
    metrics: dict,
):

    print("\n" + "=" * 80)

    print(title)

    print("=" * 80)

    print(
        f"Validation Loss:      "
        f"{metrics['validation_loss']:.6f}"
    )

    print(
        f"Perplexity:           "
        f"{metrics['perplexity']:.6f}"
    )

    print(
        f"BLEU:                 "
        f"{metrics['bleu']:.2f} / 100"
    )

    print(
        f"ROUGE-L:              "
        f"{metrics['rouge_l']:.2f} / 100"
    )

    print(
        f"Exact Match:          "
        f"{metrics['exact_match']:.2f}%"
    )

    print(
        f"Token Accuracy:       "
        f"{metrics['token_accuracy']:.2f}%"
    )

    print(
        f"Python Syntax Valid:  "
        f"{metrics['python_syntax_validity']:.2f}%"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--limit",

        type=int,

        default=None,

        help=(
            "Evaluate only the first N "
            "validation examples."
        ),
    )

    parser.add_argument(

        "--compare_base",

        action="store_true",

        help=(
            "Evaluate both base and "
            "fine-tuned models."
        ),
    )

    args = parser.parse_args()

    torch.manual_seed(SEED)

    np.random.seed(SEED)

    print(
        "=" * 80
    )

    print(
        "QWEN2.5-14B-INSTRUCT EVALUATION"
    )

    print(
        "=" * 80
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "No GPU available."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    raw_dataset = (
        load_validation_dataset()
    )

    if args.limit is not None:

        if args.limit <= 0:

            raise ValueError(
                "--limit must be greater than 0."
            )

        limit = min(
            args.limit,
            len(raw_dataset),
        )

        raw_dataset = (
            raw_dataset.select(
                range(limit)
            )
        )

    print(
        f"Validation examples: "
        f"{len(raw_dataset)}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    # ========================================================================
    # BASE MODEL
    # ========================================================================

    if args.compare_base:

        print(
            "\nLoading BASE model..."
        )

        tokenizer, model = (
            load_base_model()
        )

        base_output = (
            RESULTS_DIR
            / "evaluation_base.json"
        )

        base_metrics = evaluate_model(

            model=model,

            tokenizer=tokenizer,

            raw_dataset=raw_dataset,

            model_label=MODEL_NAME,

            output_file=base_output,
        )

        results["base"] = (
            base_metrics
        )

        print_metrics(
            "BASE MODEL",
            base_metrics,
        )

        del model

        del tokenizer

        torch.cuda.empty_cache()

    # ========================================================================
    # FINE-TUNED MODEL
    # ========================================================================

    print(
        "\nLoading FINE-TUNED model..."
    )

    tokenizer, model = (
        load_finetuned_model()
    )

    fine_output = (
        RESULTS_DIR
        / "evaluation_finetuned.json"
    )

    fine_metrics = evaluate_model(

        model=model,

        tokenizer=tokenizer,

        raw_dataset=raw_dataset,

        model_label=(
            MODEL_NAME
            + " + LoRA"
        ),

        output_file=fine_output,
    )

    results["finetuned"] = (
        fine_metrics
    )

    print_metrics(
        "FINE-TUNED MODEL",
        fine_metrics,
    )

    # ========================================================================
    # COMPARISON
    # ========================================================================

    if args.compare_base:

        base = results["base"]

        fine = results["finetuned"]

        print(
            "\n"
            + "=" * 80
        )

        print(
            "BASE VS FINE-TUNED"
        )

        print(
            "=" * 80
        )

        metrics_to_compare = [

            "validation_loss",

            "perplexity",

            "bleu",

            "rouge_l",

            "exact_match",

            "token_accuracy",

            "python_syntax_validity",

        ]

        comparison = {}

        for metric in (
            metrics_to_compare
        ):

            base_value = float(
                base[metric]
            )

            fine_value = float(
                fine[metric]
            )

            delta = (
                fine_value
                - base_value
            )

            comparison[metric] = {

                "base": base_value,

                "finetuned": fine_value,

                "delta": delta,
            }

            print(

                f"{metric:25s} "
                f"BASE={base_value:.6f} "
                f"FINE-TUNED={fine_value:.6f} "
                f"DELTA={delta:+.6f}"

            )

        comparison_file = (

            RESULTS_DIR
            / "base_vs_finetuned.json"

        )

        with comparison_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(

                comparison,

                file,

                indent=2,

            )

        print(
            "\n✓ Comparison saved to:"
        )

        print(
            comparison_file
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Fine-tuned results:\n"
        f"{fine_output}"
    )


if __name__ == "__main__":

    main()