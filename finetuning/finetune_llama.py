#!/usr/bin/env python3
"""
LoRA fine-tuning of a Llama-family Instruct model on LUMI/ROCm.

IMPORTANT:
Meta's official Llama 3.3 release is 70B, not 8B. Therefore MODEL_NAME
below is configurable. Set it to the exact Llama 3.3 8B-compatible
Hugging Face repository you have selected.

The script keeps the same dataset format and training/evaluation strategy
used by the DeepSeek script:
  instruction / input / output

It:
  - uses the model's native chat template
  - masks prompt tokens so only assistant output contributes to loss
  - uses BF16 on the LUMI MI250X
  - uses LoRA
  - performs a one-batch forward/backward numerical stability test
  - evaluates validation loss/perplexity
  - saves the LoRA adapter and tokenizer
"""

import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path("/project/project_465003167/m10-testbot/finetuning")

TRAIN_FILE = PROJECT_DIR / "dataset" / "training_data.jsonl"
VALIDATION_FILE = PROJECT_DIR / "dataset" / "validation_data.jsonl"
ALL_DATA_FILE = PROJECT_DIR / "dataset" / "all_data.jsonl"

OUTPUT_DIR = PROJECT_DIR / "llama-finetuned"
FINAL_ADAPTER_DIR = PROJECT_DIR / "llama-finetuned-final"
RESULTS_DIR = PROJECT_DIR / "training_results"

# ============================================================================
# MODEL
# ============================================================================

# Set this to the exact Llama 3.3 8B repository you intend to use.
#
# IMPORTANT:
# There is no official Meta "Llama 3.3 8B" release. If you actually mean
# the official 8B model, use:
#   meta-llama/Llama-3.1-8B-Instruct
#
# You can override this without editing the file:
#   export LLAMA_MODEL_NAME="your-org/your-llama-3.3-8b-model"
#
MODEL_NAME = "your-org/Llama-3.3-8B-Instruct"

# ============================================================================
# LoRA / TRAINING CONFIGURATION
# ============================================================================

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATE = 5e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3

MAX_SEQ_LENGTH = 2048
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0
LR_SCHEDULER = "cosine"

# Trainer accepts warmup_steps as an integer. We calculate 10% of total
# optimizer steps after the dataset is loaded.
WARMUP_RATIO = 0.10

SEED = 42

# ============================================================================
# GLOBAL
# ============================================================================

tokenizer = None


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# ENVIRONMENT
# ============================================================================

def check_environment() -> None:
    print_header("ENVIRONMENT CHECK")

    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm/HIP: {torch.version.hip}")
    print(f"CUDA API available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. This script must run inside a LUMI GPU job."
        )

    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1024**3:.2f} GiB")

    if hasattr(torch.cuda, "is_bf16_supported"):
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"BF16 supported: {bf16_supported}")
        if not bf16_supported:
            raise RuntimeError("BF16 is not supported by the allocated GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        print("BF16 reduced-precision reduction enabled")

    print("GPU/BF16 environment check passed")


# ============================================================================
# DATASET
# ============================================================================

def build_user_content(instruction: str, input_text: str) -> str:
    instruction = str(instruction or "").strip()
    input_text = str(input_text or "").strip()

    if input_text:
        return f"{instruction}\n\n{input_text}"

    return instruction


def tokenize_single_example(
    instruction: str,
    input_text: str,
    output: str,
):
    """
    Tokenize one example using the model's native chat template.

    Prompt tokens are masked with -100, so only assistant tokens contribute
    to the causal-LM loss.
    """

    user_content = build_user_content(instruction, input_text)
    output = str(output or "").strip()

    if not output:
        raise ValueError("Dataset example contains an empty output.")

    system_message = {
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

    user_messages = [
        system_message,
        {"role": "user", "content": user_content},
    ]

    full_messages = [
        system_message,
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
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
            f"Example contains {len(full_ids)} tokens, "
            f"which exceeds MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}."
        )

    assistant_tokens = len(full_ids) - len(prompt_ids)

    if assistant_tokens <= 0:
        raise ValueError(
            "Example contains no trainable assistant tokens. "
            "Check the model chat template."
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def format_and_tokenize(examples):
    result = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        item = tokenize_single_example(
            instruction,
            input_text,
            output,
        )

        result["input_ids"].append(item["input_ids"])
        result["attention_mask"].append(item["attention_mask"])
        result["labels"].append(item["labels"])

    return result


def validate_raw_dataset(dataset, dataset_name: str) -> None:
    required_columns = {"instruction", "input", "output"}
    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}"
        )

    empty_outputs = sum(
        1 for output in dataset["output"]
        if not str(output or "").strip()
    )

    if empty_outputs:
        raise ValueError(
            f"{dataset_name} contains {empty_outputs} empty outputs."
        )

    print(f"{dataset_name}: {len(dataset)} examples")
    print("Required columns present")
    print(f"Empty outputs: {empty_outputs}")


def load_datasets():
    if TRAIN_FILE.exists() and VALIDATION_FILE.exists():
        print("Using existing train/validation split.")

        train_dataset = load_dataset(
            "json",
            data_files=str(TRAIN_FILE),
        )["train"]

        validation_dataset = load_dataset(
            "json",
            data_files=str(VALIDATION_FILE),
        )["train"]

        return train_dataset, validation_dataset

    if not ALL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n"
            f"  {TRAIN_FILE}\n"
            f"  {VALIDATION_FILE}\n"
            f"or fallback:\n"
            f"  {ALL_DATA_FILE}"
        )

    print(f"Creating deterministic 80/20 split from {ALL_DATA_FILE}")

    full_dataset = load_dataset(
        "json",
        data_files=str(ALL_DATA_FILE),
    )["train"]

    split = full_dataset.train_test_split(
        test_size=0.20,
        seed=SEED,
    )

    print(f"Training examples:   {len(split['train'])}")
    print(f"Validation examples: {len(split['test'])}")

    return split["train"], split["test"]


def tokenize_dataset(raw_dataset, dataset_name: str):
    print(f"\nTokenizing {dataset_name}...")

    tokenized = raw_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing {dataset_name}",
    )

    lengths = [len(x) for x in tokenized["labels"]]

    bad_indices = [
        i
        for i, labels in enumerate(tokenized["labels"])
        if sum(1 for token in labels if token != -100) <= 0
    ]

    if bad_indices:
        raise RuntimeError(
            f"{dataset_name} contains examples with no trainable labels: "
            f"{bad_indices[:10]}"
        )

    print(f"{dataset_name} tokenized")
    print(f"  min length:  {min(lengths)}")
    print(f"  max length:  {max(lengths)}")
    print(f"  mean length: {np.mean(lengths):.2f}")

    return tokenized


# ============================================================================
# NUMERICAL STABILITY
# ============================================================================

def run_stability_test(model, data_collator, tokenized_train):
    print_header("ONE-BATCH NUMERICAL STABILITY TEST")

    batch = data_collator([tokenized_train[0]])
    batch = {k: v.to("cuda") for k, v in batch.items()}

    model.train()
    model.zero_grad(set_to_none=True)

    print("Running forward pass...")

    outputs = model(**batch)
    loss = outputs.loss

    print(f"Initial loss: {loss.item():.8f}")
    print(f"Loss finite:  {torch.isfinite(loss).item()}")

    if not torch.isfinite(loss):
        raise RuntimeError("INITIAL LOSS IS NaN/Inf. Training aborted.")

    print("Running backward pass...")
    loss.backward()

    bad_grads = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and param.grad is not None
        and not torch.isfinite(param.grad).all()
    ]

    checked = sum(
        1
        for param in model.parameters()
        if param.requires_grad
        and param.grad is not None
    )

    print(f"Gradient tensors checked: {checked}")

    if bad_grads:
        for name in bad_grads[:20]:
            print(f"  NaN/Inf grad: {name}")

        raise RuntimeError(
            "NaN/Inf gradient detected in first backward pass."
        )

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    print("Forward loss is finite")
    print("Backward gradients are finite")
    print("Numerical stability test passed")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global tokenizer

    print_header("LLAMA LoRA FINE-TUNING")
    print("LUMI / AMD ROCm")
    print(f"Model: {MODEL_NAME}")

    if MODEL_NAME.startswith("your-org/"):
        raise RuntimeError(
            "MODEL_NAME is still a placeholder. Set LLAMA_MODEL_NAME "
            "or edit MODEL_NAME to the exact model repository."
        )

    # Allow SLURM environment override.
    import os

    model_override = os.environ.get("LLAMA_MODEL_NAME")
    if model_override:
        MODEL_NAME = model_override
        print(f"Using LLAMA_MODEL_NAME override: {MODEL_NAME}")

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------------
    # 1. Environment
    # ------------------------------------------------------------------------

    print_header("[1/6] ENVIRONMENT")
    check_environment()

    # ------------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------------

    print_header("[2/6] TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        # Llama 3.x models commonly have an EOS token but no pad token.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Vocab size:      {tokenizer.vocab_size}")
    print(f"Pad token:       {tokenizer.pad_token!r}")
    print(f"EOS token:       {tokenizer.eos_token!r}")

    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "The selected model/tokenizer does not provide a chat template. "
            "This script expects an Instruct/chat model."
        )

    smoke_test = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        tokenize=True,
        add_generation_prompt=True,
    )

    print(f"Chat template smoke-test: {len(smoke_test)} tokens")
    print("Tokenizer loaded")

    # ------------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------------

    print_header("[3/6] MODEL")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model = model.to("cuda")
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    total_params = sum(
        param.numel() for param in model.parameters()
    )

    print(f"Total model parameters: {total_params / 1e9:.2f}B")
    print("Model loaded on cuda:0")

    # ------------------------------------------------------------------------
    # 4. LoRA
    # ------------------------------------------------------------------------

    print_header("[4/6] LoRA")

    # Standard Llama attention/MLP projection names.
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------------
    # 5. Dataset
    # ------------------------------------------------------------------------

    print_header("[5/6] DATASET")

    raw_train, raw_validation = load_datasets()

    validate_raw_dataset(raw_train, "Training dataset")
    validate_raw_dataset(raw_validation, "Validation dataset")

    train_dataset = tokenize_dataset(
        raw_train,
        "Training dataset",
    )

    validation_dataset = tokenize_dataset(
        raw_validation,
        "Validation dataset",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # ------------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------------

    print_header("[6/6] TRAINER")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Estimate optimizer steps for warmup.
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(
            len(train_dataset)
            / (BATCH_SIZE * GRADIENT_ACCUMULATION)
        ),
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch * NUM_EPOCHS
    )

    warmup_steps = max(
        1,
        int(total_optimizer_steps * WARMUP_RATIO),
    )

    print(f"Optimizer steps/epoch: {optimizer_steps_per_epoch}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print(f"Warmup steps:          {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        warmup_steps=warmup_steps,
        lr_scheduler_type=LR_SCHEDULER,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=MAX_GRAD_NORM,

        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=False,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_torch",

        report_to=["tensorboard"],

        seed=SEED,
        data_seed=SEED,

        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,

        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0,
            )
        ],
    )

    print(
        f"\nEffective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )

    print(f"Learning rate:        {LEARNING_RATE}")
    print(f"Max grad norm:        {MAX_GRAD_NORM}")
    print(f"Max sequence length:  {MAX_SEQ_LENGTH}")
    print(f"Epochs:               {NUM_EPOCHS}")

    # ------------------------------------------------------------------------
    # Stability test
    # ------------------------------------------------------------------------

    run_stability_test(
        model=model,
        data_collator=data_collator,
        tokenized_train=train_dataset,
    )

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    print_header("STARTING TRAINING")

    train_result = trainer.train()
    training_loss = float(train_result.training_loss)

    print(f"\nTraining loss: {training_loss:.8f}")

    if not math.isfinite(training_loss):
        raise RuntimeError(
            f"Training ended with non-finite loss: {training_loss}"
        )

    # ------------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------------

    print_header("SAVING BEST LoRA ADAPTER")

    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))

    print(
        f"LoRA adapter saved to:\n"
        f"  {FINAL_ADAPTER_DIR}"
    )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    print_header("FINAL VALIDATION")

    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss")

    if eval_loss is None:
        raise RuntimeError("Trainer did not return eval_loss.")

    eval_loss = float(eval_loss)

    if not math.isfinite(eval_loss):
        raise RuntimeError(
            f"FINAL VALIDATION LOSS IS NaN/Inf: {eval_loss}"
        )

    perplexity = math.exp(eval_loss)

    print(f"Validation loss: {eval_loss:.8f}")
    print(f"Perplexity:      {perplexity:.8f}")

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "adapter_directory": str(FINAL_ADAPTER_DIR),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "epochs": NUM_EPOCHS,
        "max_sequence_length": MAX_SEQ_LENGTH,
        "max_grad_norm": MAX_GRAD_NORM,
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "training_loss": training_loss,
        "validation_loss": eval_loss,
        "perplexity": perplexity,
        "evaluation_metrics": eval_metrics,
        "log_history": trainer.state.log_history,
    }

    metrics_file = RESULTS_DIR / "final_llama_training_evaluation.json"

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Report saved to:\n  {metrics_file}")

    print_header("TRAINING COMPLETE")
    print("Stability test passed")
    print("Training loss is finite")
    print("Validation loss is finite")
    print(f"Adapter:\n  {FINAL_ADAPTER_DIR}")
    print("\nNext step: Run evaluate_llama.py separately.")


if __name__ == "__main__":
    main()
