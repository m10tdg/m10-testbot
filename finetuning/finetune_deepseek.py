#!/usr/bin/env python3
"""
LoRA fine-tuning of DeepSeek-Coder-V2-Lite-Instruct on LUMI/ROCm.
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

# Updated output directories
OUTPUT_DIR = PROJECT_DIR / "deepseek-finetuned"
FINAL_ADAPTER_DIR = PROJECT_DIR / "deepseek-finetuned-final"
RESULTS_DIR = PROJECT_DIR / "training_results"

# ============================================================================
# MODEL / TRAINING CONFIGURATION
# ============================================================================

# Hugging Face path for deepseek-coder-v2:16b / Lite
MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
LEARNING_RATE = 5e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0
LR_SCHEDULER = "cosine"
WARMUP_STEPS = 0.10
SEED = 42

tokenizer = None


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_environment() -> None:
    print_header("ENVIRONMENT CHECK")
    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm/HIP: {torch.version.hip}")
    print(f"CUDA API available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Run inside a LUMI GPU job.")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU 0: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.2f} GiB)")

    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU.")


def build_user_content(instruction: str, input_text: str) -> str:
    instruction = str(instruction or "").strip()
    input_text = str(input_text or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def tokenize_single_example(instruction: str, input_text: str, output: str):
    user_content = build_user_content(instruction, input_text)
    output = str(output or "").strip()

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
        )
    }

    if not output:
        raise ValueError("Dataset example contains an empty output.")

    user_messages = [system_message, {"role": "user", "content": user_content}]
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

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Example contains {len(full_ids)} tokens, exceeding MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}."
        )

    if len(full_ids) - len(prompt_ids) <= 0:
        raise ValueError("Example contains no trainable assistant tokens.")

    # Mask non-assistant/prompt tokens
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def format_and_tokenize(examples):
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        item = tokenize_single_example(inst, inp, out)
        result["input_ids"].append(item["input_ids"])
        result["attention_mask"].append(item["attention_mask"])
        result["labels"].append(item["labels"])
    return result


def load_datasets():
    if TRAIN_FILE.exists() and VALIDATION_FILE.exists():
        train_ds = load_dataset("json", data_files=str(TRAIN_FILE))["train"]
        val_ds = load_dataset("json", data_files=str(VALIDATION_FILE))["train"]
        return train_ds, val_ds

    full_ds = load_dataset("json", data_files=str(ALL_DATA_FILE))["train"]
    split = full_ds.train_test_split(test_size=0.20, seed=SEED)
    return split["train"], split["test"]


def main():
    global tokenizer
    print_header("DEEPSEEK-CODER-V2 LORA FINE-TUNING")

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    check_environment()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA config covering Attention & MLP MoE Projections
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    raw_train, raw_val = load_datasets()
    train_ds = raw_train.map(format_and_tokenize, batched=True, remove_columns=raw_train.column_names)
    val_ds = raw_val.map(format_and_tokenize, batched=True, remove_columns=raw_val.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # Trainer Setup
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        report_to=["tensorboard"],
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Start training
    trainer.train()

    # Save
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print(f"Saved best adapter to {FINAL_ADAPTER_DIR}")


if __name__ == "__main__":
    main()