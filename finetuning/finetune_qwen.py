#!/usr/bin/env python3
"""
Stable LoRA fine-tuning of Qwen2.5-14B-Instruct on LUMI/ROCm.

This script:
1. Uses Qwen's official chat template.
2. Uses response-only loss masking (-100 on prompt tokens).
3. Uses BF16 on one allocated LUMI GPU/GCD.
4. Uses non-reentrant gradient checkpointing.
5. Performs a one-batch forward/backward numerical stability test
   before starting the real training.
6. Trains only LoRA adapters.
7. Evaluates validation loss during training.
8. Saves the best LoRA adapter.
9. Leaves BLEU/ROUGE/code-generation evaluation to evaluate_qwen.py.
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

PROJECT_DIR = Path(
    "/project/project_465003167/m10-testbot/finetuning"
)

TRAIN_FILE = PROJECT_DIR / "dataset" / "training_data.jsonl"
VALIDATION_FILE = PROJECT_DIR / "dataset" / "validation_data.jsonl"

# Fallback if training/validation files do not exist.
ALL_DATA_FILE = PROJECT_DIR / "dataset" / "all_data.jsonl"

OUTPUT_DIR = PROJECT_DIR / "qwen-finetuned"

FINAL_ADAPTER_DIR = PROJECT_DIR / "qwen-finetuned-final"

RESULTS_DIR = PROJECT_DIR / "training_results"


# ============================================================================
# MODEL / TRAINING CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
LEARNING_RATE = 1e-5

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8

NUM_EPOCHS = 3

MAX_SEQ_LENGTH = 2048

MAX_GRAD_NORM = 1.0

WEIGHT_DECAY = 0.0

LR_SCHEDULER = "cosine"

WARMUP_STEPS = 0.10

SEED = 42


# ============================================================================
# GLOBAL
# ============================================================================

tokenizer = None


# ============================================================================
# PRINTING
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# ENVIRONMENT CHECK
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

    print(
        f"GPU memory: "
        f"{props.total_memory / 1024**3:.2f} GiB"
    )

    if hasattr(torch.cuda, "is_bf16_supported"):

        bf16_supported = torch.cuda.is_bf16_supported()

        print(f"BF16 supported: {bf16_supported}")

        if not bf16_supported:
            raise RuntimeError(
                "BF16 is not supported by the allocated GPU."
            )

    print("✓ GPU/BF16 environment check passed")


# ============================================================================
# DATASET HELPERS
# ============================================================================

def build_user_content(
    instruction: str,
    input_text: str,
) -> str:

    instruction = str(instruction or "").strip()

    input_text = str(input_text or "").strip()

    if input_text:

        return (
            f"{instruction}\n\n"
            f"{input_text}"
        )

    return instruction


def tokenize_single_example(
    instruction: str,
    input_text: str,
    output: str,
):
    """
    Converts one dataset example into:
      input_ids
      attention_mask
      labels

    Only assistant/target tokens contribute to the loss.
    """

    user_content = build_user_content(
        instruction,
        input_text,
    )

    output = str(output or "").strip()

    if not output:
        raise ValueError(
            "Dataset example contains an empty output."
        )

    # User-only conversation.
    user_messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    # Full supervised conversation.
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

    # Qwen prompt including the assistant-generation marker.
    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Complete user + assistant conversation.
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

    response_token_count = (
        len(full_ids) - len(prompt_ids)
    )

    if response_token_count <= 0:

        raise ValueError(
            "Example contains no trainable assistant tokens."
        )

    # Ignore prompt tokens when calculating causal-LM loss.
    labels = (
        [-100] * len(prompt_ids)
        + full_ids[len(prompt_ids):]
    )

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

        result["input_ids"].append(
            item["input_ids"]
        )

        result["attention_mask"].append(
            item["attention_mask"]
        )

        result["labels"].append(
            item["labels"]
        )

    return result


def validate_raw_dataset(
    dataset,
    dataset_name: str,
) -> None:

    required_columns = {
        "instruction",
        "input",
        "output",
    }

    missing = (
        required_columns
        - set(dataset.column_names)
    )

    if missing:

        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing)}"
        )

    empty_outputs = 0

    for output in dataset["output"]:

        if not str(output or "").strip():
            empty_outputs += 1

    if empty_outputs > 0:

        raise ValueError(
            f"{dataset_name} contains "
            f"{empty_outputs} empty outputs."
        )

    print(
        f"✓ {dataset_name}: "
        f"{len(dataset)} examples"
    )

    print(
        f"✓ Required columns present"
    )

    print(
        f"✓ Empty outputs: {empty_outputs}"
    )


def load_datasets():

    if (
        TRAIN_FILE.exists()
        and VALIDATION_FILE.exists()
    ):

        print(
            "Using existing train/validation split."
        )

        train_dataset = load_dataset(
            "json",
            data_files=str(TRAIN_FILE),
        )["train"]

        validation_dataset = load_dataset(
            "json",
            data_files=str(VALIDATION_FILE),
        )["train"]

        return (
            train_dataset,
            validation_dataset,
        )

    if not ALL_DATA_FILE.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"  {TRAIN_FILE}\n"
            f"  {VALIDATION_FILE}\n"
            f"or fallback:\n"
            f"  {ALL_DATA_FILE}"
        )

    print(
        "Training/validation files not found."
    )

    print(
        f"Creating deterministic 80/20 split from:"
        f"\n  {ALL_DATA_FILE}"
    )

    full_dataset = load_dataset(
        "json",
        data_files=str(ALL_DATA_FILE),
    )["train"]

    split = full_dataset.train_test_split(
        test_size=0.20,
        seed=SEED,
    )

    train_dataset = split["train"]

    validation_dataset = split["test"]

    print(
        f"Training examples:   {len(train_dataset)}"
    )

    print(
        f"Validation examples: {len(validation_dataset)}"
    )

    return (
        train_dataset,
        validation_dataset,
    )


def tokenize_dataset(
    raw_dataset,
    dataset_name: str,
):

    print(
        f"\nTokenizing {dataset_name}..."
    )

    tokenized_dataset = raw_dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing {dataset_name}",
    )

    sequence_lengths = []

    bad_indices = []

    for i, labels in enumerate(
        tokenized_dataset["labels"]
    ):

        trainable_tokens = sum(
            1
            for token in labels
            if token != -100
        )

        if trainable_tokens <= 0:
            bad_indices.append(i)

        sequence_lengths.append(
            len(labels)
        )

    if bad_indices:

        raise RuntimeError(
            f"{dataset_name} contains examples "
            f"with no trainable labels: "
            f"{bad_indices[:10]}"
        )

    print(
        f"✓ {dataset_name} tokenized"
    )

    print(
        f"  min length:  {min(sequence_lengths)}"
    )

    print(
        f"  max length:  {max(sequence_lengths)}"
    )

    print(
        f"  mean length: {np.mean(sequence_lengths):.2f}"
    )

    return tokenized_dataset


# ============================================================================
# NUMERICAL STABILITY TEST
# ============================================================================

def run_stability_test(
    model,
    data_collator,
    tokenized_train,
):

    print_header(
        "ONE-BATCH NUMERICAL STABILITY TEST"
    )

    # Take the first training sample.
    example = tokenized_train[0]

    batch = data_collator(
        [example]
    )

    # Move tensors to LUMI GPU.
    batch = {
        key: value.to("cuda")
        for key, value in batch.items()
    }

    model.train()

    # Make sure no gradients from previous operations remain.
    model.zero_grad(
        set_to_none=True
    )

    print("Running forward pass...")

    outputs = model(
        **batch
    )

    loss = outputs.loss

    print(
        f"Initial loss: {loss.item():.8f}"
    )

    print(
        f"Loss finite: "
        f"{torch.isfinite(loss).item()}"
    )

    if not torch.isfinite(loss):

        raise RuntimeError(
            "INITIAL LOSS IS NaN/Inf.\n"
            "Training stopped before Trainer.train()."
        )

    print("Running backward pass...")

    loss.backward()

    bad_gradients = []

    checked = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        checked += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            bad_gradients.append(name)

    print(
        f"Gradient tensors checked: {checked}"
    )

    if bad_gradients:

        print(
            "\nNaN/Inf gradients detected:"
        )

        for name in bad_gradients[:20]:
            print(f"  {name}")

        raise RuntimeError(
            "NaN/Inf gradient detected "
            "during the first backward pass."
        )

    # IMPORTANT:
    # Do not leave these gradients before Trainer.train().
    model.zero_grad(
        set_to_none=True
    )

    torch.cuda.empty_cache()

    print(
        "✓ Forward loss is finite"
    )

    print(
        "✓ Backward gradients are finite"
    )

    print(
        "✓ Numerical stability test passed"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    global tokenizer

    print_header(
        "QWEN2.5-14B-INSTRUCT LoRA FINE-TUNING"
    )

    print(
        "LUMI / AMD ROCm"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    # Reproducibility.
    set_seed(SEED)

    random.seed(SEED)

    np.random.seed(SEED)

    # ------------------------------------------------------------------------
    # 1. Environment
    # ------------------------------------------------------------------------

    print_header(
        "[1/6] ENVIRONMENT"
    )

    check_environment()

    # ------------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------------

    print_header(
        "[2/6] TOKENIZER"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(
        f"Pad token: {tokenizer.pad_token!r}"
    )

    print(
        f"EOS token: {tokenizer.eos_token!r}"
    )

    print(
        "✓ Tokenizer loaded"
    )

    # ------------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------------

    print_header(
        "[3/6] MODEL"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # One allocated GCD is one visible HIP/CUDA device.
    #
    # DO NOT use device_map="auto" here.
    model = model.to("cuda")

    # Disable KV-cache while training.
    model.config.use_cache = False

    # Needed when using gradient checkpointing with frozen embeddings.
    model.enable_input_require_grads()

    print(
        "✓ Model loaded on cuda:0"
    )

    # ------------------------------------------------------------------------
    # 4. LoRA
    # ------------------------------------------------------------------------

    print_header(
        "[4/6] LoRA"
    )

    lora_config = LoraConfig(
        r=LORA_RANK,

        lora_alpha=LORA_ALPHA,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # ------------------------------------------------------------------------
    # 5. Dataset
    # ------------------------------------------------------------------------

    print_header(
        "[5/6] DATASET"
    )

    raw_train, raw_validation = load_datasets()

    validate_raw_dataset(
        raw_train,
        "Training dataset",
    )

    validate_raw_dataset(
        raw_validation,
        "Validation dataset",
    )

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

    print_header(
        "[6/6] TRAINER"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(

        output_dir=str(
            OUTPUT_DIR
        ),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_steps=WARMUP_STEPS,

        lr_scheduler_type=LR_SCHEDULER,

        bf16=True,

        fp16=False,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        max_grad_norm=MAX_GRAD_NORM,

        logging_strategy="steps",

        logging_steps=1,

        logging_first_step=True,

        # IMPORTANT:
        # Do not hide NaN/Inf losses.
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

        dataloader_num_workers=2,

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

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Max grad norm: {MAX_GRAD_NORM}"
    )

    print(
        f"Max sequence length: {MAX_SEQ_LENGTH}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    # ------------------------------------------------------------------------
    # Stability test BEFORE real training.
    # ------------------------------------------------------------------------

    run_stability_test(
        model=model,
        data_collator=data_collator,
        tokenized_train=train_dataset,
    )

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    print_header(
        "STARTING TRAINING"
    )

    train_result = trainer.train()

    training_loss = float(
        train_result.training_loss
    )

    print(
        f"\nTraining loss: {training_loss:.8f}"
    )

    if not math.isfinite(
        training_loss
    ):

        raise RuntimeError(
            f"Training ended with non-finite loss: "
            f"{training_loss}"
        )

    # ------------------------------------------------------------------------
    # Save BEST adapter
    # ------------------------------------------------------------------------

    print_header(
        "SAVING BEST LoRA ADAPTER"
    )

    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Because load_best_model_at_end=True,
    # Trainer has restored the best checkpoint.
    trainer.save_model(
        str(FINAL_ADAPTER_DIR)
    )

    tokenizer.save_pretrained(
        str(FINAL_ADAPTER_DIR)
    )

    print(
        f"✓ LoRA adapter saved to:\n"
        f"  {FINAL_ADAPTER_DIR}"
    )

    # ------------------------------------------------------------------------
    # Final validation loss
    # ------------------------------------------------------------------------

    print_header(
        "FINAL VALIDATION"
    )

    eval_metrics = trainer.evaluate()

    eval_loss = eval_metrics.get(
        "eval_loss"
    )

    if eval_loss is None:

        raise RuntimeError(
            "Trainer did not return eval_loss."
        )

    eval_loss = float(eval_loss)

    if not math.isfinite(
        eval_loss
    ):

        raise RuntimeError(
            f"FINAL VALIDATION LOSS IS NaN/Inf: "
            f"{eval_loss}"
        )

    perplexity = math.exp(
        eval_loss
    )

    print(
        f"Validation loss: {eval_loss:.8f}"
    )

    print(
        f"Perplexity:      {perplexity:.8f}"
    )

    # ------------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------------

    timestamp = datetime.now().isoformat()

    report = {
        "timestamp": timestamp,

        "model": MODEL_NAME,

        "adapter_directory": str(
            FINAL_ADAPTER_DIR
        ),

        "lora_rank": LORA_RANK,

        "lora_alpha": LORA_ALPHA,

        "lora_dropout": LORA_DROPOUT,

        "learning_rate": LEARNING_RATE,

        "batch_size": BATCH_SIZE,

        "gradient_accumulation": (
            GRADIENT_ACCUMULATION
        ),

        "effective_batch_size": (
            BATCH_SIZE
            * GRADIENT_ACCUMULATION
        ),

        "epochs": NUM_EPOCHS,

        "max_sequence_length": (
            MAX_SEQ_LENGTH
        ),

        "max_grad_norm": MAX_GRAD_NORM,

        "training_examples": len(
            train_dataset
        ),

        "validation_examples": len(
            validation_dataset
        ),

        "training_loss": training_loss,

        "validation_loss": eval_loss,

        "perplexity": perplexity,

        "evaluation_metrics": eval_metrics,

        "log_history": trainer.state.log_history,
    }

    metrics_file = (
        RESULTS_DIR
        / "final_training_evaluation.json"
    )

    with metrics_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"✓ Report saved to:\n"
        f"  {metrics_file}"
    )

    # ------------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------------

    print_header(
        "TRAINING COMPLETE"
    )

    print(
        "✓ Stability test passed"
    )

    print(
        "✓ Training loss is finite"
    )

    print(
        "✓ Validation loss is finite"
    )

    print(
        f"✓ Adapter:\n"
        f"  {FINAL_ADAPTER_DIR}"
    )

    print(
        "\nNext step:"
    )

    print(
        "Run evaluate_qwen.py separately."
    )


if __name__ == "__main__":
    main()