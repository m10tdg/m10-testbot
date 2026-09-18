#!/usr/bin/env python3
"""
Stable LoRA fine-tuning of DeepSeek-Coder-V2-Lite-Instruct on LUMI/ROCm.

Fixes applied vs previous version:
- attn_implementation="eager"  (sdpa not supported by DeepseekV2ForCausalLM
  in transformers 4.41.2; the SDPA path raises ValueError at load time)
- get_imports patch to strip flash_attn from DeepSeek's static import list
  (check_imports runs before from_pretrained sees attn_implementation)
- dataloader_num_workers=0 + dataloader_prefetch_factor=None to avoid
  tokenizer-parallelism fork warnings that stall worker processes
- torch.backends.cuda.matmul.allow_tf32 / allow_bf16_reduced_precision_reduction
  enabled for faster BF16 GEMMs on MI250X
"""

import sys
import types

# ============================================================================
# FLASH-ATTN IMPORT PATCH  (must run before any transformers import)
#
# Transformers' check_imports() statically scans DeepSeek's modeling file
# and raises ImportError if flash_attn is listed but not installed — this
# happens BEFORE from_pretrained() sees attn_implementation="eager".
#
# We monkey-patch get_imports to strip "flash_attn" from the dependency list
# of modeling_deepseek.py so the check passes cleanly.
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

TRAIN_FILE      = PROJECT_DIR / "dataset" / "training_data.jsonl"
VALIDATION_FILE = PROJECT_DIR / "dataset" / "validation_data.jsonl"
ALL_DATA_FILE   = PROJECT_DIR / "dataset" / "all_data.jsonl"

OUTPUT_DIR        = PROJECT_DIR / "deepseek-finetuned"
FINAL_ADAPTER_DIR = PROJECT_DIR / "deepseek-finetuned-final"
RESULTS_DIR       = PROJECT_DIR / "training_results"


# ============================================================================
# MODEL / TRAINING CONFIGURATION
# ============================================================================

MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

LEARNING_RATE          = 5e-6
BATCH_SIZE             = 1
GRADIENT_ACCUMULATION  = 8
NUM_EPOCHS             = 3
MAX_SEQ_LENGTH         = 2048
MAX_GRAD_NORM          = 1.0
WEIGHT_DECAY           = 0.0
LR_SCHEDULER           = "cosine"
WARMUP_STEPS           = 0.10
SEED                   = 42


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
    print(f"GPU memory: {props.total_memory / 1024**3:.2f} GiB")

    if hasattr(torch.cuda, "is_bf16_supported"):
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"BF16 supported: {bf16_supported}")
        if not bf16_supported:
            raise RuntimeError("BF16 is not supported by the allocated GPU.")

    # -----------------------------------------------------------------------
    # MI250X performance knobs
    # -----------------------------------------------------------------------
    # Allow TF32 on matrix multiplications (no-op on MI250X but harmless).
    torch.backends.cuda.matmul.allow_tf32 = True

    # Key speedup on MI250X: fused BF16 reduction in GEMM kernels.
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        print("✓ BF16 reduced-precision reduction enabled")

    print("✓ GPU/BF16 environment check passed")


# ============================================================================
# DATASET HELPERS
# ============================================================================

def build_user_content(instruction: str, input_text: str) -> str:
    instruction = str(instruction or "").strip()
    input_text  = str(input_text  or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def tokenize_single_example(instruction: str, input_text: str, output: str):
    """
    Returns input_ids / attention_mask / labels with prompt tokens masked
    (-100) so only assistant tokens contribute to the loss.
    """

    user_content = build_user_content(instruction, input_text)
    output       = str(output or "").strip()

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

    if not output:
        raise ValueError("Dataset example contains an empty output.")

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
        user_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]

    if len(full_ids) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Example contains {len(full_ids)} tokens, "
            f"which exceeds MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}."
        )

    if len(full_ids) - len(prompt_ids) <= 0:
        raise ValueError("Example contains no trainable assistant tokens.")

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


def format_and_tokenize(examples):
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        item = tokenize_single_example(instruction, input_text, output)
        result["input_ids"].append(item["input_ids"])
        result["attention_mask"].append(item["attention_mask"])
        result["labels"].append(item["labels"])
    return result


def validate_raw_dataset(dataset, dataset_name: str) -> None:
    required_columns = {"instruction", "input", "output"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {sorted(missing)}")

    empty_outputs = sum(
        1 for o in dataset["output"] if not str(o or "").strip()
    )
    if empty_outputs > 0:
        raise ValueError(f"{dataset_name} contains {empty_outputs} empty outputs.")

    print(f"✓ {dataset_name}: {len(dataset)} examples")
    print(f"✓ Required columns present")
    print(f"✓ Empty outputs: {empty_outputs}")


def load_datasets():
    if TRAIN_FILE.exists() and VALIDATION_FILE.exists():
        print("Using existing train/validation split.")
        train_dataset = load_dataset("json", data_files=str(TRAIN_FILE))["train"]
        validation_dataset = load_dataset("json", data_files=str(VALIDATION_FILE))["train"]
        return train_dataset, validation_dataset

    if not ALL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n  {TRAIN_FILE}\n  {VALIDATION_FILE}\n"
            f"or fallback:\n  {ALL_DATA_FILE}"
        )

    print(f"Creating deterministic 80/20 split from:\n  {ALL_DATA_FILE}")
    full_dataset = load_dataset("json", data_files=str(ALL_DATA_FILE))["train"]
    split = full_dataset.train_test_split(test_size=0.20, seed=SEED)
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

    lengths     = [len(lbl) for lbl in tokenized["labels"]]
    bad_indices = [
        i for i, lbl in enumerate(tokenized["labels"])
        if sum(1 for t in lbl if t != -100) <= 0
    ]

    if bad_indices:
        raise RuntimeError(
            f"{dataset_name} contains examples with no trainable labels: "
            f"{bad_indices[:10]}"
        )

    print(f"✓ {dataset_name} tokenized")
    print(f"  min length:  {min(lengths)}")
    print(f"  max length:  {max(lengths)}")
    print(f"  mean length: {np.mean(lengths):.2f}")

    return tokenized


# ============================================================================
# NUMERICAL STABILITY TEST
# ============================================================================

def run_stability_test(model, data_collator, tokenized_train):
    print_header("ONE-BATCH NUMERICAL STABILITY TEST")

    batch = data_collator([tokenized_train[0]])
    batch = {k: v.to("cuda") for k, v in batch.items()}

    model.train()
    model.zero_grad(set_to_none=True)

    print("Running forward pass...")
    outputs = model(**batch)
    loss    = outputs.loss
    print(f"Initial loss: {loss.item():.8f}")
    print(f"Loss finite:  {torch.isfinite(loss).item()}")

    if not torch.isfinite(loss):
        raise RuntimeError("INITIAL LOSS IS NaN/Inf. Training aborted.")

    print("Running backward pass...")
    loss.backward()

    bad_grads = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
        and not torch.isfinite(p.grad).all()
    ]
    checked = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    print(f"Gradient tensors checked: {checked}")

    if bad_grads:
        for name in bad_grads[:20]:
            print(f"  NaN/Inf grad: {name}")
        raise RuntimeError("NaN/Inf gradient detected in first backward pass.")

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    print("✓ Forward loss is finite")
    print("✓ Backward gradients are finite")
    print("✓ Numerical stability test passed")


# ============================================================================
# MAIN
# ============================================================================

def main():

    global tokenizer

    print_header("DEEPSEEK-CODER-V2-LITE-INSTRUCT LoRA FINE-TUNING")
    print("LUMI / AMD ROCm")
    print(f"Model: {MODEL_NAME}")

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
        MODEL_NAME, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    print(f"Vocab size:  {tokenizer.vocab_size}")
    print(f"Pad token:   {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"EOS token:   {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print(f"BOS token:   {tokenizer.bos_token!r}  (id={tokenizer.bos_token_id})")
    print("✓ Tokenizer loaded")

    _test = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        tokenize=True, add_generation_prompt=True,
    )
    print(f"✓ Chat template smoke-test: {len(_test)} tokens")

    # ------------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------------
    print_header("[3/6] MODEL")

    # attn_implementation="eager" is the only option supported by
    # DeepseekV2ForCausalLM in transformers 4.41.2.
    # "sdpa" raises ValueError; "flash_attention_2" requires flash_attn.
    # The get_imports patch above removes flash_attn from the static
    # dependency check so this load succeeds cleanly.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    model = model.to("cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e9:.2f}B")
    print("✓ Model loaded on cuda:0")

    # ------------------------------------------------------------------------
    # 4. LoRA
    # ------------------------------------------------------------------------
    print_header("[4/6] LoRA")

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

    validate_raw_dataset(raw_train,      "Training dataset")
    validate_raw_dataset(raw_validation, "Validation dataset")

    train_dataset      = tokenize_dataset(raw_train,      "Training dataset")
    validation_dataset = tokenize_dataset(raw_validation, "Validation dataset")

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

        # Keep at 0: forking after tokenizer parallelism is active causes
        # deadlocks / excessive warning spam that stalls progress reporting.
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

    print(f"\nEffective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Learning rate:        {LEARNING_RATE}")
    print(f"Max grad norm:        {MAX_GRAD_NORM}")
    print(f"Max sequence length:  {MAX_SEQ_LENGTH}")
    print(f"Epochs:               {NUM_EPOCHS}")

    # ------------------------------------------------------------------------
    # Stability test BEFORE real training
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

    train_result  = trainer.train()
    training_loss = float(train_result.training_loss)

    print(f"\nTraining loss: {training_loss:.8f}")

    if not math.isfinite(training_loss):
        raise RuntimeError(f"Training ended with non-finite loss: {training_loss}")

    # ------------------------------------------------------------------------
    # Save best adapter
    # ------------------------------------------------------------------------
    print_header("SAVING BEST LoRA ADAPTER")

    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))

    print(f"✓ LoRA adapter saved to:\n  {FINAL_ADAPTER_DIR}")

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------
    print_header("FINAL VALIDATION")

    eval_metrics = trainer.evaluate()
    eval_loss    = eval_metrics.get("eval_loss")

    if eval_loss is None:
        raise RuntimeError("Trainer did not return eval_loss.")

    eval_loss = float(eval_loss)

    if not math.isfinite(eval_loss):
        raise RuntimeError(f"FINAL VALIDATION LOSS IS NaN/Inf: {eval_loss}")

    perplexity = math.exp(eval_loss)
    print(f"Validation loss: {eval_loss:.8f}")
    print(f"Perplexity:      {perplexity:.8f}")

    # ------------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------------
    report = {
        "timestamp":            datetime.now().isoformat(),
        "model":                MODEL_NAME,
        "adapter_directory":    str(FINAL_ADAPTER_DIR),
        "lora_rank":            LORA_RANK,
        "lora_alpha":           LORA_ALPHA,
        "lora_dropout":         LORA_DROPOUT,
        "learning_rate":        LEARNING_RATE,
        "batch_size":           BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "epochs":               NUM_EPOCHS,
        "max_sequence_length":  MAX_SEQ_LENGTH,
        "max_grad_norm":        MAX_GRAD_NORM,
        "training_examples":    len(train_dataset),
        "validation_examples":  len(validation_dataset),
        "training_loss":        training_loss,
        "validation_loss":      eval_loss,
        "perplexity":           perplexity,
        "evaluation_metrics":   eval_metrics,
        "log_history":          trainer.state.log_history,
    }

    metrics_file = RESULTS_DIR / "final_training_evaluation.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Report saved to:\n  {metrics_file}")

    # ------------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------------
    print_header("TRAINING COMPLETE")
    print("✓ Stability test passed")
    print("✓ Training loss is finite")
    print("✓ Validation loss is finite")
    print(f"✓ Adapter:\n  {FINAL_ADAPTER_DIR}")
    print("\nNext step: Run evaluate_deepseek.py separately.")


if __name__ == "__main__":
    main()