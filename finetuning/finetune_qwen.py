#!/usr/bin/env python3
"""
Fine-tune Qwen2.5 14B on LUMI for Playwright scenario generation
"""

import json
import os
import torch
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-14B"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4  # Reduced for 14B model on GPU memory
GRADIENT_ACCUMULATION = 2
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024

def format_prompt(example):
    """Format instruction-input-output into a single prompt"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Format: [INST] instruction + input [/INST] output
    prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n{output}"
    return {"text": prompt}

def main():
    print("=" * 80)
    print("QWEN 2.5 14B FINE-TUNING ON LUMI")
    print("=" * 80)
    
    # Check GPU availability
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load model
    print("\n[2/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Model loaded ({MODEL_NAME})")
    
    # Setup LoRA
    print("\n[3/6] Setting up LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✓ LoRA configured")
    
    # Load training data
    print("\n[4/6] Loading training data...")
    training_data_path = "dataset/training_data.jsonl"
    if not Path(training_data_path).exists():
        raise FileNotFoundError(f"Training data not found at {training_data_path}")
    
    # Load JSONL
    dataset = load_dataset("json", data_files=training_data_path)
    dataset = dataset["train"].map(format_prompt, remove_columns=["instruction", "input", "output"])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"✓ Training data loaded ({len(dataset)} examples)")
    print(f"  Sample: {dataset[0].keys()}")
    
    # Training arguments
    print("\n[5/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir="/project/project_465003167/m10-testbot/qwen-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="linear",
        logging_dir="/project/project_465003167/m10-testbot/logs",
        use_cuda=torch.cuda.is_available(),
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=["tensorboard"],
    )
    print("✓ Training arguments configured")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )
    
    # Train
    print("\n[6/6] Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()
    
    trainer.train()
    
    # Save
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    print("\n[SAVE] Saving fine-tuned model...")
    output_dir = "/project/project_465003167/m10-testbot/qwen-finetuned-final"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model saved to: {output_dir}")
    print(f"\nTo use in your scenario agent:")
    print(f"  model_path = '{output_dir}'")
    print(f"  model = AutoModelForCausalLM.from_pretrained(model_path)")
    print(f"  tokenizer = AutoTokenizer.from_pretrained(model_path)")

if __name__ == "__main__":
    main()