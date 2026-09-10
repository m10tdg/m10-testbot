#!/usr/bin/env python3
"""
Fine-tune Qwen2.5 14B with FIXED training stability for test script generation
Critical fixes applied:
1. Response-only label masking (-100 on prompt tokens)
2. AMD ROCm non-reentrant gradient checkpointing fix (use_reentrant=False)
3. Sequence length locked to 1024 to prevent output truncation
4. Clean dataset processing avoiding text column collisions
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

# Import metric libraries
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-14B"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATE = 5e-5       # Adjusted for stable Qwen 14B LoRA fine-tuning
BATCH_SIZE = 2             # Device batch size
GRADIENT_ACCUMULATION = 4  # Effective batch size = 8
WARMUP_STEPS = 50          # Scaled for dataset size
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 1024      # Prevents output truncation during training
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 0.3        # Prevents exploding gradients on AMD ROCm
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# Global tokenizer instance for dataset mapping
tokenizer = None


# ============================================================================
# DATASET TOKENIZATION WITH RESPONSE-ONLY MASKING
# ============================================================================

def format_and_tokenize(examples):
    """Formats instruction + input into prompt, and masks prompt with -100 in labels."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n"
        response = f"{output}"

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        
        # Mask prompt tokens (-100) so loss is computed ONLY on target output code
        labels = [-100] * len(prompt_ids) + response_ids

        # Truncate if total length exceeds MAX_SEQ_LENGTH
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
            labels = labels[:MAX_SEQ_LENGTH]

        attention_mask = [1] * len(input_ids)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

    return model_inputs


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_bleu(reference, candidate):
    """Calculate BLEU score (0-100)"""
    if not reference or not candidate:
        return 0.0
    
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    
    bleu = sentence_bleu(
        [ref_tokens],
        cand_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )
    return bleu * 100.0


def calculate_rouge(reference, candidate):
    """Calculate ROUGE-L score (0-100)"""
    if not reference or not candidate:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure * 100.0
    except Exception:
        return 0.0


def calculate_exact_match(reference, candidate):
    """Calculate Exact Match (0 or 100)"""
    return 100.0 if reference.strip() == candidate.strip() else 0.0


def calculate_token_accuracy(reference, candidate):
    """Calculate Token Accuracy (0-100)"""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if not ref_tokens:
        return 0.0
    
    min_len = min(len(ref_tokens), len(cand_tokens))
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if ref_tokens[i] == cand_tokens[i])
    return (matches / len(ref_tokens)) * 100.0


# ============================================================================
# CUSTOM TRAINER FOR EVALUATION
# ============================================================================

class CustomTrainer(Trainer):
    """Trainer with custom evaluation metrics during validation steps"""
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        results = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        target_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if target_dataset is not None and hasattr(self, "raw_validation_dataset"):
            print("\nComputing BLEU/ROUGE metrics on validation sample...")
            custom_metrics = self.compute_custom_metrics(self.raw_validation_dataset)
            results.update(custom_metrics)
        
        return results
    
    def compute_custom_metrics(self, raw_eval_dataset, num_samples=10):
        metrics = {
            'eval_bleu': [],
            'eval_rouge': [],
            'eval_exact_match': [],
            'eval_token_accuracy': [],
        }
        
        sample_size = min(num_samples, len(raw_eval_dataset))
        sample_indices = np.random.choice(len(raw_eval_dataset), sample_size, replace=False)
        
        self.model.eval()
        device = self.model.device
        
        for idx in sample_indices:
            try:
                example = raw_eval_dataset[int(idx)]
                reference = example.get('output', '')
                if not reference:
                    continue
                
                instruction = example.get('instruction', '')
                input_text = example.get('input', '')
                prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n"
                
                inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        temperature=0.2,
                        top_p=0.9,
                        do_sample=False,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if '[/INST]\n' in generated_text:
                    candidate = generated_text.split('[/INST]\n')[1].strip()
                else:
                    candidate = generated_text.strip()
                
                metrics['eval_bleu'].append(calculate_bleu(reference, candidate))
                metrics['eval_rouge'].append(calculate_rouge(reference, candidate))
                metrics['eval_exact_match'].append(calculate_exact_match(reference, candidate))
                metrics['eval_token_accuracy'].append(calculate_token_accuracy(reference, candidate))
                
            except Exception as e:
                print(f"  Warning: Error evaluating example {idx}: {e}")
                continue
        
        return {
            'eval_bleu_custom': np.mean(metrics['eval_bleu']) if metrics['eval_bleu'] else 0.0,
            'eval_rouge_custom': np.mean(metrics['eval_rouge']) if metrics['eval_rouge'] else 0.0,
            'eval_exact_match_custom': np.mean(metrics['eval_exact_match']) if metrics['eval_exact_match'] else 0.0,
            'eval_token_accuracy_custom': np.mean(metrics['eval_token_accuracy']) if metrics['eval_token_accuracy'] else 0.0,
        }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    global tokenizer

    print("=" * 80)
    print("QWEN 2.5 14B FINE-TUNING (AMD ROCm STABLE VERSION)")
    print("=" * 80)
    
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # [1/6] Load Tokenizer & Model
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    print("\n[2/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✓ Model loaded")
    
    # [3/6] Setup LoRA
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
    
    # [4/6] Load and Tokenize Datasets
    print("\n[4/6] Processing training and validation datasets...")
    
    raw_training = load_dataset("json", data_files="dataset/training_data.jsonl")["train"]
    training_dataset = raw_training.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw_training.column_names,
    )
    
    raw_validation = None
    validation_dataset = None
    if Path("dataset/validation_data.jsonl").exists():
        raw_validation = load_dataset("json", data_files="dataset/validation_data.jsonl")["train"]
        validation_dataset = raw_validation.map(
            format_and_tokenize,
            batched=True,
            remove_columns=raw_validation.column_names,
        )
        print(f"✓ Validation data loaded ({len(validation_dataset)} examples)")
    
    print(f"✓ Training data loaded ({len(training_dataset)} examples)")
    
    # [5/6] Training Arguments & Setup
    print("\n[5/6] Setting up training arguments...")
    results_dir = "/project/project_465003167/m10-testbot/finetuning/training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir="/project/project_465003167/m10-testbot/finetuning/qwen-finetuned",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        save_steps=50,
        save_total_limit=3,
        eval_steps=50,
        logging_steps=5,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=True,
        fp16=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Fixes NaN gradients on AMD ROCm
        max_grad_norm=MAX_GRAD_NORM,
        report_to=["tensorboard"],
        eval_strategy="steps" if validation_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if validation_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        seed=42,
    )
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        callbacks=[early_stopping] if validation_dataset else [],
    )
    
    trainer.tokenizer = tokenizer
    trainer.raw_validation_dataset = raw_validation
    
    # [6/6] Train Model
    print("\n[6/6] Starting training...")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max grad norm: {MAX_GRAD_NORM}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    
    trainer.train()
    
    # Save Model and Report
    print("\n" + "=" * 80)
    print("SAVING MODEL AND GENERATING REPORT")
    print("=" * 80)
    
    final_model_dir = "/project/project_465003167/m10-testbot/finetuning/qwen-finetuned-final"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✓ Model saved to: {final_model_dir}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "lora_rank": LORA_RANK,
        "max_seq_length": MAX_SEQ_LENGTH,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset) if validation_dataset else 0,
    }
    
    if validation_dataset:
        print("\nRunning final validation evaluation...")
        eval_result = trainer.evaluate(eval_dataset=validation_dataset)
        results["metrics"] = eval_result
        
        print("\nFINAL EVALUATION METRICS")
        print("-" * 80)
        if "eval_loss" in eval_result:
            print(f"✓ Validation Loss: {eval_result['eval_loss']:.4f}")
        if "eval_bleu_custom" in eval_result:
            print(f"✓ BLEU Score:      {eval_result['eval_bleu_custom']:.2f} / 100")
        if "eval_rouge_custom" in eval_result:
            print(f"✓ ROUGE-L Score:   {eval_result['eval_rouge_custom']:.2f} / 100")
        if "eval_exact_match_custom" in eval_result:
            print(f"✓ Exact Match:     {eval_result['eval_exact_match_custom']:.2f}%")
        if "eval_token_accuracy_custom" in eval_result:
            print(f"✓ Token Accuracy:  {eval_result['eval_token_accuracy_custom']:.2f}%")
    
    metrics_file = f"{results_dir}/evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    text_file = f"{results_dir}/evaluation_report.txt"
    with open(text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINE-TUNING EVALUATION REPORT (FIXED)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model:                    {results['model']}\n")
        f.write(f"Epochs:                   {results['epochs']}\n")
        f.write(f"Batch Size:               {results['batch_size']}\n")
        f.write(f"Gradient Accumulation:    {results['gradient_accumulation']}\n")
        f.write(f"Effective Batch Size:     {results['batch_size'] * results['gradient_accumulation']}\n")
        f.write(f"Learning Rate:            {results['learning_rate']}\n")
        f.write(f"Max Grad Norm:            {results['max_grad_norm']}\n")
        f.write(f"LoRA Rank:                {results['lora_rank']}\n")
        f.write(f"Max Sequence Length:      {results['max_seq_length']}\n")
        f.write(f"Training Examples:        {results['training_examples']}\n")
        f.write(f"Validation Examples:      {results['validation_examples']}\n")
        f.write(f"Timestamp:                {results['timestamp']}\n\n")
        
        if "metrics" in results:
            f.write("EVALUATION METRICS\n")
            f.write("-" * 80 + "\n")
            metrics = results["metrics"]
            if "eval_loss" in metrics:
                f.write(f"Validation Loss:     {metrics['eval_loss']:.4f}\n")
            if "eval_bleu_custom" in metrics:
                f.write(f"BLEU Score:          {metrics['eval_bleu_custom']:.2f} / 100\n")
            if "eval_rouge_custom" in metrics:
                f.write(f"ROUGE-L Score:       {metrics['eval_rouge_custom']:.2f} / 100\n")
            if "eval_exact_match_custom" in metrics:
                f.write(f"Exact Match Rate:    {metrics['eval_exact_match_custom']:.2f}%\n")
            if "eval_token_accuracy_custom" in metrics:
                f.write(f"Token Accuracy:      {metrics['eval_token_accuracy_custom']:.2f}%\n\n")
    
    print(f"✓ Report saved to: {text_file}")
    print("\n✅ TRAINING COMPLETE")


if __name__ == "__main__":
    main()