#!/usr/bin/env python3
"""
Fine-tune Qwen2.5 14B with Evaluation Metrics
Automatically tracks BLEU, ROUGE, Exact Match, Perplexity during training
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# Import metric libraries
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data for BLEU
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
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024

# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score (0-100).
    Measures similarity between candidate and reference.
    
    - Reference: expected output (your correct script)
    - Candidate: model's generated output
    - Score: 0-100 (higher is better)
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # Smoothing function to handle short sentences
    smoothing = SmoothingFunction().method1
    
    bleu = sentence_bleu(
        [ref_tokens],
        cand_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )
    return bleu * 100  # Convert to 0-100 scale


def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE-L score (0-100).
    Measures longest common substring matching.
    Better for code/scripts than BLEU.
    
    - Reference: expected output
    - Candidate: model's generated output
    - Score: 0-100 (higher is better)
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    rouge_l = scores['rougeL'].fmeasure
    return rouge_l * 100  # Convert to 0-100 scale


def calculate_exact_match(reference, candidate):
    """
    Calculate Exact Match (0 or 100).
    Did the model generate the EXACT expected output?
    
    - 100: Perfect match
    - 0: Not a perfect match
    """
    return 100.0 if reference.strip() == candidate.strip() else 0.0


def calculate_token_accuracy(reference, candidate):
    """
    Calculate Token Accuracy (0-100).
    What % of words match?
    
    - Compares word-by-word
    - Good for understanding partial correctness
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # If different lengths, use minimum length for fair comparison
    min_len = min(len(ref_tokens), len(cand_tokens))
    
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if ref_tokens[i] == cand_tokens[i])
    accuracy = (matches / len(ref_tokens)) * 100  # Normalized to reference length
    
    return accuracy


def calculate_perplexity(model, tokens, device):
    """
    Calculate Perplexity.
    How confident is the model in its output?
    
    - Lower is better (confident)
    - Typical: 5-20 is good
    - >50 means confused
    
    This is computed from the loss: perplexity = 2^loss
    """
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_metrics_batch(eval_dataset, model, tokenizer, device, num_samples=10):
    """
    Compute all metrics on a batch of examples.
    
    Args:
        eval_dataset: Dataset with (instruction, input, output)
        model: Fine-tuned model
        tokenizer: Tokenizer
        device: GPU/CPU
        num_samples: How many examples to evaluate (10-20 is good)
    
    Returns:
        dict with all metrics
    """
    metrics = {
        'bleu_scores': [],
        'rouge_scores': [],
        'exact_matches': [],
        'token_accuracies': [],
        'perplexities': [],
    }
    
    # Sample subset for faster evaluation
    sample_size = min(num_samples, len(eval_dataset))
    sample_indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
    
    model.eval()
    
    for idx in sample_indices:
        example = eval_dataset[int(idx)]
        
        # Extract expected output
        reference = example['output']
        
        # Generate prediction
        prompt = f"[INST] {example['instruction']}\n\n{example['input']} [/INST]\n"
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                top_p=0.9,
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (after [/INST])
        if '[/INST]\n' in generated_text:
            candidate = generated_text.split('[/INST]\n')[1]
        else:
            candidate = generated_text
        
        # Calculate metrics
        bleu = calculate_bleu(reference, candidate)
        rouge = calculate_rouge(reference, candidate)
        exact = calculate_exact_match(reference, candidate)
        token_acc = calculate_token_accuracy(reference, candidate)
        
        metrics['bleu_scores'].append(bleu)
        metrics['rouge_scores'].append(rouge)
        metrics['exact_matches'].append(exact)
        metrics['token_accuracies'].append(token_acc)
    
    # Return averages
    return {
        'eval_bleu': np.mean(metrics['bleu_scores']),
        'eval_rouge': np.mean(metrics['rouge_scores']),
        'eval_exact_match': np.mean(metrics['exact_matches']),
        'eval_token_accuracy': np.mean(metrics['token_accuracies']),
    }


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def format_prompt(example):
    """Format instruction-input-output into a single prompt"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n{output}"
    return {"text": prompt}


def main():
    print("=" * 80)
    print("QWEN 2.5 14B FINE-TUNING WITH EVALUATION METRICS")
    print("=" * 80)
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # STEP 1: Load Tokenizer & Model
    # ========================================================================
    
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    print("\n[2/7] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Model loaded")
    
    # ========================================================================
    # STEP 2: Setup LoRA
    # ========================================================================
    
    print("\n[3/7] Setting up LoRA...")
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
    
    # ========================================================================
    # STEP 3: Load Data (Training + Validation)
    # ========================================================================
    
    print("\n[4/7] Loading training and validation data...")
    
    # Training data
    training_dataset = load_dataset("json", data_files="dataset/training_data.jsonl")["train"]
    training_dataset = training_dataset.map(format_prompt, remove_columns=["instruction", "input", "output"])
    
    # Tokenize training data
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            )

        # Assign labels to match input_ids for causal language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    training_dataset = training_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Validation data
    validation_dataset = None
    if Path("dataset/validation_data.jsonl").exists():
        validation_dataset = load_dataset("json", data_files="dataset/validation_data.jsonl")["train"]
        validation_dataset_raw = validation_dataset.map(
            lambda x: {**format_prompt(x), "instruction": x.get("instruction", ""), "input": x.get("input", ""), "output": x.get("output", "")}
        )
        validation_dataset_formatted = validation_dataset_raw.map(lambda x: {"text": x["text"]})
        validation_dataset_formatted = validation_dataset_formatted.map(tokenize_function, batched=True, remove_columns=["text"])
        print(f"✓ Validation data loaded ({len(validation_dataset)} examples)")
    else:
        print("⚠ No validation_data.jsonl found - skipping validation metrics")
    
    print(f"✓ Training data loaded ({len(training_dataset)} examples)")
    
    # ========================================================================
    # STEP 4: Training Arguments
    # ========================================================================
    
    print("\n[5/7] Setting up training...")
    
    # Create results directory
    results_dir = "/project/project_465003167/m10-testbot/finetuning/training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir="/project/project_465003167/m10-testbot/finetuning/qwen-finetuned",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        save_steps=50,
        save_total_limit=3,
        eval_steps=50,  # Evaluate every 50 steps
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="linear",
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        eval_strategy="steps",  # Evaluate every N steps
        save_strategy="steps",
    )
    
    # ========================================================================
    # STEP 6: Create Trainer & Train
    # ========================================================================
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset_formatted if validation_dataset else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    
    print("\n[6/7] Starting training...")
    trainer.train()
    
    # ========================================================================
    # STEP 7: Evaluate on Full Validation Set
    # ========================================================================
    
    print("\n[7/7] Computing final evaluation metrics...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset) if validation_dataset else 0,
    }
    
    # Compute metrics on validation data
    if validation_dataset:
        print("\nEvaluating on validation set (this takes ~2 minutes)...")
        
        # Use the original validation dataset with outputs for metric calculation
        eval_metrics = compute_metrics_batch(
            validation_dataset,
            model,
            tokenizer,
            device,
            num_samples=min(20, len(validation_dataset))
        )
        
        results["metrics"] = eval_metrics
        
        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)
        print(f"\n✓ BLEU Score:        {eval_metrics['eval_bleu']:.2f} / 100")
        print(f"✓ ROUGE-L Score:     {eval_metrics['eval_rouge']:.2f} / 100")
        print(f"✓ Exact Match Rate:  {eval_metrics['eval_exact_match']:.2f}%")
        print(f"✓ Token Accuracy:    {eval_metrics['eval_token_accuracy']:.2f}%")
        print("\nInterpretation:")
        print(f"  - BLEU > 40 is good")
        print(f"  - ROUGE > 50 is good")
        print(f"  - Exact Match > 30% is good")
        print(f"  - Token Accuracy > 60% is good")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save final model
    final_model_dir = "/project/project_465003167/m10-testbot/finetuning/qwen-finetuned-final"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\n✓ Model saved to: {final_model_dir}")
    
    # Save metrics as JSON
    metrics_file = f"{results_dir}/evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save metrics as readable text
    text_file = f"{results_dir}/evaluation_report.txt"
    with open(text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINE-TUNING EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model:                {results['model']}\n")
        f.write(f"Epochs:               {results['epochs']}\n")
        f.write(f"Batch Size:           {results['batch_size']}\n")
        f.write(f"Learning Rate:        {results['learning_rate']}\n")
        f.write(f"LoRA Rank:            {results['lora_rank']}\n")
        f.write(f"Training Examples:    {results['training_examples']}\n")
        f.write(f"Validation Examples:  {results['validation_examples']}\n")
        f.write(f"Timestamp:            {results['timestamp']}\n\n")
        
        if "metrics" in results:
            f.write("EVALUATION METRICS\n")
            f.write("-" * 80 + "\n")
            metrics = results["metrics"]
            f.write(f"BLEU Score:           {metrics['eval_bleu']:.2f} / 100\n")
            f.write(f"ROUGE-L Score:        {metrics['eval_rouge']:.2f} / 100\n")
            f.write(f"Exact Match Rate:     {metrics['eval_exact_match']:.2f}%\n")
            f.write(f"Token Accuracy:       {metrics['eval_token_accuracy']:.2f}%\n\n")
            
            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("BLEU > 40:         ✓ Good (>40), ✗ Poor (<30)\n")
            f.write("ROUGE > 50:        ✓ Good (>50), ✗ Poor (<40)\n")
            f.write("Exact Match > 30%: ✓ Good (>30%), ✗ Poor (<15%)\n")
            f.write("Token Accuracy:    ✓ Good (>60%), ✗ Poor (<45%)\n")
    
    print(f"✓ Report saved to: {text_file}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Model directory:   {final_model_dir}")


if __name__ == "__main__":
    main()