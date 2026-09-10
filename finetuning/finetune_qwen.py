#!/usr/bin/env python3
"""
Fine-tune Qwen2.5 14B with FIXED training stability for test script generation
Critical fixes:
1. Gradient clipping + stable learning rate
2. Reduced dataset size = reduced LR
3. Proper max_length based on actual data
4. Early stopping + validation monitoring
5. Proper metric computation
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
# FIXED CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-14B"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ⚠️ CRITICAL FIXES:
LEARNING_RATE = 2e-5  # REDUCED: Small dataset needs smaller LR (was 1e-4)
BATCH_SIZE = 2        # REDUCED: Was 4, now 2 for gradient stability
GRADIENT_ACCUMULATION = 4  # INCREASED: Maintain effective batch size = 8
WARMUP_STEPS = 200    # MORE WARMUP: Prevents early loss spike
NUM_EPOCHS = 5        # MORE EPOCHS: For small dataset, more passes help
MAX_SEQ_LENGTH = 512  # REDUCED: Your data is ~400 tokens, not 1024
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 1.0   # ADD: Prevents gradient explosion (NaN fix)
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"  # CHANGED: Better than linear for stability

# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def analyze_dataset():
    """Analyze your actual data to set proper hyperparameters"""
    print("\n" + "=" * 80)
    print("ANALYZING TRAINING DATA")
    print("=" * 80)
    
    training_data = load_dataset("json", data_files="dataset/training_data.jsonl")["train"]
    
    lengths = []
    for example in training_data:
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        full_text = f"[INST] {instruction}\n\n{input_text} [/INST]\n{output}"
        lengths.append(len(full_text.split()))  # Word count
    
    lengths = np.array(lengths)
    print(f"\nToken statistics (approximate):")
    print(f"  Mean:   {lengths.mean():.0f} tokens")
    print(f"  Median: {np.median(lengths):.0f} tokens")
    print(f"  Max:    {lengths.max():.0f} tokens")
    print(f"  95th%:  {np.percentile(lengths, 95):.0f} tokens")
    print(f"\n→ Recommended MAX_SEQ_LENGTH: {int(np.percentile(lengths, 95) * 1.2)}")
    
    return int(np.percentile(lengths, 95) * 1.2)


# ============================================================================
# METRIC CALCULATION FUNCTIONS (FIXED)
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
    return bleu * 100


def calculate_rouge(reference, candidate):
    """Calculate ROUGE-L score (0-100)"""
    if not reference or not candidate:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(reference, candidate)
        rouge_l = scores['rougeL'].fmeasure
        return rouge_l * 100
    except:
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
        return 0.0 if len(ref_tokens) > 0 else 100.0
    
    matches = sum(1 for i in range(min_len) if ref_tokens[i] == cand_tokens[i])
    accuracy = (matches / len(ref_tokens)) * 100
    
    return accuracy


# ============================================================================
# CUSTOM TRAINER FOR EVALUATION
# ============================================================================

class CustomTrainer(Trainer):
    """Trainer with custom evaluation metrics"""
    
    def evaluate(self, eval_dataset=None, **kwargs):
        """Override to add custom metrics"""
        results = super().evaluate(eval_dataset, **kwargs)
        
        # Add custom metrics if validation dataset exists
        if eval_dataset is not None:
            print("\n⏳ Computing BLEU/ROUGE metrics (this takes ~2 min)...")
            custom_metrics = self.compute_custom_metrics(eval_dataset)
            results.update(custom_metrics)
        
        return results
    
    def compute_custom_metrics(self, eval_dataset, num_samples=10):
        """Compute BLEU, ROUGE on a sample of validation data"""
        metrics = {
            'eval_bleu': [],
            'eval_rouge': [],
            'eval_exact_match': [],
            'eval_token_accuracy': [],
        }
        
        sample_size = min(num_samples, len(eval_dataset))
        sample_indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
        
        self.model.eval()
        device = self.model.device
        
        for idx in sample_indices:
            try:
                example = eval_dataset[int(idx)]
                
                reference = example.get('output', '')
                if not reference:
                    continue
                
                # Build prompt
                instruction = example.get('instruction', '')
                input_text = example.get('input', '')
                prompt = f"[INST] {instruction}\n\n{input_text} [/INST]\n"
                
                inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract generated part
                if '[/INST]\n' in generated_text:
                    candidate = generated_text.split('[/INST]\n')[1].strip()
                else:
                    candidate = generated_text
                
                # Compute metrics
                metrics['eval_bleu'].append(calculate_bleu(reference, candidate))
                metrics['eval_rouge'].append(calculate_rouge(reference, candidate))
                metrics['eval_exact_match'].append(calculate_exact_match(reference, candidate))
                metrics['eval_token_accuracy'].append(calculate_token_accuracy(reference, candidate))
                
            except Exception as e:
                print(f"  ⚠️ Error evaluating example {idx}: {e}")
                continue
        
        # Average metrics
        return {
            'eval_bleu_custom': np.mean(metrics['eval_bleu']) if metrics['eval_bleu'] else 0.0,
            'eval_rouge_custom': np.mean(metrics['eval_rouge']) if metrics['eval_rouge'] else 0.0,
            'eval_exact_match_custom': np.mean(metrics['eval_exact_match']) if metrics['eval_exact_match'] else 0.0,
            'eval_token_accuracy_custom': np.mean(metrics['eval_token_accuracy']) if metrics['eval_token_accuracy'] else 0.0,
        }


# ============================================================================
# MAIN TRAINING FUNCTION
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
    print("QWEN 2.5 14B FINE-TUNING (FIXED VERSION)")
    print("=" * 80)
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # STEP 0: Analyze Data
    # ========================================================================
    
    recommended_seq_length = analyze_dataset()
    seq_length = min(512, max(256, recommended_seq_length))
    print(f"→ Using MAX_SEQ_LENGTH = {seq_length}")
    
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
        torch_dtype=torch.bfloat16,
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
    # STEP 3: Load Data
    # ========================================================================
    
    print("\n[4/7] Loading training and validation data...")
    
    # Training data
    training_dataset = load_dataset("json", data_files="dataset/training_data.jsonl")["train"]
    training_dataset = training_dataset.map(format_prompt, remove_columns=["instruction", "input", "output"])
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_length,
        )
        labels = [list(ids) for ids in model_inputs["input_ids"]]
        
        for i in range(len(labels)):
            labels[i] = [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in labels[i]
            ]
        
        model_inputs["labels"] = labels
        return model_inputs
    
    training_dataset = training_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Validation data
    validation_dataset = None
    validation_dataset_formatted = None
    if Path("dataset/validation_data.jsonl").exists():
        validation_dataset = load_dataset("json", data_files="dataset/validation_data.jsonl")["train"]
        validation_dataset_formatted = validation_dataset.map(format_prompt, remove_columns=["instruction", "input", "output"])
        validation_dataset_formatted = validation_dataset_formatted.map(tokenize_function, batched=True, remove_columns=["text"])
        print(f"✓ Validation data loaded ({len(validation_dataset)} examples)")
    
    print(f"✓ Training data loaded ({len(training_dataset)} examples)")
    
    # ========================================================================
    # STEP 4: Training Arguments (FIXED)
    # ========================================================================
    
    print("\n[5/7] Setting up training...")
    
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
        max_grad_norm=MAX_GRAD_NORM,  # ← FIX: Prevents NaN gradients
        report_to=["tensorboard"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # New additions for stability
        optim="adamw_8bit",  # Memory efficient
        seed=42,
    )
    
    # ========================================================================
    # STEP 6: Create Trainer & Train
    # ========================================================================
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset_formatted if validation_dataset else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        callbacks=[early_stopping],
    )
    
    print("\n[6/7] Starting training...")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max grad norm: {MAX_GRAD_NORM}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    
    trainer.train()
    
    # ========================================================================
    # STEP 7: Final Evaluation
    # ========================================================================
    
    print("\n[7/7] Computing final metrics...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "lora_rank": LORA_RANK,
        "max_seq_length": seq_length,
        "training_examples": len(training_dataset),
        "validation_examples": len(validation_dataset) if validation_dataset else 0,
    }
    
    if validation_dataset:
        print("\nEvaluating on full validation set...")
        eval_result = trainer.evaluate(eval_dataset=validation_dataset_formatted)
        results["metrics"] = eval_result
        
        # Print nicely
        print("\n" + "=" * 80)
        print("FINAL EVALUATION METRICS")
        print("=" * 80)
        if "eval_loss" in eval_result:
            print(f"\n✓ Validation Loss: {eval_result['eval_loss']:.4f}")
        if "eval_bleu_custom" in eval_result:
            print(f"✓ BLEU Score:      {eval_result['eval_bleu_custom']:.2f} / 100")
        if "eval_rouge_custom" in eval_result:
            print(f"✓ ROUGE-L Score:   {eval_result['eval_rouge_custom']:.2f} / 100")
        if "eval_exact_match_custom" in eval_result:
            print(f"✓ Exact Match:     {eval_result['eval_exact_match_custom']:.2f}%")
        if "eval_token_accuracy_custom" in eval_result:
            print(f"✓ Token Accuracy:  {eval_result['eval_token_accuracy_custom']:.2f}%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
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
    
    # Save readable report
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
            
            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("Loss > 1.0:        ✓ Good (training is learning)\n")
            f.write("BLEU > 30:         ✓ Good for code generation\n")
            f.write("ROUGE > 40:        ✓ Good for code structure\n")
            f.write("Exact Match > 10%: ✓ Reasonable for code gen\n")
            f.write("Token Accuracy > 50%: ✓ Good for code gen\n")
    
    print(f"✓ Report saved to: {text_file}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Model directory:   {final_model_dir}")


if __name__ == "__main__":
    main()