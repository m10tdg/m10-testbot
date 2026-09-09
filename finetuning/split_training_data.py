#!/usr/bin/env python3
"""
Split your data into training (80%) and validation (20%) sets.

Usage:
    python split_training_data.py --input all_data.jsonl --train-ratio 0.8

This will create:
    - training_data.jsonl (80% of examples)
    - validation_data.jsonl (20% of examples)
"""

import json
import argparse
import random
from pathlib import Path


def split_data(input_file, train_ratio=0.8, seed=42):
    """
    Split JSONL file into training and validation sets.
    
    Args:
        input_file: Path to input JSONL file
        train_ratio: Fraction for training (default 0.8 = 80%)
        seed: Random seed for reproducibility
    
    Returns:
        (train_data, valid_data) lists
    """
    # Read all examples
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Found {len(all_data)} total examples")
    
    if len(all_data) < 20:
        print("⚠ Warning: Very small dataset (< 20 examples)")
        print("  Recommended minimum: 50 examples")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(all_data)
    
    # Split
    split_point = int(len(all_data) * train_ratio)
    train_data = all_data[:split_point]
    valid_data = all_data[split_point:]
    
    return train_data, valid_data


def save_jsonl(data, output_file):
    """Save list of dicts to JSONL file."""
    with open(output_file, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    print(f"✓ Saved {len(data)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split data into training and validation sets"
    )
    parser.add_argument(
        "--input",
        default="dataset/training_data.jsonl",
        help="Input JSONL file (default: dataset/training_data.jsonl)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training (default: 0.8 = 80/20 split)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"✗ Error: {args.input} not found")
        print("\nUsage: python split_training_data.py --input your_file.jsonl")
        return
    
    # Validate ratio
    if not 0 < args.train_ratio < 1:
        print(f"✗ Error: train_ratio must be between 0 and 1, got {args.train_ratio}")
        return
    
    print("=" * 80)
    print("DATA SPLITTING TOOL")
    print("=" * 80)
    
    # Split data
    train_data, valid_data = split_data(
        args.input,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Save
    print()
    save_jsonl(train_data, "dataset/training_data.jsonl")
    save_jsonl(valid_data, "dataset/validation_data.jsonl")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal examples:      {len(train_data) + len(valid_data)}")
    print(f"Training examples:   {len(train_data)} ({len(train_data)/(len(train_data)+len(valid_data))*100:.1f}%)")
    print(f"Validation examples: {len(valid_data)} ({len(valid_data)/(len(train_data)+len(valid_data))*100:.1f}%)")
    
    # Recommendations
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Verify the split looks good:")
    print("   wc -l training_data.jsonl validation_data.jsonl")
    
    print("\n2. Use with fine-tuning script:")
    print("   python finetune_qwen_with_metrics.py")
    
    print("\n3. The script will automatically:")
    print("   - Load training_data.jsonl for training")
    print("   - Load validation_data.jsonl for evaluation")
    print("   - Calculate BLEU, ROUGE, Exact Match, Token Accuracy")
    print("   - Save results to training_results/")


if __name__ == "__main__":
    main()