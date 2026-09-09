#!/usr/bin/env python3
"""
Analyze fine-tuning results after training completes.

Usage:
    python analyze_results.py

This will:
1. Read evaluation_metrics.json
2. Print formatted results
3. Create simple visualizations
4. Give interpretation
"""

import json
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_metric(name, value, range_good, range_fair, unit="%"):
    """Print a metric with color-coded interpretation"""
    
    if isinstance(value, str):
        value = float(value.rstrip('%'))
    
    # Interpret
    if value >= range_good:
        status = "✓ GOOD"
    elif value >= range_fair:
        status = "~ FAIR"
    else:
        status = "✗ POOR"
    
    # Bar chart
    bar_length = 40
    filled = int((value / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"{name:20} {value:6.2f}{unit}  {bar}  {status}")


def main():
    print_header("FINE-TUNING RESULTS ANALYZER")
    
    # Check if results exist
    results_file = "training_results/evaluation_metrics.json"
    
    if not Path(results_file).exists():
        print("✗ Error: evaluation_metrics.json not found")
        print(f"  Expected at: {results_file}")
        print("\n  Make sure training completed and results were saved.")
        return
    
    # Load results
    print("Loading results...")
    with open(results_file) as f:
        results = json.load(f)
    
    print("✓ Results loaded\n")
    
    # Print configuration
    print_header("TRAINING CONFIGURATION")
    config = {
        "Model": results.get("model", "Unknown"),
        "Epochs": results.get("epochs", "?"),
        "Batch Size": results.get("batch_size", "?"),
        "Learning Rate": results.get("learning_rate", "?"),
        "LoRA Rank": results.get("lora_rank", "?"),
        "Training Examples": results.get("training_examples", "?"),
        "Validation Examples": results.get("validation_examples", "?"),
        "Timestamp": results.get("timestamp", "?"),
    }
    
    for key, value in config.items():
        print(f"  {key:.<30} {value}")
    
    # Print metrics
    if "metrics" not in results or not results["metrics"]:
        print("\n⚠ No evaluation metrics found in results.")
        print("  Make sure validation_data.jsonl was provided during training.")
        return
    
    metrics = results["metrics"]
    
    print_header("EVALUATION METRICS")
    
    print("Metric scores (higher is generally better):\n")
    
    print_metric(
        "BLEU Score",
        metrics.get("eval_bleu", 0),
        range_good=40,
        range_fair=30,
        unit="/100"
    )
    
    print_metric(
        "ROUGE-L Score",
        metrics.get("eval_rouge", 0),
        range_good=50,
        range_fair=40,
        unit="/100"
    )
    
    print_metric(
        "Exact Match Rate",
        metrics.get("eval_exact_match", 0),
        range_good=30,
        range_fair=15,
        unit="%"
    )
    
    print_metric(
        "Token Accuracy",
        metrics.get("eval_token_accuracy", 0),
        range_good=60,
        range_fair=50,
        unit="%"
    )
    
    # Interpretation
    print_header("INTERPRETATION")
    
    bleu = metrics.get("eval_bleu", 0)
    rouge = metrics.get("eval_rouge", 0)
    exact = metrics.get("eval_exact_match", 0)
    token = metrics.get("eval_token_accuracy", 0)
    
    print("What these scores mean:\n")
    
    print(f"BLEU: {bleu:.1f}/100")
    if bleu >= 40:
        print("  → Good! Model generates reasonably similar scripts")
    elif bleu >= 30:
        print("  → Fair. Model learned some patterns")
    else:
        print("  → Low. Model may need more examples or epochs")
    
    print()
    
    print(f"ROUGE: {rouge:.1f}/100")
    if rouge >= 50:
        print("  → Good! Strong text overlap with references")
    elif rouge >= 40:
        print("  → Fair. Some structural similarity")
    else:
        print("  → Low. Structure differs significantly")
    
    print()
    
    print(f"Exact Match: {exact:.1f}%")
    if exact >= 30:
        print("  → Very good! Many scripts match exactly")
    elif exact >= 15:
        print("  → Normal. Exact matching is difficult for code")
    else:
        print("  → Expected. Exact match is hard but that's OK")
    
    print()
    
    print(f"Token Accuracy: {token:.1f}%")
    if token >= 60:
        print("  → Good! Most words positioned correctly")
    elif token >= 50:
        print("  → Fair. Partial word-level accuracy")
    else:
        print("  → Low. Words not in expected positions")
    
    # Overall assessment
    print_header("OVERALL ASSESSMENT")
    
    good_metrics = sum([
        bleu >= 40,
        rouge >= 50,
        token >= 60,
        # Note: exact match is not counted because it's often low
    ])
    
    if good_metrics >= 3:
        print("✅ EXCELLENT - Model is ready to use!")
        print("   All metrics are in good range.")
        print("   You can integrate this into your scenario agent.")
    elif good_metrics >= 2:
        print("✅ GOOD - Model is usable")
        print("   Most metrics are acceptable.")
        print("   Consider retraining with more examples for improvement.")
    elif good_metrics >= 1:
        print("⚠️ FAIR - Model shows learning")
        print("   Some metrics are below target.")
        print("   Consider: more training data, more epochs, or adjusting hyperparameters.")
    else:
        print("❌ POOR - Model may need retraining")
        print("   Metrics are all below acceptable ranges.")
        print("   Try: more diverse examples, higher learning rate, or more epochs.")
    
    # Recommendations
    print_header("NEXT STEPS")
    
    if good_metrics >= 2:
        print("1. ✓ Model is ready! Integrate into scenario agent:")
        print("     model_path = '/project/.../qwen-finetuned-final/'")
        print("     model = AutoModelForCausalLM.from_pretrained(model_path)")
        print()
        print("2. Test with sample requests from your scenario agent")
        print()
        print("3. (Optional) If you want even better results:")
        print("   - Collect 50+ more examples")
        print("   - Retrain with: epochs=5, learning_rate=1e-4")
    else:
        print("1. Analyze what went wrong:")
        print("   - Check training_data.jsonl format")
        print("   - Verify examples are clear and correct")
        print("   - Ensure validation_data.jsonl was created")
        print()
        print("2. Retrain with improvements:")
        print("   - Increase training examples (aim for 100+)")
        print("   - Try higher epochs (5 instead of 3)")
        print("   - Add more diverse test scenarios")
        print()
        print("3. Run again: python finetune_qwen_with_metrics.py")
    
    # Summary
    print_header("SUMMARY")
    
    summary = f"""
Your fine-tuning results:
  • Model: {results.get('model', 'Unknown')}
  • Training Examples: {results.get('training_examples', '?')}
  • BLEU Score: {bleu:.1f}/100
  • ROUGE Score: {rouge:.1f}/100
  • Exact Match: {exact:.1f}%
  • Token Accuracy: {token:.1f}%

Model saved to:
  /project/project_465003167/m10-testbot/qwen-finetuned-final/

Results saved to:
  {results_file}
  training_results/evaluation_report.txt
"""
    print(summary)
    
    # Save summary to file
    summary_file = "training_results/ANALYSIS_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("FINE-TUNING ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary)
    
    print(f"\n✓ Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()