#!/usr/bin/env python3
"""
Validate training data before fine-tuning
Checks:
- JSONL format validity
- Required fields
- Data quality
- Length statistics
- Duplicate detection
"""

import json
import sys
from pathlib import Path
from collections import Counter

def validate_jsonl_file(filepath, max_display=5):
    """Validate a single JSONL file"""
    
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return False, {}
    
    print(f"\n{'='*80}")
    print(f"VALIDATING: {filepath}")
    print(f"{'='*80}")
    
    stats = {
        'total': 0,
        'valid': 0,
        'errors': [],
        'lengths': [],
        'instruction_lengths': [],
        'input_lengths': [],
        'output_lengths': [],
        'empty_fields': Counter(),
    }
    
    examples_to_show = []
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                stats['total'] += 1
                
                try:
                    example = json.loads(line)
                    
                    # Check required fields
                    required_fields = ['instruction', 'input', 'output']
                    missing = [f for f in required_fields if f not in example]
                    
                    if missing:
                        stats['errors'].append(f"Line {line_num}: Missing {missing}")
                        continue
                    
                    instruction = str(example.get('instruction', '')).strip()
                    input_text = str(example.get('input', '')).strip()
                    output = str(example.get('output', '')).strip()
                    
                    # Check for empty fields
                    if not instruction:
                        stats['empty_fields']['instruction'] += 1
                    if not input_text:
                        stats['empty_fields']['input'] += 1
                    if not output:
                        stats['empty_fields']['output'] += 1
                    
                    # Build full text
                    full_text = f"[INST] {instruction}\n\n{input_text} [/INST]\n{output}"
                    
                    # Calculate lengths
                    full_len = len(full_text.split())
                    inst_len = len(instruction.split())
                    inp_len = len(input_text.split())
                    out_len = len(output.split())
                    
                    stats['lengths'].append(full_len)
                    stats['instruction_lengths'].append(inst_len)
                    stats['input_lengths'].append(inp_len)
                    stats['output_lengths'].append(out_len)
                    
                    stats['valid'] += 1
                    
                    # Save first few examples
                    if len(examples_to_show) < max_display:
                        examples_to_show.append({
                            'line': line_num,
                            'instruction': instruction[:60],
                            'output_preview': output[:60],
                            'total_tokens': full_len,
                        })
                
                except json.JSONDecodeError as e:
                    stats['errors'].append(f"Line {line_num}: Invalid JSON: {e}")
                
                except Exception as e:
                    stats['errors'].append(f"Line {line_num}: {e}")
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False, stats
    
    # Print results
    print(f"\n📊 SUMMARY:")
    print(f"  Total lines:     {stats['total']}")
    print(f"  Valid examples:  {stats['valid']} ✓")
    print(f"  Errors:          {len(stats['errors'])} ❌")
    
    if stats['errors']:
        print(f"\n⚠️  ERRORS:")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"    {error}")
        if len(stats['errors']) > 10:
            print(f"    ... and {len(stats['errors']) - 10} more")
    
    # Token length statistics
    if stats['lengths']:
        import numpy as np
        
        print(f"\n📏 TOKEN STATISTICS:")
        print(f"  Full text (instruction+input+output):")
        print(f"    Mean:        {np.mean(stats['lengths']):.0f}")
        print(f"    Median:      {np.median(stats['lengths']):.0f}")
        print(f"    Min:         {np.min(stats['lengths']):.0f}")
        print(f"    Max:         {np.max(stats['lengths']):.0f}")
        print(f"    95th %ile:   {np.percentile(stats['lengths'], 95):.0f}")
        
        print(f"\n  Instruction only:")
        print(f"    Mean:        {np.mean(stats['instruction_lengths']):.0f}")
        print(f"    Max:         {np.max(stats['instruction_lengths']):.0f}")
        
        print(f"\n  Input only:")
        print(f"    Mean:        {np.mean(stats['input_lengths']):.0f}")
        print(f"    Max:         {np.max(stats['input_lengths']):.0f}")
        
        print(f"\n  Output only:")
        print(f"    Mean:        {np.mean(stats['output_lengths']):.0f}")
        print(f"    Max:         {np.max(stats['output_lengths']):.0f}")
        
        recommended_max = int(np.percentile(stats['lengths'], 95) * 1.2)
        print(f"\n  → Recommended MAX_SEQ_LENGTH: {min(512, max(256, recommended_max))}")
    
    # Check for empty fields
    if any(stats['empty_fields'].values()):
        print(f"\n⚠️  EMPTY FIELDS:")
        for field, count in stats['empty_fields'].items():
            if count > 0:
                print(f"    {field}: {count} examples")
    
    # Show examples
    if examples_to_show:
        print(f"\n📝 EXAMPLE LINES:")
        for ex in examples_to_show:
            print(f"\n  Line {ex['line']} ({ex['total_tokens']} tokens):")
            print(f"    Instruction: {ex['instruction']}...")
            print(f"    Output:      {ex['output_preview']}...")
    
    return stats['valid'] > 0, stats


def validate_data_quality(filepath):
    """Check for data quality issues"""
    
    print(f"\n{'='*80}")
    print("DATA QUALITY CHECK")
    print(f"{'='*80}")
    
    outputs = []
    instructions = []
    duplicates = []
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line)
                    output = example.get('output', '').strip()
                    instruction = example.get('instruction', '').strip()
                    
                    if output:
                        outputs.append((line_num, output))
                    if instruction:
                        instructions.append((line_num, instruction))
                
                except:
                    pass
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Check for duplicates
    output_set = set()
    for line_num, output in outputs:
        if output in output_set:
            duplicates.append((line_num, output[:50]))
        output_set.add(output)
    
    if duplicates:
        print(f"\n⚠️  DUPLICATE OUTPUTS ({len(duplicates)}):")
        for line_num, output in duplicates[:5]:
            print(f"    Line {line_num}: {output}...")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")
    else:
        print(f"\n✓ No duplicate outputs")
    
    # Check for very short outputs
    short_outputs = [(line_num, output) for line_num, output in outputs if len(output.split()) < 5]
    if short_outputs:
        print(f"\n⚠️  VERY SHORT OUTPUTS ({len(short_outputs)}):")
        for line_num, output in short_outputs[:5]:
            print(f"    Line {line_num}: {output}")
    else:
        print(f"\n✓ All outputs have reasonable length")
    
    # Check for very long outputs
    long_outputs = [(line_num, len(output.split())) for line_num, output in outputs if len(output.split()) > 1000]
    if long_outputs:
        print(f"\n⚠️  VERY LONG OUTPUTS ({len(long_outputs)}):")
        for line_num, length in long_outputs[:5]:
            print(f"    Line {line_num}: {length} tokens")
    else:
        print(f"\n✓ All outputs are reasonably sized")
    
    # Check for diversity
    print(f"\n📊 DIVERSITY CHECK:")
    print(f"    Unique outputs:      {len(output_set)} / {len(outputs)}")
    print(f"    Unique instructions: {len(set(instructions))} / {len(instructions)}")


def main():
    print("\n" + "="*80)
    print("QWEN FINE-TUNING DATA VALIDATION")
    print("="*80)
    
    data_dir = Path("dataset")
    
    if not data_dir.exists():
        print(f"\n❌ Dataset directory not found: {data_dir}")
        print("   Run this script from: /project/project_465003167/m10-testbot/finetuning/")
        sys.exit(1)
    
    # Check training data
    train_file = data_dir / "training_data.jsonl"
    print(f"\n[1/3] Checking training data...")
    train_ok, train_stats = validate_jsonl_file(train_file)
    
    # Check validation data
    val_file = data_dir / "validation_data.jsonl"
    print(f"\n[2/3] Checking validation data...")
    val_ok, val_stats = validate_jsonl_file(val_file, max_display=3)
    
    # Check quality
    print(f"\n[3/3] Analyzing data quality...")
    validate_data_quality(train_file)
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    all_ok = train_ok and (not val_file.exists() or val_ok)
    
    if not all_ok:
        print("\n❌ ISSUES FOUND - Fix before training!")
        if not train_ok:
            print("   • Training data has errors")
        if val_file.exists() and not val_ok:
            print("   • Validation data has errors")
        sys.exit(1)
    
    else:
        print("\n✅ DATA LOOKS GOOD!")
        print("\nYou can now run fine-tuning:")
        print("   python finetune_qwen_fixed.py")
        
        if train_stats['valid'] < 100:
            print(f"\n⚠️  Note: You have only {train_stats['valid']} examples")
            print("    Consider collecting more data for better results")
        
        if train_stats['valid'] > 1000:
            print(f"\n⚠️  Note: You have {train_stats['valid']} examples")
            print("    Consider reducing dataset for faster training")


if __name__ == "__main__":
    main()