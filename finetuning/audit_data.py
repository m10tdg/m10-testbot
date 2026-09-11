# audit_data.py
import json
from pathlib import Path
from collections import Counter

data_file = Path("dataset/training_data.jsonl")
examples = []

with open(data_file) as f:
    for line in f:
        examples.append(json.loads(line))

# Check for generic test-element outputs
generic_count = sum(1 for e in examples if '#test-element' in e['output'])
specific_count = sum(1 for e in examples if '#test-element' not in e['output'])

print(f"Total examples: {len(examples)}")
print(f"Generic (#test-element): {generic_count} ({100*generic_count/len(examples):.1f}%)")
print(f"Specific (DOM-aware): {specific_count} ({100*specific_count/len(examples):.1f}%)")

# Show distribution of output types
keywords = ['locator(', 'fill(', 'click(', 'select_option(', 'wait_for_url(', 'get_by_text(']
for kw in keywords:
    count = sum(1 for e in examples if kw in e['output'])
    print(f"  '{kw}': {count} examples")