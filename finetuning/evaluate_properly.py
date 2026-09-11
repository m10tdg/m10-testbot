# evaluate_properly.py
"""
Proper evaluation for Playwright test script generation.
Metrics that actually matter for your use case.
"""
import json
import ast
import re
from pathlib import Path
from collections import defaultdict

def extract_selectors_from_dom(input_text):
    """Extract all ID selectors from the DOM structure."""
    return set(re.findall(r'id="([^"]+)"', input_text))

def extract_selectors_from_script(script):
    """Extract selectors used in a Playwright script."""
    return set(re.findall(r"['\"]#([^'\"]+)['\"]", script))

def is_valid_python(code):
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def has_playwright_patterns(code):
    """Check for real Playwright API usage."""
    patterns = [
        'page.locator(',
        'page.fill(',
        'page.click(',
        'page.wait_for',
        'page.goto(',
        '.fill(',
        '.click(',
        '.select_option(',
        '.get_by_text(',
    ]
    return sum(1 for p in patterns if p in code)

def uses_dom_selectors(script, dom_selectors):
    """What fraction of available DOM selectors does the script use?"""
    if not dom_selectors:
        return 0.0
    used = extract_selectors_from_script(script)
    overlap = dom_selectors & used
    return len(overlap) / len(dom_selectors)

def is_generic_fallback(script):
    """Detect if the model generated the generic fallback pattern."""
    return '#test-element' in script

def evaluate_predictions(predictions_file):
    
    with open(predictions_file) as f:
        data = json.load(f)
    
    predictions = data['predictions']
    
    results = {
        'total': len(predictions),
        'valid_python': 0,
        'has_playwright_api': 0,        # At least 1 Playwright call
        'has_good_playwright_api': 0,   # 3+ Playwright calls
        'uses_dom_selectors': 0,        # Uses at least 1 correct ID
        'not_generic_fallback': 0,      # Avoids #test-element
        'has_error_handling': 0,        # Has try/except
        'is_code_not_prose': 0,         # Didn't generate English prose
        'selector_coverage': [],         # % of DOM selectors used
    }
    
    per_example = []
    
    for pred in predictions:
        script = pred.get('prediction', '')
        reference = pred.get('reference', '')
        input_text = pred.get('input', '')
        instruction = pred.get('instruction', '')
        
        dom_selectors = extract_selectors_from_dom(input_text)
        
        # Is it code at all? (not a multi-paragraph English essay)
        lines = [l.strip() for l in script.split('\n') if l.strip()]
        code_lines = sum(1 for l in lines if 
                        l.startswith('#') or 
                        '(' in l or 
                        l.startswith('page.') or
                        l.startswith('try:') or
                        l.startswith('except') or
                        l.startswith('assert'))
        is_code = code_lines > len(lines) * 0.5 and len(lines) < 100
        
        valid_py = is_valid_python(script)
        playwright_count = has_playwright_patterns(script)
        dom_coverage = uses_dom_selectors(script, dom_selectors)
        not_generic = not is_generic_fallback(script)
        has_try_except = 'try:' in script and 'except' in script
        
        if valid_py:
            results['valid_python'] += 1
        if playwright_count >= 1:
            results['has_playwright_api'] += 1
        if playwright_count >= 3:
            results['has_good_playwright_api'] += 1
        if dom_coverage > 0:
            results['uses_dom_selectors'] += 1
        if not_generic:
            results['not_generic_fallback'] += 1
        if has_try_except:
            results['has_error_handling'] += 1
        if is_code:
            results['is_code_not_prose'] += 1
        
        results['selector_coverage'].append(dom_coverage)
        
        per_example.append({
            'index': pred.get('index'),
            'instruction': instruction,
            'valid_python': valid_py,
            'playwright_calls': playwright_count,
            'dom_coverage': dom_coverage,
            'not_generic': not_generic,
            'has_error_handling': has_try_except,
            'is_code': is_code,
        })
    
    total = results['total']
    avg_coverage = sum(results['selector_coverage']) / total
    
    print("\n" + "="*60)
    print("PLAYWRIGHT SCRIPT GENERATION EVALUATION")
    print("="*60)
    print(f"\nTotal examples: {total}")
    print(f"\n{'Metric':<35} {'Count':>6} {'%':>7}")
    print("-"*50)
    print(f"{'Is code (not prose)':<35} {results['is_code_not_prose']:>6} {100*results['is_code_not_prose']/total:>6.1f}%")
    print(f"{'Valid Python syntax':<35} {results['valid_python']:>6} {100*results['valid_python']/total:>6.1f}%")
    print(f"{'Has any Playwright API call':<35} {results['has_playwright_api']:>6} {100*results['has_playwright_api']/total:>6.1f}%")
    print(f"{'Has 3+ Playwright calls':<35} {results['has_good_playwright_api']:>6} {100*results['has_good_playwright_api']/total:>6.1f}%")
    print(f"{'Uses DOM selector IDs':<35} {results['uses_dom_selectors']:>6} {100*results['uses_dom_selectors']/total:>6.1f}%")
    print(f"{'Has try/except':<35} {results['has_error_handling']:>6} {100*results['has_error_handling']/total:>6.1f}%")
    print(f"{'Avoids #test-element fallback':<35} {results['not_generic_fallback']:>6} {100*results['not_generic_fallback']/total:>6.1f}%")
    print(f"\n{'Avg DOM selector coverage':<35} {avg_coverage:>13.1%}")
    
    # Composite score
    composite = (
        results['is_code_not_prose'] * 0.25 +
        results['valid_python'] * 0.20 +
        results['has_good_playwright_api'] * 0.20 +
        results['uses_dom_selectors'] * 0.20 +
        results['has_error_handling'] * 0.15
    ) / total
    
    print(f"\n{'COMPOSITE SCORE':<35} {composite:>13.1%}")
    print("\n(Weighted: code 25%, valid Python 20%, Playwright API 20%, DOM selectors 20%, error handling 15%)")
    
    # Show worst examples
    failures = [e for e in per_example if not e['is_code'] or e['playwright_calls'] == 0]
    if failures:
        print(f"\n--- TOP FAILURES ({min(5, len(failures))} of {len(failures)}) ---")
        for f in failures[:5]:
            print(f"  [{f['index']}] {f['instruction']}")
            print(f"       Code: {f['is_code']}, Playwright calls: {f['playwright_calls']}, DOM: {f['dom_coverage']:.0%}")
    
    return results, per_example

if __name__ == '__main__':
    results, per_example = evaluate_predictions(
        'training_results/final_training_evaluation.json'
    )