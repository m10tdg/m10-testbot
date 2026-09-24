# evaluate_properly.py
"""
Proper evaluation for Playwright test script generation.
Works correctly for both Qwen and DeepSeek fine-tuned models.

Key fix vs original:
  DeepSeek predictions often start with a leading space/newline before the
  first line of code (an artifact of the chat template's generation prompt).
  ast.parse() raises IndentationError on "  page.locator(...)" at the top
  level, causing Valid Python to show 0%.  We now strip() every prediction
  before any analysis step.
"""
import json
import ast
import re
from pathlib import Path


def strip_prediction(text: str) -> str:
    """
    Normalise a model prediction before any metric is applied.

    1. Strip leading/trailing whitespace (fixes DeepSeek's leading space).
    2. Remove markdown code fences if the model accidentally emits them.
    3. Collapse runs of 3+ blank lines to a single blank line.
    """
    text = text.strip()
    # Remove ```python ... ``` or ``` ... ``` wrappers.
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_selectors_from_dom(input_text: str):
    """Extract all ID selectors from the DOM structure."""
    return set(re.findall(r'id="([^"]+)"', input_text))


def extract_selectors_from_script(script: str):
    """Extract selectors used in a Playwright script."""
    return set(re.findall(r"['\"]#([^'\"]+)['\"]", script))


def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, IndentationError):
        return False


def has_playwright_patterns(code: str) -> int:
    """Count real Playwright API usage patterns present in code."""
    patterns = [
        "page.locator(",
        "page.fill(",
        "page.click(",
        "page.wait_for",
        "page.goto(",
        ".fill(",
        ".click(",
        ".select_option(",
        ".get_by_text(",
    ]
    return sum(1 for p in patterns if p in code)


def uses_dom_selectors(script: str, dom_selectors: set) -> float:
    """What fraction of available DOM selectors does the script use?"""
    if not dom_selectors:
        return 0.0
    used = extract_selectors_from_script(script)
    overlap = dom_selectors & used
    return len(overlap) / len(dom_selectors)


def is_generic_fallback(script: str) -> bool:
    """Detect if the model generated the generic fallback pattern."""
    return "#test-element" in script


def evaluate_predictions(predictions_file: str, model_label: str = "Model"):

    with open(predictions_file) as f:
        data = json.load(f)

    # Support both formats:
    #   {"predictions": [...]}          (evaluate_deepseek.py output)
    #   {"summary": {...}, "predictions": [...]}  (same)
    #   [...]                           (bare list)
    if isinstance(data, list):
        predictions = data
    else:
        predictions = data.get("predictions", [])

    results = {
        "total":                  len(predictions),
        "valid_python":           0,
        "has_playwright_api":     0,
        "has_good_playwright_api":0,
        "uses_dom_selectors":     0,
        "not_generic_fallback":   0,
        "has_error_handling":     0,
        "is_code_not_prose":      0,
        "selector_coverage":      [],
    }

    per_example = []

    for pred in predictions:
        raw_script  = pred.get("prediction", "")
        script      = strip_prediction(raw_script)      # <-- KEY FIX
        input_text  = pred.get("input", "")
        instruction = pred.get("instruction", "")

        dom_selectors  = extract_selectors_from_dom(input_text)

        # Is it code at all? (not a multi-paragraph English essay)
        lines = [l.strip() for l in script.split("\n") if l.strip()]
        code_lines = sum(
            1 for l in lines
            if (
                l.startswith("#")
                or "(" in l
                or l.startswith("page.")
                or l.startswith("try:")
                or l.startswith("except")
                or l.startswith("assert")
            )
        )
        is_code = code_lines > len(lines) * 0.5 and len(lines) < 100

        valid_py        = is_valid_python(script)
        playwright_count = has_playwright_patterns(script)
        dom_coverage    = uses_dom_selectors(script, dom_selectors)
        not_generic     = not is_generic_fallback(script)
        has_try_except  = "try:" in script and "except" in script

        if valid_py:           results["valid_python"]            += 1
        if playwright_count >= 1: results["has_playwright_api"]   += 1
        if playwright_count >= 3: results["has_good_playwright_api"] += 1
        if dom_coverage > 0:   results["uses_dom_selectors"]      += 1
        if not_generic:        results["not_generic_fallback"]    += 1
        if has_try_except:     results["has_error_handling"]      += 1
        if is_code:            results["is_code_not_prose"]       += 1

        results["selector_coverage"].append(dom_coverage)

        per_example.append({
            "index":             pred.get("index"),
            "instruction":       instruction,
            "valid_python":      valid_py,
            "playwright_calls":  playwright_count,
            "dom_coverage":      dom_coverage,
            "not_generic":       not_generic,
            "has_error_handling":has_try_except,
            "is_code":           is_code,
            # Store the stripped version so failures are easy to read.
            "prediction_stripped": script,
        })

    total        = results["total"]
    avg_coverage = sum(results["selector_coverage"]) / total if total else 0

    print("\n" + "=" * 60)
    print(f"PLAYWRIGHT SCRIPT GENERATION EVALUATION  —  {model_label}")
    print("=" * 60)
    print(f"\nTotal examples: {total}")
    print(f"\n{'Metric':<35} {'Count':>6} {'%':>7}")
    print("-" * 50)

    rows = [
        ("Is code (not prose)",            results["is_code_not_prose"],       "25.0%"),
        ("Valid Python syntax",             results["valid_python"],            "20.0%"),
        ("Has any Playwright API call",     results["has_playwright_api"],      "20.0%"),
        ("Has 3+ Playwright calls",         results["has_good_playwright_api"], "secondary"),
        ("Uses DOM selector IDs",           results["uses_dom_selectors"],      "20.0%"),
        ("Has try/except",                  results["has_error_handling"],      "15.0%"),
        ("Avoids #test-element fallback",   results["not_generic_fallback"],    "sanity"),
    ]

    for label, count, weight in rows:
        pct = 100 * count / total if total else 0
        print(f"{label:<35} {count:>6} {pct:>6.1f}%")

    print(f"\n{'Avg DOM selector coverage':<35} {avg_coverage:>13.1%}")

    # Composite score (identical weights to original evaluate_properly.py).
    composite = (
        results["is_code_not_prose"]      * 0.25 +
        results["valid_python"]           * 0.20 +
        results["has_good_playwright_api"]* 0.20 +
        results["uses_dom_selectors"]     * 0.20 +
        results["has_error_handling"]     * 0.15
    ) / total if total else 0

    print(f"\n{'COMPOSITE SCORE':<35} {composite:>13.1%}")
    print(
        "\n(Weighted: code 25%, valid Python 20%, "
        "Playwright API 20%, DOM selectors 20%, error handling 15%)"
    )

    # Show worst examples.
    failures = [
        e for e in per_example
        if not e["is_code"] or e["playwright_calls"] == 0
    ]
    if failures:
        print(f"\n--- TOP FAILURES ({min(5, len(failures))} of {len(failures)}) ---")
        for f in failures[:5]:
            print(f"  [{f['index']}] {f['instruction']}")
            print(
                f"       Code: {f['is_code']}, "
                f"Playwright calls: {f['playwright_calls']}, "
                f"DOM: {f['dom_coverage']:.0%}"
            )
    else:
        print("\n✓ No hard failures (every example is code with ≥1 Playwright call)")

    return results, per_example


if __name__ == "__main__":
    import sys

    # Usage:
    #   python evaluate_properly.py                          <- DeepSeek default
    #   python evaluate_properly.py qwen                     <- Qwen results
    #   python evaluate_properly.py path/to/file.json label  <- explicit path

    if len(sys.argv) == 3:
        predictions_file = sys.argv[1]
        model_label      = sys.argv[2]
    elif len(sys.argv) == 2 and sys.argv[1].lower() == "qwen":
        predictions_file = "training_results/evaluation_finetuned.json"
        model_label      = "Qwen2.5-14B-Instruct (fine-tuned)"
    else:
        predictions_file = "training_results/evaluation_deepseek_finetuned.json"
        model_label      = "DeepSeek-Coder-V2-Lite-Instruct (fine-tuned)"

    evaluate_predictions(predictions_file, model_label)