#!/usr/bin/env python3
"""
Proper evaluation for Playwright test script generation (Llama / Qwen / DeepSeek).

Usage:
    python evaluate_llama_properly.py
        -> evaluates training_results/evaluation_llama_finetuned.json

    python evaluate_llama_properly.py path/to/predictions.json
    python evaluate_llama_properly.py path/to/predictions.json --label "My model"

Input format: JSON with {"predictions": [...]} or a bare list. Each prediction
needs: "prediction", "input", and optionally "instruction", "reference", "index".

Metrics (same weights as the earlier evaluate_properly.py, so composite scores
are comparable across Qwen / DeepSeek / Llama), plus:
  - Exact match vs reference (after normalization)
  - Hallucinated selectors: IDs used in the script that are NOT in the DOM
"""
import argparse
import ast
import json
import re
from pathlib import Path

DEFAULT_FILE = "training_results/evaluation_llama_finetuned.json"
DEFAULT_LABEL = "Llama-3.1-8B-Instruct (fine-tuned)"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def strip_prediction(text: str) -> str:
    """Normalize a prediction: strip whitespace, code fences, extra blank lines."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_selectors_from_dom(input_text: str):
    return set(re.findall(r'id="([^"]+)"', input_text or ""))


def extract_selectors_from_script(script: str):
    return set(re.findall(r"['\"]#([^'\"]+)['\"]", script))


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, IndentationError, ValueError):
        return False


def has_playwright_patterns(code: str) -> int:
    patterns = [
        "page.locator(",
        "page.fill(",
        "page.click(",
        "page.wait_for",
        "page.goto(",
        ".fill(",
        ".click(",
        ".check(",
        ".select_option(",
        ".get_by_text(",
    ]
    return sum(1 for p in patterns if p in code)


def dom_coverage(script: str, dom_selectors: set) -> float:
    if not dom_selectors:
        return 0.0
    used = extract_selectors_from_script(script)
    return len(dom_selectors & used) / len(dom_selectors)


def is_generic_fallback(script: str) -> bool:
    return "#test-element" in script


def looks_like_code(script: str) -> bool:
    lines = [l.strip() for l in script.split("\n") if l.strip()]
    if not lines:
        return False
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
    return code_lines > len(lines) * 0.5 and len(lines) < 100


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate_predictions(predictions_file: str, model_label: str):
    with open(predictions_file, encoding="utf-8") as f:
        data = json.load(f)

    predictions = data if isinstance(data, list) else data.get("predictions", [])
    total = len(predictions)

    counts = {
        "is_code": 0,
        "valid_python": 0,
        "any_playwright": 0,
        "good_playwright": 0,
        "uses_dom_selectors": 0,
        "has_try_except": 0,
        "not_generic": 0,
        "exact_match": 0,
        "no_hallucinated_selectors": 0,
    }
    coverages = []
    per_example = []

    for pred in predictions:
        script = strip_prediction(pred.get("prediction", ""))
        reference = strip_prediction(pred.get("reference", ""))
        input_text = pred.get("input", "")

        dom_ids = extract_selectors_from_dom(input_text)
        used_ids = extract_selectors_from_script(script)
        hallucinated = sorted(used_ids - dom_ids)

        is_code = looks_like_code(script)
        valid = is_valid_python(script)
        pw_count = has_playwright_patterns(script)
        cov = dom_coverage(script, dom_ids)
        not_generic = not is_generic_fallback(script)
        try_except = "try:" in script and "except" in script
        exact = bool(reference) and script == reference

        counts["is_code"] += is_code
        counts["valid_python"] += valid
        counts["any_playwright"] += pw_count >= 1
        counts["good_playwright"] += pw_count >= 3
        counts["uses_dom_selectors"] += cov > 0
        counts["has_try_except"] += try_except
        counts["not_generic"] += not_generic
        counts["exact_match"] += exact
        counts["no_hallucinated_selectors"] += (len(hallucinated) == 0)
        coverages.append(cov)

        per_example.append({
            "index": pred.get("index"),
            "instruction": pred.get("instruction", ""),
            "is_code": is_code,
            "valid_python": valid,
            "playwright_calls": pw_count,
            "dom_coverage": cov,
            "hallucinated_selectors": hallucinated,
            "has_error_handling": try_except,
            "exact_match": exact,
            "prediction_stripped": script,
            "reference_stripped": reference,
        })

    avg_cov = sum(coverages) / total if total else 0.0

    composite = (
        counts["is_code"] * 0.25
        + counts["valid_python"] * 0.20
        + counts["good_playwright"] * 0.20
        + counts["uses_dom_selectors"] * 0.20
        + counts["has_try_except"] * 0.15
    ) / total if total else 0.0

    # ---- Print report ----
    print("\n" + "=" * 60)
    print(f"PLAYWRIGHT SCRIPT GENERATION EVALUATION  -  {model_label}")
    print("=" * 60)
    print(f"\nTotal examples: {total}")
    print(f"\n{'Metric':<35} {'Count':>6} {'%':>7}")
    print("-" * 50)

    rows = [
        ("Is code (not prose)", counts["is_code"]),
        ("Valid Python syntax", counts["valid_python"]),
        ("Has any Playwright API call", counts["any_playwright"]),
        ("Has 3+ Playwright calls", counts["good_playwright"]),
        ("Uses DOM selector IDs", counts["uses_dom_selectors"]),
        ("Has try/except", counts["has_try_except"]),
        ("Avoids #test-element fallback", counts["not_generic"]),
        ("No hallucinated selectors", counts["no_hallucinated_selectors"]),
        ("Exact match vs reference", counts["exact_match"]),
    ]
    for label, count in rows:
        pct = 100 * count / total if total else 0
        print(f"{label:<35} {count:>6} {pct:>6.1f}%")

    print(f"\n{'Avg DOM selector coverage':<35} {avg_cov:>13.1%}")
    print(f"\n{'COMPOSITE SCORE':<35} {composite:>13.1%}")
    print(
        "\n(Weighted: code 25%, valid Python 20%, 3+ Playwright calls 20%, "
        "DOM selectors 20%, error handling 15%)"
    )

    # ---- Failures ----
    hard_failures = [e for e in per_example if not e["is_code"] or e["playwright_calls"] == 0]
    if hard_failures:
        print(f"\n--- HARD FAILURES ({min(5, len(hard_failures))} of {len(hard_failures)}) ---")
        for e in hard_failures[:5]:
            print(f"  [{e['index']}] {e['instruction']}")
    else:
        print("\nNo hard failures (every example is code with >=1 Playwright call)")

    halluc = [e for e in per_example if e["hallucinated_selectors"]]
    if halluc:
        print(f"\n--- HALLUCINATED SELECTORS ({min(5, len(halluc))} of {len(halluc)}) ---")
        for e in halluc[:5]:
            print(f"  [{e['index']}] {e['instruction']}")
            print(f"       Not in DOM: {e['hallucinated_selectors']}")

    non_exact = [e for e in per_example if not e["exact_match"]]
    if non_exact:
        print(f"\n--- NON-EXACT MATCHES ({min(5, len(non_exact))} of {len(non_exact)}) ---")
        for e in non_exact[:5]:
            print(f"  [{e['index']}] {e['instruction']}")

    # ---- Save ----
    out_path = Path(predictions_file).with_name(
        Path(predictions_file).stem + "_properly_scored.json"
    )
    summary = {
        "model": model_label,
        "total": total,
        "counts": counts,
        "rates": {k: (round(v / total, 4) if total else 0) for k, v in counts.items()},
        "avg_dom_selector_coverage": round(avg_cov, 4),
        "composite_score": round(composite, 4),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, indent=2, ensure_ascii=False)
    print(f"\nScored results saved to:\n  {out_path}")

    return summary, per_example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Playwright generation predictions.")
    parser.add_argument("predictions_file", nargs="?", default=DEFAULT_FILE,
                        help=f"Predictions JSON (default: {DEFAULT_FILE})")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Model label for the report")
    args = parser.parse_args()

    evaluate_predictions(args.predictions_file, args.label)