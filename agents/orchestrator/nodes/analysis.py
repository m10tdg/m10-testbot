import openai


def analysis_agent_node(state: dict) -> dict:
    print(f"[analysis-agent] analyzing results for run {state['run_id']}")

    failures = [r for r in state["execution_results"] if not r.get("passed", True)]
    visual_issues = [d for d in state.get("visual_diffs", []) if d["verdict"] == "needs_review"]

    print(f"[analysis-agent] found {len(failures)} test failures and {len(visual_issues)} visual issues")

    if not failures and not visual_issues:
        print("[analysis-agent] no issues found")
        return {
            **state,
            "root_cause": None,
            "severity": "info",
            "recommendation": "No issues detected. All checks passed.",
            "status": "analyzing",
        }

    prompt = f"""Test failures: {failures}
Visual issues: {visual_issues}

Provide a short root cause analysis for a developer:
1) Most likely root cause
2) Severity: critical, warning, or info
3) A concrete, actionable recommendation

Be specific about which page/element is affected."""

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    analysis_text = completion.choices[0].message.content

    severity = "critical" if failures else "warning"
    print(f"[analysis-agent] severity={severity}")

    print(f"[analysis-agent] root cause analysis:\n{analysis_text}")

    return {
        **state,
        "root_cause": analysis_text,
        "severity": severity,
        "recommendation": analysis_text,
        "status": "analyzing",
    }
