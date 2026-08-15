import openai

def analysis_agent_node(state: dict) -> dict:
    print(f"[analysis-agent] analyzing results for run {state['run_id']}")
    
    failures = [r for r in state.get("execution_results", []) if not r.get("passed", True)]
    
    # Handle case where visual_analysis node was skipped
    visual_diffs = state.get("visual_diffs") or []
    visual_issues = [d for d in visual_diffs if d.get("verdict") == "needs_review"]
    
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
    
    # Build error details for LLM
    failure_details = "\n".join([
        f"- {f.get('error', 'Unknown error')} (context: {f.get('error_context', 'N/A')})"
        for f in failures
    ])
    
    prompt = f"""Test failures detected:
{failure_details}

Visual issues: {len(visual_issues)} found

Provide a short root cause analysis for a developer:
1) Most likely root cause (be specific)
2) Severity: critical, warning, or info
3) A concrete, actionable recommendation
4) Which page element or interaction failed"""
    
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