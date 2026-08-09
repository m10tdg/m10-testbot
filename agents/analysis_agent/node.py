# agents/analysis_agent/node.py
from openai import OpenAI
from libs.shared_py.rag import retrieve_context

client = OpenAI()

def analysis_agent_node(state: dict, qdrant_client) -> dict:
    failures = [r for r in state["execution_results"] if not r.get("passed", True)]
    visual_issues = [d for d in state.get("visual_diffs", []) if d["verdict"] == "needs_review"]

    if not failures and not visual_issues:
        return {**state, "root_cause": None, "severity": "info",
                "recommendation": "No issues detected.", "status": "analyzing"}

    # Pull historical failures for this project — this is the "learning from past runs" piece
    historical = retrieve_context(
        qdrant_client, state["tenant_id"], state["project_id"],
        query_vector=embed(str(failures)), types=["historical_failure"], top_k=3,
    )

    prompt = f"""Failures: {failures}
Visual issues: {visual_issues}
Similar past incidents in this project: {historical}

Provide: 1) most likely root cause, 2) severity (critical/warning/info), 3) a concrete
recommendation for the developer. Be specific — reference the failing page and element type."""

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )
    text = completion.choices[0].message.content

    # In production, ask the model for structured JSON output rather than parsing prose.
    return {
        **state,
        "root_cause": text,
        "severity": "critical" if failures else "warning",
        "recommendation": text,
        "status": "analyzing",
    }