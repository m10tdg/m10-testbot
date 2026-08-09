# agents/orchestrator/state.py
from typing import TypedDict, Optional
from langgraph.graph import add_messages
from typing_extensions import Annotated

class RunState(TypedDict):
    run_id: str
    tenant_id: str
    project_id: str
    url: str
    prompt: str
    correlation_id: str

    # filled in progressively by each node
    requirements_context: Optional[list[str]]
    playwright_script: Optional[str]
    scenario_count: Optional[int]

    execution_results: Optional[list[dict]]     # per-page: loadTime, consoleErrors, dom_s3_path, screenshot_s3_path
    visual_diffs: Optional[list[dict]]           # per-page diff results

    root_cause: Optional[str]
    severity: Optional[str]
    recommendation: Optional[str]

    report_url: Optional[str]
    status: str    # mirrors test_runs.status in Postgres
    error: Optional[str]