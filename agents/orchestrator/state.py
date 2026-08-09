"""
RunState is the single object that flows through every node in the graph.
Each node reads what it needs from it and returns an updated copy with its
own results added. Nothing is shared between nodes except through this object.
"""

from typing import TypedDict, Optional


class RunState(TypedDict, total=False):
    # set at the start, from the incoming test.requested event
    run_id: str
    tenant_id: str
    project_id: str
    url: str
    prompt: str
    correlation_id: str

    # filled in by scenario node
    requirements_context: list[str]
    playwright_script: str

    # filled in by execution node
    execution_results: list[dict]

    # filled in by visual analysis node
    visual_diffs: list[dict]

    # filled in by analysis node
    root_cause: str
    severity: str
    recommendation: str

    # filled in by reporting node
    report_url: str

    status: str      # queued -> scenario_generated -> executing -> analyzing -> completed/failed
    error: Optional[str]
