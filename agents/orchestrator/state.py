from typing import TypedDict, Optional

class RunState(TypedDict, total=False):
    # From incoming event
    run_id: str
    tenant_id: str
    project_id: str
    url: str
    prompt: str
    correlation_id: str
    baseline_s3_path: Optional[str]
    attempt: int
    max_attempts: int
    is_retry: bool
    retry_context: dict
    
    # From crawler node
    page_structure: str
    page_title: str
    viewport_size: dict
    
    # From scenario node
    requirements_context: list[str]
    playwright_script: str
    
    # From execution node
    execution_results: list[dict]
    
    # From visual analysis node
    visual_diffs: list[dict]
    
    # From analysis node
    root_cause: Optional[str]
    severity: str
    recommendation: str
    
    # From reporting node
    report_url: str
    
    # Status tracking
    status: str
    error: Optional[str]
    _previous_error: Optional[str]
    _previous_state: Optional[dict]
