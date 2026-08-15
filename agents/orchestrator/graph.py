"""
Enhanced LangGraph - TESTBOT v2

Node pipeline:
1. crawler         -> Crawl live URL, extract interactive elements
2. scenario        -> Generate Playwright script using crawler output + RAG context
3. execution       -> Run script, capture errors and diagnostics
4. visual_analysis -> Compare screenshots against baselines
5. analysis        -> LLM-powered root cause analysis
6. reporting       -> Generate HTML report

Self-healing:
- If execution fails, orchestrator retries with error context
- No loops in graph - errors handled at orchestrator level
"""

from langgraph.graph import StateGraph, END
from state import RunState
from nodes.crawler import crawler_node
from nodes.scenario import scenario_agent_node
from nodes.execution import execution_agent_node
from nodes.visual_analysis import visual_analysis_agent_node
from nodes.analysis import analysis_agent_node
from nodes.reporting import reporting_agent_node


def route_after_execution(state: RunState) -> str:
    """Route based on execution results."""
    # If execution failed, skip visual analysis and go straight to analysis+reporting
    if state.get("status") == "failed":
        return "analysis"
    # Otherwise proceed normally
    return "visual_analysis"


def build_graph():
    """Build the LangGraph workflow."""
    graph = StateGraph(RunState)
    
    # Add all nodes
    graph.add_node("crawler", crawler_node)
    graph.add_node("scenario", scenario_agent_node)
    graph.add_node("execution", execution_agent_node)
    graph.add_node("visual_analysis", visual_analysis_agent_node)
    graph.add_node("analysis", analysis_agent_node)
    graph.add_node("reporting", reporting_agent_node)
    
    # Set entry point
    graph.set_entry_point("crawler")
    
    # Define edges
    graph.add_edge("crawler", "scenario")
    graph.add_edge("scenario", "execution")
    graph.add_conditional_edges("execution", route_after_execution, {
        "visual_analysis": "visual_analysis",
        "analysis": "analysis",
    })
    graph.add_edge("visual_analysis", "analysis")
    graph.add_edge("analysis", "reporting")
    graph.add_edge("reporting", END)
    
    return graph.compile()