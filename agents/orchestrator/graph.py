"""
LangGraph v3 - TESTBOT Enhanced

Improvements:
1. Crawler internally handles re-crawl logic on retry attempts
2. Better routing based on execution success
3. Status field properly tracked through pipeline
4. Cleaner state flow for self-healing
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
    """
    Route based on execution results.
    
    Success: visual_analysis → analysis → reporting
    Failure: analysis (skip visual) → reporting
    """
    execution_results = state.get("execution_results", [])
    
    if execution_results:
        result = execution_results[0]
        if result.get("passed", False):
            print("[graph] Execution passed, proceeding to visual analysis")
            return "visual_analysis"
        else:
            print("[graph] Execution failed, skipping visual analysis")
            return "analysis"
    
    # Default to analysis if no results (shouldn't happen)
    return "analysis"


def build_graph():
    """
    Build the LangGraph workflow with enhanced self-healing.
    
    Flow:
    1. Crawler (handles both first-attempt and retry crawling internally)
    2. Scenario agent (generates test script, uses retry context if available)
    3. Execution (runs script, captures detailed errors)
    4. Route based on success/failure
    5. Visual analysis (if passed) OR Analysis (if failed)
    6. Analysis (root cause for failures)
    7. Reporting (generate HTML report)
    8. END
    
    Note: The crawler node checks state["attempt"] internally:
    - Attempt 1: Single page crawl
    - Attempt 2+: Re-crawl to get fresh page state
    """
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
    # Crawler → Scenario (always)
    graph.add_edge("crawler", "scenario")
    
    # Scenario → Execution (always)
    graph.add_edge("scenario", "execution")
    
    # Conditional routing after execution
    graph.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "visual_analysis": "visual_analysis",
            "analysis": "analysis",
        }
    )
    
    # Visual analysis → Analysis (if execution passed)
    graph.add_edge("visual_analysis", "analysis")
    
    # Analysis → Reporting (always)
    graph.add_edge("analysis", "reporting")
    
    # Reporting → End
    graph.add_edge("reporting", END)
    
    return graph.compile()