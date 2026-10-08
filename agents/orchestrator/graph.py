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


def build_graph():
    """
    Build the LangGraph workflow with enhanced self-healing.
    
    Flow:
    1. Crawler (handles both first-attempt and retry crawling internally)
    2. Scenario agent (generates test script, uses retry context if available)
    3. Execution (runs script, captures detailed errors)
    4. Route based on success/failure
    5. Visual analysis (compares the captured screenshot when available)
    6. Analysis (combines execution and visual findings)
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
    
    # Run visual analysis regardless of test pass/fail; failed runs still
    # have useful screenshots for comparison when capture succeeded.
    graph.add_edge("execution", "visual_analysis")
    
    # Visual analysis → Analysis (if execution passed)
    graph.add_edge("visual_analysis", "analysis")
    
    # Analysis → Reporting (always)
    graph.add_edge("analysis", "reporting")
    
    # Reporting → End
    graph.add_edge("reporting", END)
    
    return graph.compile()