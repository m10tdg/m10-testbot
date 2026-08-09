# agents/orchestrator/graph.py
from langgraph.graph import StateGraph, END
from state import RunState
from agents.scenario_agent.node import scenario_agent_node
from agents.execution_agent.node import execution_agent_node
from agents.visual_analysis_agent.node import visual_analysis_agent_node
from agents.analysis_agent.node import analysis_agent_node
from agents.reporting_agent.node import reporting_agent_node

def route_after_execution(state: RunState) -> str:
    """Human-in-the-Loop and retry logic live in these edge functions, per the doc's
    'halting the process in cases requiring human approval' requirement."""
    if state["status"] == "failed":
        return "retry_or_halt"
    return "visual_analysis"

def retry_or_halt_node(state: RunState) -> dict:
    # simplistic: halt after first failure. Add a retry counter in RunState for real retry logic.
    return {**state, "status": "failed"}

def build_graph(qdrant_client):
    graph = StateGraph(RunState)

    graph.add_node("scenario", lambda s: scenario_agent_node(s, qdrant_client))
    graph.add_node("execution", execution_agent_node)
    graph.add_node("visual_analysis", visual_analysis_agent_node)
    graph.add_node("analysis", lambda s: analysis_agent_node(s, qdrant_client))
    graph.add_node("reporting", reporting_agent_node)
    graph.add_node("retry_or_halt", retry_or_halt_node)

    graph.set_entry_point("scenario")
    graph.add_edge("scenario", "execution")
    graph.add_conditional_edges("execution", route_after_execution, {
        "visual_analysis": "visual_analysis",
        "retry_or_halt": "retry_or_halt",
    })
    graph.add_edge("visual_analysis", "analysis")
    graph.add_edge("analysis", "reporting")
    graph.add_edge("reporting", END)
    graph.add_edge("retry_or_halt", END)

    return graph.compile()