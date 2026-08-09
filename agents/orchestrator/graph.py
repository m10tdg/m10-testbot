from langgraph.graph import StateGraph, END
from state import RunState
from nodes.scenario import scenario_agent_node
from nodes.execution import execution_agent_node
from nodes.visual_analysis import visual_analysis_agent_node
from nodes.analysis import analysis_agent_node
from nodes.reporting import reporting_agent_node


def route_after_execution(state: RunState) -> str:
    """This is where Human-in-the-Loop / retry logic will eventually live.
    For now: if execution failed outright, skip straight to reporting so the
    developer still gets a report explaining the failure."""
    if state["status"] == "failed":
        return "reporting"
    return "visual_analysis"


def build_graph():
    graph = StateGraph(RunState)

    graph.add_node("scenario", scenario_agent_node)
    graph.add_node("execution", execution_agent_node)
    graph.add_node("visual_analysis", visual_analysis_agent_node)
    graph.add_node("analysis", analysis_agent_node)
    graph.add_node("reporting", reporting_agent_node)

    graph.set_entry_point("scenario")
    graph.add_edge("scenario", "execution")
    graph.add_conditional_edges("execution", route_after_execution, {
        "visual_analysis": "visual_analysis",
        "reporting": "reporting",
    })
    graph.add_edge("visual_analysis", "analysis")
    graph.add_edge("analysis", "reporting")
    graph.add_edge("reporting", END)

    return graph.compile()
