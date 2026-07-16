"""
Assembles the five agents into a LangGraph directed state graph.

Pipeline flow:
  START → router → query_expander → retrieval → synthesizer → reviewer → END

All nodes share AgentState. Each node receives the full state and returns
only the fields it updates — LangGraph merges them automatically.

Error handling:
  If the router sets state["error"], the pipeline short-circuits to END
  via a conditional edge. This prevents downstream agents from running
  on bad input and wasting LLM calls.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.router import route
from app.agents.query_expander import expand_query
from app.agents.retrieval import retrieve
from app.agents.synthesizer import synthesize
from app.agents.reviewer import review


def _should_continue(state: AgentState) -> str:
    """
    Conditional edge after the router.
    If routing failed (error set), skip to END immediately.
    Otherwise continue to query expansion.
    """
    if state.get("error"):
        return "end"
    return "query_expander"


def build_graph() -> StateGraph:
    """
    Builds and compiles the full agent pipeline graph.
    Returns a compiled LangGraph StateGraph ready to invoke.
    """
    graph = StateGraph(AgentState)

    #  Add nodes 
    graph.add_node("router", route)
    graph.add_node("query_expander", expand_query)
    graph.add_node("retrieval", retrieve)
    graph.add_node("synthesizer", synthesize)
    graph.add_node("reviewer", review)

    #  Add edges 
    graph.add_edge(START, "router")

    # Conditional: if router failed, go straight to END
    graph.add_conditional_edges(
        "router",
        _should_continue,
        {
            "query_expander": "query_expander",
            "end": END,
        },
    )

    graph.add_edge("query_expander", "retrieval")
    graph.add_edge("retrieval", "synthesizer")
    graph.add_edge("synthesizer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


#  Singleton — compiled once at import time 
# The compiled graph is stateless and safe to reuse across requests.
agent_graph = build_graph()