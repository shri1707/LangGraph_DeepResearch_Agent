# from langgraph.graph import StateGraph
# from langgraph.checkpoint.memory import MemorySaver

# from schemas.state import ResearchState
# from agents.planner import planner_agent
# from agents.searcher import search_agent
# from agents.reader import reader_agent
# from agents.verifier import verifier_agent
# from agents.synthesizer import synthesizer_agent

# def route_after_planner(state):
#     if not state.get("clarification_complete", False):
#         return "planner"  # stay here (interrupt-driven)
#     return "search"


# def build_graph():
#     builder = StateGraph(ResearchState)

#     builder.add_node("planner", planner_agent)
#     builder.add_node("search", search_agent)
#     builder.add_node("reader", reader_agent)
#     builder.add_node("verifier", verifier_agent)
#     builder.add_node("synthesizer", synthesizer_agent)

#     builder.set_entry_point("planner")

#     builder.add_edge("planner", "search")
#     builder.add_edge("search", "reader")
#     builder.add_edge("reader", "verifier")
#     builder.add_edge("verifier", "synthesizer")

#     # ✅ REQUIRED for interrupt()
#     checkpointer = MemorySaver()

#     return builder.compile(checkpointer=checkpointer)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from schemas.state import ResearchState

# ----------------------------
# Import agents
# ----------------------------
from agents.planner import planner_agent
from agents.searcher import search_agent
from agents.reader import reader_agent
from agents.verifier import verifier_agent
from agents.synthesizer import synthesizer_agent


# ----------------------------
# Routing logic (CRITICAL)
# ----------------------------
def route_after_planner(state: ResearchState) -> str:
    """
    Decide whether to continue research or stay in clarification loop.
    """
    # If clarification is NOT complete, stop graph progression
    if not state.get("clarification_complete", False):
        # Stay at planner — execution will be paused via interrupt
        return "planner"

    # Clarification done → proceed normally
    return "search"


# ----------------------------
# Build graph
# ----------------------------
def build_graph():
    builder = StateGraph(ResearchState)

    # ----------------------------
    # Nodes
    # ----------------------------
    builder.add_node("planner", planner_agent)
    builder.add_node("search", search_agent)
    builder.add_node("reader", reader_agent)
    builder.add_node("verifier", verifier_agent)
    builder.add_node("synthesizer", synthesizer_agent)

    # ----------------------------
    # Entry point
    # ----------------------------
    builder.set_entry_point("planner")

    # ----------------------------
    # Conditional routing AFTER planner
    # ----------------------------
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "planner": "planner",  # clarification loop (interrupt-driven)
            "search": "search",    # proceed when ready
        },
    )

    # ----------------------------
    # Normal research flow
    # ----------------------------
    builder.add_edge("search", "reader")
    builder.add_edge("reader", "verifier")
    builder.add_edge("verifier", "synthesizer")
    builder.add_edge("synthesizer", END)

    # ----------------------------
    # Checkpointer (REQUIRED for interrupt)
    # ----------------------------
    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
