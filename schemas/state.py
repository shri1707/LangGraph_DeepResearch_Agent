from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    query: str

    # planner
    plan: Dict[str, Any] | None = None

    # Clarification / HITL
    clarified_query: Optional[str] = None
    clarification_questions: List[str] = []
    clarification_answers: List[str] = []
    clarification_round: int = 0
    clarification_complete: bool = False

    # Search & reading
    search_queries: List[str] = []
    sources: List[Dict[str, Any]] = []
    # sources: Annotated[list[Dict[str,Any]], add_messages]= [] , we will use this later for message conversion

    notes: List[Dict[str, Any]] = []

    # Verification
    verified_facts: List[str] = []
    conflicts: List[str] = []
    uncertain_facts: List[str] = []

    # Final output
    final_answer: str | None = None
