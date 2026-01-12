# from typing import List
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import Field, BaseModel
# from schemas.state import ResearchState
# from dotenv import load_dotenv
# from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
# load_dotenv()

# class PlannerOutput(BaseModel):
#     objectives: List[str] = Field(
#         description="High-level research objectives"
#     )
#     search_queries: List[str] = Field(
#         description="Concrete web search queries derived from the objectives"
#     )
# planner = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(PlannerOutput)


# def planner_agent(state: ResearchState):
#     system_message = SystemMessage(
#         content="""You are a research planner. 
#         Break the user query into clear research objectives and concrete web search queries. 
#         Do NOT answer the question."""
#     )

#     human_message = HumanMessage(
#         content=f"Create a research plan for the following query: {state['query']}"
#     )

#     response = planner.invoke([system_message, human_message])
#     return {"plan": response.model_dump(), # model_dump() is a Pydantic method that converts a Pydantic model instance into a plain Python dictionary.
#             "search_queries": response.search_queries}


from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt
from typing import List, Literal
from pydantic import BaseModel, Field

from schemas.state import ResearchState

load_dotenv()

# ---------- Models ----------

class AmbiguityCheckOutput(BaseModel):
    status: Literal["CLEAR", "AMBIGUOUS"]
    reason: str

class ClarificationOutput(BaseModel):
    questions: List[str]

class PlannerOutput(BaseModel):
    objectives: List[str]
    search_queries: List[str]

# ---------- LLMs ----------

ambiguity_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
).with_structured_output(AmbiguityCheckOutput)

clarification_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
).with_structured_output(ClarificationOutput)

planner_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
).with_structured_output(PlannerOutput)

# ---------- Prompts ----------

AMBIGUITY_PROMPT = """
You are a research planner.

Your task:
1. Determine whether the user's query is sufficiently specific for factual research.
2. If ambiguous, ask clarification questions.
3. If clear, produce a structured research plan.

Rules:
- DO NOT assume missing criteria.
- Subjective terms like "best", "top", "most", "worst", etc.(these words are just used for example there are many more words like these) REQUIRE clarification.
- Be conservative.
"""
CLARIFICATION_PROMPT = """
Generate clarification questions that will reduce ambiguity.

Rules:
- Be more specific than previous rounds.
- Ask questions with respect to the query asked by the user and the ambiguity reason.
- Max 3 questions.
"""

PLANNER_PROMPT = """
        You are a research planner.
        Break the user query into clear research objectives and concrete web search queries.
        Do NOT answer the question.
"""

# ---------- Agent ----------
MAX_CLARIFICATION_ROUNDS = 3

# ---------- Planner Node ----------

def planner_agent(state: ResearchState):
    """
    Planner owns the clarification loop.
    It is the ONLY place where ambiguity is decided.
    """

    # Resume input (if any)
    resume = state.get("__resume__")

    if resume:
        state["clarified_query"] = resume["clarified_query"]
        state["clarification_round"] = resume["clarification_round"]

    query = state.get("clarified_query") or state["query"]
    round_num = state.get("clarification_round", 0)

    # Stop clarification if max rounds reached
    if round_num >= MAX_CLARIFICATION_ROUNDS:
        return _produce_plan(query)

    # Ambiguity check
    ambiguity = ambiguity_llm.invoke([
        SystemMessage(content=AMBIGUITY_PROMPT),
        HumanMessage(content=query),
    ])

    if ambiguity.status == "AMBIGUOUS":
        questions = clarification_llm.invoke([
            SystemMessage(content=CLARIFICATION_PROMPT),
            HumanMessage(content=query),
        ])

        return interrupt({
            "type": "clarification",
            "reason": ambiguity.reason,
            "questions": questions.questions,
            "round": round_num + 1,
        })

    # Clear → produce plan
    return _produce_plan(query)

def _produce_plan(query: str):
    plan = planner_llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=query),
    ])

    return {
        "plan": plan.model_dump(),
        "search_queries": plan.search_queries,
        "clarification_complete": True,
    }
