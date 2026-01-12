from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# Output Schemas
# =========================

class Evidence(BaseModel):
    url: str
    title: Optional[str] = None


class VerifiedFact(BaseModel):
    fact: str = Field(description="A short, atomic factual statement")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score based on strength and agreement of evidence"
    )
    evidence: List[Evidence]


class Conflict(BaseModel):
    claim: str = Field(description="The disputed or unclear claim")
    conflicting_sources: List[Evidence]
    reason: str = Field(description="Why this claim cannot be verified confidently")


class VerifierOutput(BaseModel):
    verified_facts: List[VerifiedFact] = []
    conflicts: List[Conflict] = []
    uncertain_facts: List[str] = []


# =========================
# System Prompt
# =========================

SYSTEM_PROMPT = """
You are a FACT VERIFICATION AGENT.

You will receive factual statements extracted from multiple sources.
Each statement includes:
- fact
- source_type (official, independent_blog, vendor_blog, forum)
- url
- title

Your task:
- Verify facts conservatively.
- Compare claims across sources.
- Identify agreement, contradiction, or uncertainty.

Rules:
1. DO NOT introduce new facts.
2. A fact is VERIFIED only if supported by:
   - at least TWO independent sources, OR
   - ONE official source with no contradiction.
3. Assign confidence scores:
   - 0.9–1.0 → multiple strong sources agree
   - 0.7–0.8 → multiple sources agree with minor ambiguity
   - 0.4–0.6 → single source or weak support
4. If sources contradict → classify as CONFLICT.
5. If evidence is insufficient → classify as UNCERTAIN.
6. Be conservative. Accuracy > completeness.

Output must strictly follow the provided schema.

IMPORTANT:
- NEVER output incomplete or truncated facts.
- EVERY verified_fact MUST include:
  - fact
  - confidence
  - evidence (at least one source)
- If a fact cannot meet this requirement, move it to uncertain_facts.
- ALWAYS include conflicts and uncertain_facts fields, even if empty.

"""


# =========================
# LLM Initialization
# =========================

verifier_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0
).with_structured_output(VerifierOutput)


# =========================
# Verifier Agent
# =========================

def verifier_agent(state):
    notes = state.get("notes", [])

    if not notes:
        return {
            "verified_facts": [],
            "conflicts": [],
            "uncertain_facts": ["No extracted facts were available for verification."]
        }

    # Flatten facts with metadata
    evidence_items = []

    for note in notes:
        for fact in note["facts"]:
            evidence_items.append({
                "fact": fact,
                "source_type": note["source_type"],
                "url": note["url"],
                "title": note.get("title", "Unknown source")
            })

    # Build LLM input
    human_message = HumanMessage(
        content="\n".join(
            f"- FACT: {e['fact']}\n"
            f"  SOURCE_TYPE: {e['source_type']}\n"
            f"  TITLE: {e['title']}\n"
            f"  URL: {e['url']}\n"
            for e in evidence_items
        )
    )

    system_message = SystemMessage(content=SYSTEM_PROMPT)

    # Invoke verifier
    result: VerifierOutput = verifier_llm.invoke(
        [system_message, human_message]
    )

    return {
        "verified_facts": result.verified_facts,
        "conflicts": result.conflicts,
        "uncertain_facts": result.uncertain_facts
    }
