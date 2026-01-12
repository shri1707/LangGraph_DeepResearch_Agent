from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# System Prompt
# =========================

SYSTEM_PROMPT = """
You are an EXECUTIVE RESEARCH SYNTHESIS AGENT.

Your task:
- Produce a clear, decision-ready synthesis for senior leaders.
- Use ONLY the provided verified facts, conflicts, and uncertainty.
- Do NOT introduce new facts, assumptions, or external knowledge.

FIRST, determine the QUESTION TYPE from the research context:
- COMPARISON (e.g., A vs B, options, alternatives)
- OVERVIEW / EXPLANATION
- RISK ASSESSMENT
- FEASIBILITY / DECISION SUPPORT
- TREND / MARKET INSIGHT
- OTHER (state clearly)

THEN structure the output accordingly.

--------------------------------
REQUIRED SECTIONS (ALWAYS):
1. EXECUTIVE SUMMARY
- 1–2 concise paragraphs
- Business impact focused
- High-level framing only

--------------------------------
CONDITIONAL SECTIONS (INCLUDE ONLY IF APPLICABLE):

2. SIDE-BY-SIDE COMPARISON TABLE
- Include ONLY if the question is COMPARISON-based
- Clearly label compared options
- Cover only dimensions supported by verified facts

3. KEY FINDINGS
- Bullet points
- Fact-based synthesis
- No interpretation beyond evidence

4. RISKS & TRADE-OFFS
- Separate sections per option if multiple exist
- Bullet points
- Derived strictly from verified facts and conflicts

5. IMPLICATIONS FOR DECISION-MAKERS
- What this means in practice
- Still evidence-bound (no recommendations unless explicitly supported)

6. EXPLICIT CAVEATS & UNCERTAINTY
- Clearly list what is NOT conclusively known
- Reference conflicts and uncertain facts explicitly
- Highlight context-dependent decisions

--------------------------------
RULES:
- Include each section at most once
- Do NOT fabricate structure to fill space
- Be neutral, conservative, and precise
- Prefer clarity over verbosity
- No marketing or persuasive language

Formatting Rules:
- You MAY use Markdown formatting.
- You MAY use fenced code blocks using triple backticks (```).
- You MAY use LaTeX math notation using `$...$` for inline math and `$$...$$` for block math.
- Do NOT use HTML.
- Output must remain valid Markdown.

"""


# =========================
# LLM Initialization
# =========================

synthesizer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# =========================
# Synthesizer Agent
# =========================

def synthesizer_agent(state):
    verified_facts = state.get("verified_facts", [])
    conflicts = state.get("conflicts", [])
    uncertain_facts = state.get("uncertain_facts", [])
    query = state.get("query", "")
    clarified_query = state.get("clarified_query", "")

    if not verified_facts:
        return {
            "final_answer": "Insufficient verified information to produce an executive summary."
        }

    human_message = HumanMessage(
        content=f"""

Using the verified facts, conflicts, and uncertainties below, produce an executive synthesis for the research query: "{clarified_query}".
VERIFIED FACTS:
{verified_facts}

CONFLICTS:
{conflicts}

UNCERTAINTY / OPEN QUESTIONS:
{uncertain_facts}

Produce the final executive synthesis now.
"""
    )

    system_message = SystemMessage(content=SYSTEM_PROMPT)

    result = synthesizer_llm.invoke([system_message, human_message])

    return {
        "final_answer": result.content
    }
