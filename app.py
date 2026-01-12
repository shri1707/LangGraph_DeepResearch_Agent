# import streamlit as st
# from graph.research_graph import build_graph
# from schemas.state import ResearchState

# # ----------------------------
# # Page setup
# # ----------------------------
# st.set_page_config(
#     page_title="Deep Research Agent",
#     layout="wide"
# )

# st.title("🧠 Deep Research Agent")
# st.caption("Streaming, step-wise research with verified evidence")

# # ----------------------------
# # Build graph once
# # ----------------------------
# @st.cache_resource
# def load_graph():
#     return build_graph()

# graph = load_graph()

# # ----------------------------
# # Query input
# # ----------------------------
# query = st.text_input(
#     "Enter a research question",
#     placeholder="Compare self-hosted LLMs vs cloud LLMs for enterprises"
# )
# # What is Retrieval-Augmented Generation (RAG) and why is it used in enterprise AI?

# run_btn = st.button("Run Research")

# # ----------------------------
# # Helper renderers
# # ----------------------------

# def render_verified_fact(fact, idx):
#     st.markdown(f"### Verified Fact {idx + 1}")
#     st.markdown(f"**Statement:** {fact.fact}")
#     st.markdown(f"**Confidence:** `{fact.confidence}`")

#     st.markdown("**Evidence:**")
#     for e in fact.evidence:
#         title = e.title or "Unknown source"
#         st.markdown(f"- [{title}]({e.url})")

#     st.divider()



# def render_conflict(conflict, idx):
#     st.markdown(f"### Conflict {idx + 1}")
#     st.markdown(f"**Claim:** {conflict.claim}")
#     st.markdown(f"**Reason:** {conflict.reason}")

#     st.markdown("**Conflicting Sources:**")
#     for e in conflict.conflicting_sources:
#         title = e.title or "Unknown source"
#         st.markdown(f"- [{title}]({e.url})")

#     st.divider()



# def render_uncertainty(item, idx):
#     st.markdown(f"### Uncertain Item {idx + 1}")
#     st.write(item)
#     st.divider()


# # ----------------------------
# # Run graph (STREAMING)
# # ----------------------------
# if run_btn:
#     st.divider()

#     state = ResearchState(query=query)

#     progress = st.progress(0)
#     status = st.empty()

#     step_map = {
#         "planner": 0.15,
#         "search": 0.30,
#         "reader": 0.50,
#         "verifier": 0.75,
#         "synthesizer": 1.0,
#     }

#     final_state = {}

#     status.info("Starting research...")

#     for event in graph.stream(state):
#         for node_name, node_output in event.items():
#             # 🔍 DEBUG: show streaming updates
#             print(f"\n[STREAM] Node: {node_name}")
#             print(f"[STREAM] Keys: {list(node_output.keys())}")

#             final_state.update(node_output)

#             pct = step_map.get(node_name, progress._value)
#             progress.progress(pct)

#             status.info(f"Running **{node_name}** step...")

#     progress.progress(1.0)
#     status.success("Research complete")

#     # 🔍 FINAL STATE DEBUG
#     print("\n[FINAL STATE KEYS]")
#     print(final_state.keys())

#     print("\n[FINAL STATE COUNTS]")
#     print("verified_facts:", len(final_state.get("verified_facts", [])))
#     print("conflicts:", len(final_state.get("conflicts", [])))
#     print("uncertain_facts:", len(final_state.get("uncertain_facts", [])))

#     st.divider()

#     # ----------------------------
#     # RESULTS
#     # ----------------------------

#     # VERIFIED FACTS
#     st.subheader("✅ Verified Facts")

#     if final_state.get("verified_facts"):
#         with st.expander("View Verified Facts", expanded=False):
            # for i, fact in enumerate(final_state["verified_facts"]):
            #     render_verified_fact(fact, i)
#     else:
#         st.info("No verified facts found.")


#     # CONFLICTS
#     st.subheader("⚠️ Conflicts")

#     if final_state.get("conflicts"):
#         with st.expander("View Conflicts", expanded=False):
#             for i, conflict in enumerate(final_state["conflicts"]):
#                 render_conflict(conflict, i)
#     else:
#         st.success("No conflicts detected.")


#     # UNCERTAINTY
#     st.subheader("❓ Uncertainty")

#     if final_state.get("uncertain_facts"):
#         with st.expander("View Uncertain Items", expanded=False):
#             for i, u in enumerate(final_state["uncertain_facts"]):
#                 render_uncertainty(u, i)
#     else:
#         st.success("No uncertain facts.")


#     # FINAL ANSWER
#     st.subheader("📌 Executive Summary")
#     st.markdown(final_state.get("final_answer", "No final answer generated."))
import streamlit as st
from langgraph.types import Command
from graph.research_graph import build_graph

# ======================================================
# Page setup
# ======================================================
st.set_page_config(page_title="Deep Research Agent", layout="wide")

st.title("🧠 Deep Research Agent")
st.caption("Human-in-the-loop, step-wise deep research")

# ======================================================
# Load LangGraph
# ======================================================
@st.cache_resource
def load_graph():
    print("[DEBUG] Loading LangGraph...")
    return build_graph()

graph = load_graph()

# ======================================================
# Helper renderers
# ======================================================
def render_verified_fact(fact, idx):
    st.markdown(f"### Verified Fact {idx + 1}")
    st.markdown(f"**Statement:** {fact.fact}")
    st.markdown(f"**Confidence:** `{fact.confidence}`")

    st.markdown("**Evidence:**")
    for e in fact.evidence:
        title = e.title or "Unknown source"
        st.markdown(f"- [{title}]({e.url})")

    st.divider()


def render_conflict(conflict, idx):
    st.markdown(f"### Conflict {idx + 1}")
    st.markdown(f"**Claim:** {conflict.claim}")
    st.markdown(f"**Reason:** {conflict.reason}")

    st.markdown("**Conflicting Sources:**")
    for e in conflict.conflicting_sources:
        title = e.title or "Unknown source"
        st.markdown(f"- [{title}]({e.url})")

    st.divider()


def render_uncertainty(item, idx):
    st.markdown(f"### Uncertain Item {idx + 1}")
    st.write(item)
    st.divider()

# ======================================================
# Session state
# ======================================================
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None

if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None

if "running" not in st.session_state:
    st.session_state.running = False

if "base_query" not in st.session_state:
    st.session_state.base_query = None

# ======================================================
# Progress UI
# ======================================================
progress_bar = st.progress(0.0)
step_text = st.empty()

def set_progress(pct: float, text: str):
    progress_bar.progress(pct)
    step_text.info(text)

# ======================================================
# Query input
# ======================================================
query = st.text_input("Enter a research question", placeholder="Best phone of 2024")

if st.button("Run Research") and query:
    print("[DEBUG] Run Research clicked")
    print("[DEBUG] Initial query:", query)

    st.session_state.base_query = query
    st.session_state.graph_state = {
        "query": query,
        "clarification_round": 0,
    }
    st.session_state.interrupt_payload = None
    st.session_state.running = True

    set_progress(0.05, "🚀 Initializing research...")

# ======================================================
# Graph execution (ONLY when running & NOT interrupted)
# ======================================================
if (
    st.session_state.running
    and st.session_state.graph_state
    and not st.session_state.interrupt_payload
):
    try:
        print("\n[DEBUG] Invoking graph with state:")
        print(st.session_state.graph_state)

        config = {"configurable": {"thread_id": "deep-research"}}

        # ---- Planner phase ----
        set_progress(0.15, "🧠 Planner: analyzing query")

        result = graph.invoke(st.session_state.graph_state, config=config)

        print("\n[DEBUG] Graph returned result:")
        print(result)

        # ================= HITL INTERRUPT =================
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print("[DEBUG] Interrupt payload:", payload)

            set_progress(0.15, "⏸ Awaiting human clarification")

            st.session_state.interrupt_payload = payload
            st.session_state.running = False

        # ================= FINAL RESULT =================
        else:
            # Deterministic progress (invoke is atomic)
            set_progress(0.35, "🔍 Search: gathering sources")
            set_progress(0.55, "📖 Reader: extracting facts")
            set_progress(0.75, "✅ Verifier: validating claims")
            set_progress(1.0, "🧾 Synthesizer: generating final answer")
            step_text.success("✅ Research complete")

            st.divider()

            st.subheader("📌 Executive Summary")
            st.markdown(result.get("final_answer", "No final answer generated."))

            st.subheader("✅ Verified Facts")
            if result.get("verified_facts"):
                with st.expander("View Verified Facts"):
                    for i, fact in enumerate(result["verified_facts"]):
                        render_verified_fact(fact, i)
            else:
                st.info("No verified facts found.")

            st.subheader("⚠️ Conflicts")
            if result.get("conflicts"):
                with st.expander("View Conflicts"):
                    for i, conflict in enumerate(result["conflicts"]):
                        render_conflict(conflict, i)
            else:
                st.success("No conflicts detected.")

            st.subheader("❓ Uncertainty")
            if result.get("uncertain_facts"):
                with st.expander("View Uncertainty"):
                    for i, item in enumerate(result["uncertain_facts"]):
                        render_uncertainty(item, i)
            else:
                st.success("No uncertain facts.")

            # Reset
            st.session_state.graph_state = None
            st.session_state.running = False
            st.session_state.base_query = None

    except Exception as e:
        print("[ERROR]", e)
        st.error("❌ An unexpected error occurred.")
        st.exception(e)

# ======================================================
# HITL CLARIFICATION UI (NO GRAPH INVOKE HERE)
# ======================================================
if st.session_state.interrupt_payload:
    payload = st.session_state.interrupt_payload

    st.warning("⚠️ Clarification required to continue research")
    st.markdown(f"**Reason:** {payload['reason']}")

    answers = []
    for i, q in enumerate(payload["questions"]):
        ans = st.text_input(
            label=q,
            key=f"clarify_{payload['round']}_{i}"
        )
        answers.append(ans)

    if st.button("Continue Research"):
        print("[DEBUG] Continue Research clicked")
        print("[DEBUG] User answers:", answers)

        clarified_query = (
            st.session_state.base_query
            + " | "
            + " ".join(a for a in answers if a.strip())
        )

        print("[DEBUG] Clarified query:", clarified_query)

        st.session_state.graph_state = Command(
            resume={
                "clarified_query": clarified_query,
                "clarification_round": payload["round"],
            }
        )

        st.session_state.interrupt_payload = None
        st.session_state.running = True

        set_progress(0.15, "🧠 Planner: re-evaluating clarified query")
        st.rerun()
