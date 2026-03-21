"""
streamlit_app.py — Streamlit chat UI for the GraphRAG Lore Assistant.

Can run in two modes:
  1. Direct mode (no backend needed) — imports pipeline directly
  2. API mode — calls the FastAPI backend via HTTP

Run locally (direct mode):
    streamlit run streamlit_app.py

Run against deployed backend:
    BACKEND_URL=https://your-app.railway.app streamlit run streamlit_app.py
"""

import os
import time
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "").strip()
USE_DIRECT  = not BACKEND_URL  # if no backend URL, import pipeline directly

if USE_DIRECT:
    from pipeline.query_pipeline import query as direct_query

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Middle-earth Lore Assistant",
    page_icon="🧙",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1e1e2e;
        border-left: 4px solid #7c3aed;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .debug-box {
        background-color: #12121f;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🧙 Middle-earth Lore Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about Tolkien\'s world — powered by Hybrid GraphRAG</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.show_debug = st.toggle("Show debug info", value=False)
    st.markdown("---")
    st.markdown("### 💡 Example questions")
    example_questions = [
        "Who forged the One Ring?",
        "What is the relationship between Aragorn and Isildur?",
        "Who is Gandalf?",
        "What happened at the Battle of Helm's Deep?",
        "What is Mordor?",
        "How are Frodo and Bilbo related?",
        "What are the Silmarils?",
        "Who is the dark lord of Mordor?",
    ]
    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state.pending_question = eq

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **GraphRAG** combines:
    - 🔵 **Neo4j** knowledge graph
    - 🟣 **ChromaDB** vector search  
    - ✨ **Gemini** answer generation
    
    Built from **71 Wikipedia pages** covering Tolkien's legendarium.
    """)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Query function ────────────────────────────────────────────────────────────

def run_query(question: str) -> dict:
    """Run query either directly or via backend API."""
    if USE_DIRECT:
        return direct_query(question)
    else:
        try:
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": question},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            # Normalize to match direct query result shape
            data["graph_sentences"] = data.get("graph_sentences", [])
            data["chunk_ids"]       = data.get("chunk_ids", [])
            return data
        except requests.exceptions.Timeout:
            return {"question": question, "answer": "", "error": "Request timed out. Please try again.", "graph_sentences": [], "chunk_ids": [], "latency_ms": 0}
        except Exception as exc:
            return {"question": question, "answer": "", "error": str(exc), "graph_sentences": [], "chunk_ids": [], "latency_ms": 0}


# ── Chat history display ──────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and st.session_state.show_debug and message.get("debug"):
            debug = message["debug"]
            st.markdown(f"""
<div class="debug-box">
📊 Graph: {debug.get('graph_count', 0)} sentences &nbsp;|&nbsp;
📄 Vector: {debug.get('vector_count', 0)} chunks &nbsp;|&nbsp;
⏱️ {debug.get('latency_ms', 0):.0f}ms
</div>
""", unsafe_allow_html=True)


# ── Handle example question from sidebar ──────────────────────────────────────

pending = st.session_state.pop("pending_question", None)

# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input("Ask about Middle-earth lore...") or pending

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🧙 Searching the archives of Middle-earth..."):
            result = run_query(question)

        if result.get("error"):
            answer = f"❌ Sorry, I encountered an error: {result['error']}"
        elif not result.get("answer"):
            answer = "I don't have enough information in my knowledge base to answer this confidently."
        else:
            answer = result["answer"]

        st.markdown(answer)

        debug_info = {
            "graph_count":  len(result.get("graph_sentences", [])),
            "vector_count": len(result.get("chunk_ids", [])),
            "latency_ms":   result.get("latency_ms", 0),
            "chunk_ids":    result.get("chunk_ids", []),
        }

        if st.session_state.show_debug:
            st.markdown(f"""
<div class="debug-box">
📊 Graph: {debug_info['graph_count']} sentences &nbsp;|&nbsp;
📄 Vector: {debug_info['vector_count']} chunks &nbsp;|&nbsp;
⏱️ {debug_info['latency_ms']:.0f}ms
</div>
""", unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "debug":   debug_info,
    })