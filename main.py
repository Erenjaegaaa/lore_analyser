"""
pipeline/query_pipeline.py — Full query pipeline for the GraphRAG Lore Assistant.

Flow:
    question
      → context_assembler.assemble_from_question()   (graph + vector retrieval)
      → build_answer_prompt()                         (format LLM prompt)
      → Gemini answer generation                      (gemini-2.5-flash)
      → return answer + debug metadata

Public functions:
    query(question)  -> dict   # answer + metadata
"""

import time

import google.generativeai as genai

from retrieval.context_assembler import assemble_from_question
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# ── Gemini answer model (lazy singleton) ──────────────────────────────────────

_answer_model = None


def _get_answer_model() -> genai.GenerativeModel:
    global _answer_model
    if _answer_model is None:
        genai.configure(api_key=settings.llm.gemini_api_key)
        _answer_model = genai.GenerativeModel(
            model_name=settings.llm.answer_model,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=settings.llm.max_output_tokens,
            ),
        )
        log.debug("Gemini answer model initialised: %s", settings.llm.answer_model)
    return _answer_model


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_answer_prompt(question: str, context: str) -> str:
    """
    Build the final LLM prompt from the question and assembled context.

    Instructs Gemini to:
    - Answer only from the provided context
    - Cite sources by page title where possible
    - Admit uncertainty rather than hallucinate
    - Keep answers focused and lore-accurate

    Args:
        question: The user's natural language question.
        context:  Assembled context string from context_assembler.

    Returns:
        Complete prompt string ready to send to Gemini.
    """
    if not context:
        # No context retrieved — tell Gemini explicitly
        context_block = "No relevant context was retrieved from the knowledge base."
    else:
        context_block = context

    return f"""You are a knowledgeable assistant specialising in Tolkien's Middle-earth lore.
Answer the question below using ONLY the provided context.

Rules:
- Base your answer strictly on the context provided. Do not use outside knowledge.
- If the context contains graph facts (GRAPH FACTS section), use them to establish relationships.
- If the context contains text passages (TEXT CONTEXT section), use them for detail and nuance.
- Cite your sources by referencing the page title in brackets, e.g. [Aragorn - Wikipedia].
- If the context does not contain enough information to answer the question, say:
  "I don't have enough information in my knowledge base to answer this confidently."
- Do not make up names, events, or relationships not present in the context.
- Keep your answer concise but complete. Aim for 2-5 sentences for simple questions,
  more for complex ones.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT:
{context_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}

ANSWER:"""


# ── Main query function ───────────────────────────────────────────────────────

def query(
    question: str,
    vector_top_k: int | None = None,
) -> dict:
    """
    Run the full GraphRAG query pipeline for a user question.

    Args:
        question:     The user's natural language question.
        vector_top_k: Override number of vector chunks (default from config).

    Returns:
        Dict with keys:
          - question:        str   — original question
          - answer:          str   — Gemini's answer
          - graph_sentences: list  — graph triples used (for debug)
          - chunk_ids:       list  — vector chunk IDs used (for debug)
          - context:         str   — full assembled context (for debug)
          - latency_ms:      float — total pipeline latency in milliseconds
          - error:           str | None — error message if pipeline failed
    """
    start_time = time.time()

    result = {
        "question":        question,
        "answer":          "",
        "graph_sentences": [],
        "chunk_ids":       [],
        "context":         "",
        "latency_ms":      0.0,
        "error":           None,
    }

    if not question or not question.strip():
        result["error"] = "Empty question."
        return result

    # ── Step 1: Retrieve context ──────────────────────────────────────────
    try:
        context, graph_sentences, chunk_ids = assemble_from_question(
            question, vector_top_k=vector_top_k
        )
        result["graph_sentences"] = graph_sentences
        result["chunk_ids"]       = chunk_ids
        result["context"]         = context
    except Exception as exc:
        log.error("Context assembly failed: %s", exc)
        result["error"] = f"Retrieval failed: {exc}"
        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    # ── Step 2: Build prompt ──────────────────────────────────────────────
    prompt = build_answer_prompt(question, context)
    log.debug("Answer prompt built (%d chars).", len(prompt))

    # ── Step 3: Generate answer with Gemini ───────────────────────────────
    try:
        model = _get_answer_model()
        response = model.generate_content(prompt)
        answer = response.text.strip()
        result["answer"] = answer
        log.info(
            "Query complete | graph=%d sentences | chunks=%d | answer=%d chars",
            len(graph_sentences), len(chunk_ids), len(answer),
        )
    except Exception as exc:
        log.error("Gemini answer generation failed: %s", exc)
        result["error"] = f"Answer generation failed: {exc}"

    result["latency_ms"] = (time.time() - start_time) * 1000
    return result


# ── Debug printer ─────────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """
    Pretty-print a query result dict to stdout.
    Useful for interactive testing from main.py or a REPL.
    """
    print("\n" + "=" * 60)
    print(f"QUESTION: {result['question']}")
    print("=" * 60)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print(f"\nANSWER:\n{result['answer']}")

    print(f"\n── Debug ──────────────────────────────────────────────────")
    print(f"Graph sentences : {len(result.get('graph_sentences', []))}")
    print(f"Vector chunks   : {len(result.get('chunk_ids', []))}")
    print(f"Chunk IDs       : {result.get('chunk_ids', [])}")
    print(f"Latency         : {result.get('latency_ms', 0):.0f}ms")
    print("=" * 60 + "\n")