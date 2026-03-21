"""
retrieval/context_assembler.py — Assembles the final context block for the LLM.

Takes graph triple sentences (from graph_retriever) and vector chunks
(from vector_retriever) and merges them into a single formatted string
that is passed to the LLM for answer generation.

Output format:
    GRAPH FACTS
    - Aragorn is the heir of Isildur.
    - Aragorn rules over Gondor.

    TEXT CONTEXT
    [Source: Aragorn - Wikipedia]
    chunk text...

    [Source: One Ring - Wikipedia]
    chunk text...

Public functions:
    assemble(graph_sentences, vector_chunks) -> str
    assemble_from_question(question)         -> tuple[str, list[str], list[str]]
"""

from retrieval.graph_retriever import retrieve as graph_retrieve
from retrieval.vector_retriever import retrieve as vector_retrieve
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ── Core assembly ─────────────────────────────────────────────────────────────

def assemble(
    graph_sentences: list[str],
    vector_chunks: list[dict],
) -> str:
    """
    Merge graph sentences and vector chunks into a single context string.

    Args:
        graph_sentences: Serialized triple sentences from graph_retriever.
        vector_chunks:   Chunk dicts from vector_retriever (with text + metadata).

    Returns:
        Formatted context string ready to be injected into the LLM prompt.
        Returns empty string if both inputs are empty.
    """
    parts: list[str] = []

    # ── Graph facts section ───────────────────────────────────────────────
    if graph_sentences:
        lines = ["GRAPH FACTS"]
        for sentence in graph_sentences:
            lines.append(f"- {sentence}")
        parts.append("\n".join(lines))
        log.debug("Context: %d graph sentences added.", len(graph_sentences))
    else:
        log.debug("Context: no graph sentences — graph section omitted.")

    # ── Text context section ──────────────────────────────────────────────
    if vector_chunks:
        text_parts: list[str] = ["TEXT CONTEXT"]
        for chunk in vector_chunks:
            text = (chunk.get("text") or "").strip()
            metadata = chunk.get("metadata") or {}
            page_title = metadata.get("page_title") or chunk.get("chunk_id", "Unknown")
            source_url = metadata.get("source_url", "")

            if not text:
                continue

            # Format: source header + chunk text
            source_line = f"[Source: {page_title}]"
            if source_url:
                source_line += f"  ({source_url})"
            text_parts.append(f"\n{source_line}\n{text}")

        if len(text_parts) > 1:  # more than just the header
            parts.append("\n".join(text_parts))
            log.debug("Context: %d vector chunks added.", len(vector_chunks))
    else:
        log.debug("Context: no vector chunks — text section omitted.")

    if not parts:
        log.warning("assemble() produced empty context — both inputs were empty.")
        return ""

    context = "\n\n".join(parts)
    log.debug("Context assembled: %d chars total.", len(context))
    return context


# ── Convenience entry point ───────────────────────────────────────────────────

def assemble_from_question(
    question: str,
    vector_top_k: int | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Run both retrievers for a question and assemble the full context.

    This is the main entry point called by query_pipeline.py.
    Always runs both retrievers — hybrid retrieval is unconditional.

    Args:
        question:     The user's natural language question.
        vector_top_k: Override for number of vector chunks (default from config).

    Returns:
        Tuple of:
          - context_str:      Full assembled context string for the LLM
          - graph_sentences:  Raw graph sentences (for debug logging)
          - chunk_ids:        List of chunk_ids used (for traceability)
    """
    if not question or not question.strip():
        log.warning("assemble_from_question() called with empty question.")
        return "", [], []

    top_k = vector_top_k or settings.retrieval.vector_top_k

    # ── Run both retrievers unconditionally ───────────────────────────────
    log.debug("Running graph retriever for: %r", question[:80])
    graph_sentences = graph_retrieve(question)

    log.debug("Running vector retriever for: %r", question[:80])
    vector_chunks = vector_retrieve(question, top_k=top_k)

    # ── Assemble context ──────────────────────────────────────────────────
    context = assemble(graph_sentences, vector_chunks)

    # ── Extract chunk_ids for traceability ────────────────────────────────
    chunk_ids = [c.get("chunk_id", "") for c in vector_chunks if c.get("chunk_id")]

    log.info(
        "Context assembled for question=%r | graph=%d sentences | vector=%d chunks",
        question[:60],
        len(graph_sentences),
        len(vector_chunks),
    )

    return context, graph_sentences, chunk_ids