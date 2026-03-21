"""
retrieval/vector_retriever.py — ChromaDB vector similarity search.

Wraps chroma_store.query() with logging and config-driven top_k.
Returns top-K most semantically similar chunks for a given query text.

Public functions:
    retrieve(query_text, top_k) -> list[dict]
"""

from embeddings.chroma_store import query as chroma_query
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


def retrieve(
    query_text: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Return the top-K most semantically similar chunks from ChromaDB.

    Args:
        query_text: The user's question or any search string.
        top_k:      Number of chunks to return. Defaults to
                    ``settings.retrieval.vector_top_k``.

    Returns:
        List of chunk dicts ordered by similarity (closest first):
        ``{"chunk_id": str, "text": str, "metadata": dict, "distance": float}``
        Empty list if ChromaDB is empty or query fails.
    """
    top_k = top_k or settings.retrieval.vector_top_k

    if not query_text or not query_text.strip():
        log.warning("retrieve() called with empty query — returning []")
        return []

    try:
        results = chroma_query(query_text, top_k=top_k)
        log.debug(
            "Vector retrieval: query=%r → %d chunks returned",
            query_text[:80], len(results),
        )
        return results
    except Exception as exc:
        log.error("Vector retrieval failed: %s", exc)
        return []