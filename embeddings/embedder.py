"""
embeddings/embedder.py — Sentence-transformer wrapper for embedding text.

Public interface:
    embed_text(text: str) -> list[float]
    embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]
"""

from sentence_transformers import SentenceTransformer

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (loaded once, reused)."""
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", settings.embedding.model_name)
        _model = SentenceTransformer(settings.embedding.model_name)
        log.info("Embedding model loaded (dim=%d).", _model.get_sentence_embedding_dimension())
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns a list of floats."""
    model = _get_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of texts in batches.

    Args:
        texts: List of strings to embed.
        batch_size: Number of texts per batch (controls memory usage).

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = _get_model()
    log.info("Embedding %d texts (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    log.info("Embedding complete.")
    return [e.tolist() for e in embeddings]