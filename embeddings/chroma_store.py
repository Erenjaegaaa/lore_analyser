"""
embeddings/chroma_store.py — ChromaDB persistent vector store for lore chunks.

Public interface:
    get_collection() -> chromadb.Collection
    store_chunks(chunks: list[dict]) -> int
    query(query_text: str, top_k: int = 5) -> list[dict]
    count() -> int
    reset_collection() -> None
"""

import chromadb

from config import settings
from embeddings.embedder import embed_text, embed_batch
from utils.logger import get_logger

log = get_logger(__name__)

_client = None
_collection = None


def _get_client() -> chromadb.PersistentClient:
    """Lazy-initialise the ChromaDB persistent client."""
    global _client
    if _client is None:
        log.info("Initialising ChromaDB client at: %s", settings.chroma.persist_path)
        _client = chromadb.PersistentClient(path=settings.chroma.persist_path)
    return _client


def get_collection() -> chromadb.Collection:
    """Get (or create) the ChromaDB collection for lore chunks."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB collection '%s' ready (%d existing documents).",
            settings.chroma.collection_name,
            _collection.count(),
        )
    return _collection


def store_chunks(chunks: list[dict]) -> int:
    """
    Embed and store chunks into ChromaDB.

    Skips chunks whose chunk_id already exists in the collection.
    Uses batch embedding for efficiency.

    Args:
        chunks: List of chunk dicts (must have "chunk_id", "text",
                "page_slug", "source_url", "page_title").

    Returns:
        Number of new chunks added.
    """
    collection = get_collection()

    # Filter out chunks that are already stored
    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        log.info("All %d chunks already in ChromaDB — nothing to add.", len(chunks))
        return 0

    log.info(
        "%d / %d chunks are new — embedding and storing...",
        len(new_chunks), len(chunks),
    )

    # Prepare data for ChromaDB upsert
    ids = [c["chunk_id"] for c in new_chunks]
    texts = [c["text"] for c in new_chunks]
    metadatas = [
        {
            "page_slug": c["page_slug"],
            "source_url": c["source_url"],
            "page_title": c["page_title"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
        }
        for c in new_chunks
    ]

    # Embed all texts in one batch
    embeddings = embed_batch(texts)

    # ChromaDB has a max batch size; upsert in slices of 500
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        log.debug("Upserted batch %d–%d to ChromaDB.", i, end - 1)

    log.info("Stored %d new chunks in ChromaDB (total: %d).", len(new_chunks), collection.count())
    return len(new_chunks)


def query(query_text: str, top_k: int | None = None) -> list[dict]:
    """
    Query ChromaDB with a text string.

    Args:
        query_text: The search query.
        top_k: Number of results to return (defaults to settings.retrieval.vector_top_k).

    Returns:
        List of dicts with keys: chunk_id, text, metadata, distance.
    """
    if top_k is None:
        top_k = settings.retrieval.vector_top_k

    collection = get_collection()
    query_embedding = embed_text(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten ChromaDB's nested list structure
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    log.debug("ChromaDB query returned %d results.", len(hits))
    return hits


def count() -> int:
    """Return the number of documents in the collection."""
    return get_collection().count()


def reset_collection() -> None:
    """Delete and recreate the collection (destructive)."""
    global _collection
    client = _get_client()
    client.delete_collection(settings.chroma.collection_name)
    _collection = None
    log.warning("Deleted ChromaDB collection '%s'.", settings.chroma.collection_name)
    get_collection()  # recreate it