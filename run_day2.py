"""
run_day2.py — Day 2 runner: embed chunks and store in ChromaDB.
"""

from config import settings
from ingestion.document_loader import load_chunks
from embeddings.chroma_store import store_chunks, count
from utils.logger import get_logger

log = get_logger(__name__)


def main():
    print()
    print("=" * 60)
    print("  GraphRAG Lore Assistant — Day 2: Embeddings + ChromaDB")
    print("=" * 60)
    print()

    # ── Step 1: Load chunks from disk ─────────────────────────────────────
    log.info("=== STEP 1: Loading chunks from %s ===", settings.data.chunks_file)
    chunks = load_chunks(settings.data.chunks_file)

    if not chunks:
        log.error("No chunks found. Run Day 1 first (python run_day1.py).")
        print("\n[ERROR] No chunks to embed. Run Day 1 first.")
        return

    log.info("Loaded %d chunks.", len(chunks))

    # ── Step 2: Embed and store in ChromaDB ───────────────────────────────
    log.info("=== STEP 2: Embedding and storing in ChromaDB ===")
    added = store_chunks(chunks)

    # ── Summary ───────────────────────────────────────────────────────────
    total = count()
    log.info("=== DAY 2 COMPLETE ===")
    log.info("  Chunks loaded:    %d", len(chunks))
    log.info("  New chunks added: %d", added)
    log.info("  Total in ChromaDB: %d", total)

    print(f"\n[OK] Day 2 complete — {total} chunks in ChromaDB ({added} newly added).")


if __name__ == "__main__":
    main()