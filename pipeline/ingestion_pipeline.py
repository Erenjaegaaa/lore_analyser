"""
pipeline/ingestion_pipeline.py — Day 1 ingestion pipeline.

Orchestrates:
    scrape -> save raw -> clean -> chunk -> save chunks.json

Day 2 will extend this to also run:
    -> embed + store in ChromaDB  (Branch A)
    -> LLM extract + write Neo4j  (Branch B)

Public interface:
    run_ingestion(urls: list[str] = None) -> list[dict]
        Returns the list of chunks produced (also saved to disk).
"""

from config import settings
from ingestion.scraper import scrape_pages, DEFAULT_URLS
from ingestion.document_loader import save_raw_pages, save_chunks
from chunking.text_cleaner import clean_page
from chunking.chunker import chunk_pages
from utils.logger import get_logger

log = get_logger(__name__)


def run_ingestion(urls: list[str] = None) -> list[dict]:
    """
    Run the full Day 1 ingestion pipeline.

    Args:
        urls: List of wiki page URLs to scrape.
              Defaults to DEFAULT_URLS in scraper.py if not provided.

    Returns:
        Flat list of all chunk dicts, also saved to settings.data.chunks_file.
    """
    if urls is None:
        urls = DEFAULT_URLS
        log.info("No URLs provided, using %d default URLs.", len(urls))

    # ── Step 1: Scrape ────────────────────────────────────────────────────────
    log.info("=== STEP 1: Scraping %d pages ===", len(urls))
    raw_pages = scrape_pages(urls, delay=1.5)

    if not raw_pages:
        log.error("No pages scraped. Check URLs and network connection.")
        return []

    # ── Step 2: Save raw pages to disk ────────────────────────────────────────
    log.info("=== STEP 2: Saving raw pages ===")
    save_raw_pages(raw_pages, settings.data.raw_dir)

    # ── Step 3: Clean HTML ────────────────────────────────────────────────────
    log.info("=== STEP 3: Cleaning HTML ===")
    cleaned_pages = [clean_page(page) for page in raw_pages]

    # Log a quick sanity check on cleaning quality
    for page in cleaned_pages:
        char_count = len(page.get("clean_text", ""))
        log.debug("  '%s': %d chars after cleaning", page["title"], char_count)

    # ── Step 4: Chunk ─────────────────────────────────────────────────────────
    log.info("=== STEP 4: Chunking ===")
    chunks = chunk_pages(cleaned_pages)

    if not chunks:
        log.error("No chunks produced. Check cleaning output.")
        return []

    # ── Step 5: Save chunks to disk ───────────────────────────────────────────
    log.info("=== STEP 5: Saving chunks ===")
    save_chunks(chunks, settings.data.chunks_file)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=== INGESTION COMPLETE ===")
    log.info("  Pages scraped:  %d", len(raw_pages))
    log.info("  Pages cleaned:  %d", len(cleaned_pages))
    log.info("  Total chunks:   %d", len(chunks))
    log.info("  Chunks saved:   %s", settings.data.chunks_file)

    return chunks