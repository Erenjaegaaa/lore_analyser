"""
ingestion/document_loader.py — Saves and loads raw scraped pages and chunks to/from disk.

All intermediate data is written to the data/ directory so that:
  - You can re-run later pipeline stages without re-scraping
  - You can inspect intermediate output at any stage
  - Pipeline stages are decoupled (Day 2 reads chunks.json, doesn't re-scrape)

Public interface:
    save_raw_pages(pages: list[dict], output_dir: str) -> None
    load_raw_pages(input_dir: str) -> list[dict]
    save_chunks(chunks: list[dict], output_file: str) -> None
    load_chunks(input_file: str) -> list[dict]
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)


# ── Raw pages ─────────────────────────────────────────────────────────────────

def save_raw_pages(pages: list[dict], output_dir: str) -> None:
    """
    Save each scraped page as a separate JSON file in output_dir.

    Files are named by slug derived from the URL so they're human-readable.
    The HTML field is preserved so text_cleaner can be re-run without re-scraping.

    Example output: data/raw/aragorn.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for page in pages:
        slug = _url_to_slug(page["url"])
        filepath = out / f"{slug}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)

        log.debug("Saved raw page: %s", filepath)
        saved += 1

    log.info("Saved %d raw pages to %s", saved, output_dir)


def load_raw_pages(input_dir: str) -> list[dict]:
    """
    Load all raw page JSON files from input_dir.

    Returns:
        List of page dicts in the same format produced by scraper.py.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        log.error("Raw pages directory does not exist: %s", input_dir)
        return []

    pages = []
    for filepath in sorted(input_path.glob("*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            page = json.load(f)
        pages.append(page)
        log.debug("Loaded raw page: %s", filepath.name)

    log.info("Loaded %d raw pages from %s", len(pages), input_dir)
    return pages


# ── Chunks ────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_file: str) -> None:
    """
    Save all chunks to a single JSON file.

    Includes a metadata header with timestamp and chunk count so you can
    quickly verify the file without parsing the whole array.

    Example output: data/chunks/chunks.json
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(chunks),
        },
        "chunks": chunks,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info("Saved %d chunks to %s", len(chunks), output_file)


def load_chunks(input_file: str) -> list[dict]:
    """
    Load chunks from a chunks.json file produced by save_chunks().

    Returns:
        List of chunk dicts. Empty list if file does not exist.
    """
    path = Path(input_file)
    if not path.exists():
        log.error("Chunks file does not exist: %s", input_file)
        return []

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", [])
    meta = payload.get("metadata", {})
    log.info(
        "Loaded %d chunks from %s (created: %s)",
        len(chunks), input_file, meta.get("created_at", "unknown")
    )
    return chunks


# ── Internal ──────────────────────────────────────────────────────────────────

def _url_to_slug(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    slug = slug.lower()
    slug = re.sub(r"[^\w]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug