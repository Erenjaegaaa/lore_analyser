"""
run_day3.py — Day 3 pipeline: entity extraction → Neo4j ingestion.

Flow:
  1. Load chunks.json  (output of Day 1/2)
  2. For each chunk:
       a. Call Gemini to extract entities + relations
       b. MERGE into Neo4j via graph_builder
  3. Log progress every 50 chunks; log summary at the end.

This script is idempotent: re-running it will MERGE (not duplicate) nodes
and relationships, and skips chunks that produced no extractions.

Usage:
    python run_day3.py
    python run_day3.py --limit 100          # process first N chunks only
    python run_day3.py --delay 1.5          # seconds between Gemini calls
    python run_day3.py --start 200          # resume from chunk index 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import settings
from extraction.entity_extractor import extract_from_chunk
from graph.graph_builder import merge_batch
from graph.neo4j_client import close as close_neo4j
from graph import deduplicator
from utils.logger import get_logger

log = get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DELAY: float = 1.0       # seconds between Gemini API calls
PROGRESS_EVERY: int = 50         # log progress every N chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_chunks(path: str) -> list[dict]:
    """Load chunks.json and return list of chunk dicts."""
    p = Path(path)
    if not p.exists():
        log.error("chunks.json not found at %s", path)
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    chunks_file: str | None = None,
    limit: int | None = None,
    start: int = 0,
    delay: float = DEFAULT_DELAY,
) -> None:
    """
    Run the Day 3 pipeline.

    Args:
        chunks_file: Path to chunks.json (defaults to settings value).
        limit:       Maximum number of chunks to process (None = all).
        start:       Zero-based index of the first chunk to process.
        delay:       Seconds to sleep between Gemini calls.
    """
    chunks_path = chunks_file or settings.data.chunks_file
    all_chunks = load_chunks(chunks_path)

    # Slice for resume / partial runs
    chunks = all_chunks[start:]
    if limit is not None:
        chunks = chunks[:limit]

    total = len(chunks)
    log.info(
        "Processing %d chunks (start=%d, limit=%s, delay=%.2fs).",
        total, start, limit, delay,
    )

    # Reset deduplicator for a clean run (no carry-over from previous imports)
    deduplicator.reset()

    stats = {
        "processed": 0,
        "skipped_empty": 0,
        "total_entities": 0,
        "total_relations": 0,
        "errors": 0,
    }

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"<index {start + i}>")

        # ── Extract ───────────────────────────────────────────────────────
        try:
            entities, relations = extract_from_chunk(chunk)
        except Exception as exc:
            # extract_from_chunk should never raise, but guard anyway
            log.error("[%s] Unexpected error during extraction: %s", chunk_id, exc)
            stats["errors"] += 1
            entities, relations = [], []

        if not entities and not relations:
            log.debug("[%s] No extractions — skipping Neo4j write.", chunk_id)
            stats["skipped_empty"] += 1
        else:
            # ── Merge into Neo4j ──────────────────────────────────────────
            try:
                merge_batch(entities, relations)
                stats["total_entities"] += len(entities)
                stats["total_relations"] += len(relations)
            except Exception as exc:
                log.error("[%s] Neo4j merge failed: %s", chunk_id, exc)
                stats["errors"] += 1

        stats["processed"] += 1

        # ── Progress logging ──────────────────────────────────────────────
        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == total:
            log.info(
                "Progress: %d/%d chunks | entities=%d | relations=%d | skipped=%d | errors=%d",
                i + 1,
                total,
                stats["total_entities"],
                stats["total_relations"],
                stats["skipped_empty"],
                stats["errors"],
            )

        # ── Rate-limit delay ──────────────────────────────────────────────
        if delay > 0 and i < total - 1:
            time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Day 3 complete.")
    log.info("  Chunks processed : %d", stats["processed"])
    log.info("  Skipped (empty)  : %d", stats["skipped_empty"])
    log.info("  Entities merged  : %d", stats["total_entities"])
    log.info("  Relations merged : %d", stats["total_relations"])
    log.info("  Errors           : %d", stats["errors"])
    log.info("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 3: extract entities from chunks and ingest into Neo4j."
    )
    parser.add_argument(
        "--chunks-file",
        default=None,
        help=f"Path to chunks.json (default: {settings.data.chunks_file})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to process (default: all).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of first chunk to process (for resuming).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to sleep between Gemini API calls (default: {DEFAULT_DELAY}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(
            chunks_file=args.chunks_file,
            limit=args.limit,
            start=args.start,
            delay=args.delay,
        )
    finally:
        close_neo4j()