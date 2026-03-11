"""
run_day1.py — Standalone script to run and verify the Day 1 pipeline.

Run from the project root:
    python run_day1.py

What it does:
    1. Runs the full ingestion pipeline (scrape -> clean -> chunk -> save)
    2. Prints a verification report so you can confirm the output is correct
    3. Prints 3 sample chunks so you can visually inspect cleaning quality

No arguments needed. Edit CUSTOM_URLS below to scrape different pages.
"""

import json
import sys
from pathlib import Path

# ── Allow running from project root without installing the package ────────────
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.ingestion_pipeline import run_ingestion
from ingestion.document_loader import load_chunks
from config import settings

# ── Optional: override the default URL list here ─────────────────────────────
# Set to None to use the DEFAULT_URLS from scraper.py
CUSTOM_URLS = None

# Example of a custom list:
# CUSTOM_URLS = [
#     "https://lotr.fandom.com/wiki/Aragorn",
#     "https://lotr.fandom.com/wiki/Gandalf",
# ]


def main():
    print("\n" + "=" * 60)
    print("  GraphRAG Lore Assistant — Day 1 Ingestion")
    print("=" * 60 + "\n")

    # Run pipeline
    chunks = run_ingestion(urls=CUSTOM_URLS)

    if not chunks:
        print("\n[ERROR] Pipeline produced no chunks. Check logs above.")
        sys.exit(1)

    # ── Verification report ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERIFICATION REPORT")
    print("=" * 60)

    # Re-load from disk to verify file was written correctly
    loaded_chunks = load_chunks(settings.data.chunks_file)

    print(f"\n  chunks.json location : {settings.data.chunks_file}")
    print(f"  Chunks in memory     : {len(chunks)}")
    print(f"  Chunks loaded back   : {len(loaded_chunks)}")
    print(f"  Match                : {'YES' if len(chunks) == len(loaded_chunks) else 'NO - check for write errors'}")

    # Per-page breakdown
    from collections import Counter
    page_counts = Counter(c["page_title"] for c in loaded_chunks)
    print(f"\n  Chunks per page:")
    for title, count in sorted(page_counts.items()):
        print(f"    {count:>3}  {title}")

    # Token stats
    token_counts = [c["token_count"] for c in loaded_chunks]
    print(f"\n  Token stats:")
    print(f"    Min:  {min(token_counts)}")
    print(f"    Max:  {max(token_counts)}")
    print(f"    Avg:  {sum(token_counts) // len(token_counts)}")

    # ── Sample chunks ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SAMPLE CHUNKS (first 3)")
    print("=" * 60)

    for chunk in loaded_chunks[:3]:
        print(f"\n  chunk_id   : {chunk['chunk_id']}")
        print(f"  page_title : {chunk['page_title']}")
        print(f"  tokens     : {chunk['token_count']}")
        print(f"  source_url : {chunk['source_url']}")
        print(f"  text preview:")
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"    {preview}...")

    print("\n" + "=" * 60)
    print("  Day 1 complete. Ready for Day 2 (embeddings + ChromaDB).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()