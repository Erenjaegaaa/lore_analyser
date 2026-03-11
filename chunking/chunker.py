"""
chunking/chunker.py — Splits cleaned text into overlapping token-based chunks
and assigns deterministic chunk_ids.

Uses LangChain's RecursiveCharacterTextSplitter which respects natural text
boundaries (paragraphs > sentences > words) before falling back to hard splits.

Public interface:
    chunk_page(page: dict) -> list[dict]
        Takes a cleaned page dict (must have "clean_text", "url", "title").
        Returns a list of chunk dicts.

    chunk_pages(pages: list[dict]) -> list[dict]
        Convenience wrapper to chunk all pages in one call.

Chunk dict schema:
    {
        "chunk_id":    str,   # e.g. "aragorn_003"
        "page_slug":   str,   # e.g. "aragorn"
        "source_url":  str,
        "page_title":  str,
        "text":        str,   # the chunk text
        "token_count": int,   # approximate token count
        "chunk_index": int,   # position within the page (0-based)
    }

Config used:
    settings.chunking.chunk_size
    settings.chunking.chunk_overlap
    settings.chunking.min_chunk_size
"""

import re
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# cl100k_base is used by GPT-4 and is a good general-purpose tokenizer.
# It gives more accurate token counts than character-based estimates.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


# ── Splitter (initialised once at module load) ────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunking.chunk_size,
    chunk_overlap=settings.chunking.chunk_overlap,
    length_function=_count_tokens,          # measure size in tokens, not chars
    separators=["\n\n", "\n", ". ", " ", ""],  # respect natural boundaries
)


# ── Slug helper ───────────────────────────────────────────────────────────────

def _url_to_slug(url: str) -> str:
    """
    Convert a wiki URL to a clean lowercase slug for use in chunk_ids.

    Example:
        "https://lotr.fandom.com/wiki/Frodo_Baggins" -> "frodo_baggins"
    """
    # Take the last path segment
    slug = url.rstrip("/").split("/")[-1]
    # Lowercase and replace non-alphanumeric (except underscores) with _
    slug = slug.lower()
    slug = re.sub(r"[^\w]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


# ── Public interface ──────────────────────────────────────────────────────────

def chunk_page(page: dict) -> list[dict]:
    """
    Split a single cleaned page into overlapping chunks.

    Args:
        page: Must contain keys: "clean_text", "url", "title".
              Produced by chunking.text_cleaner.clean_page().

    Returns:
        List of chunk dicts. Empty list if page has no usable text.
    """
    text = page.get("clean_text", "").strip()
    if not text:
        log.warning("Empty clean_text for page: %s", page.get("url", "unknown"))
        return []

    slug = _url_to_slug(page["url"])
    raw_chunks = _splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        token_count = _count_tokens(chunk_text)

        # Discard chunks that are too small to be useful
        if token_count < settings.chunking.min_chunk_size:
            log.debug("Skipping small chunk (%d tokens) from %s", token_count, slug)
            continue

        chunk = {
            "chunk_id": f"{slug}_{i:03d}",
            "page_slug": slug,
            "source_url": page["url"],
            "page_title": page["title"],
            "text": chunk_text,
            "token_count": token_count,
            "chunk_index": i,
        }
        chunks.append(chunk)

    log.info(
        "Chunked '%s' -> %d chunks (from %d raw splits)",
        page["title"], len(chunks), len(raw_chunks)
    )
    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunk all pages and return a flat list of all chunks.

    Args:
        pages: List of cleaned page dicts (each must have clean_text, url, title).

    Returns:
        Flat list of all chunk dicts across all pages.
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_page(page)
        all_chunks.extend(page_chunks)

    log.info(
        "Total chunks across %d pages: %d",
        len(pages), len(all_chunks)
    )
    return all_chunks