"""
graph/deduplicator.py — Entity name deduplication using RapidFuzz.

Before writing any entity to Neo4j, run its name through ``canonical()``.
The registry maps ``(normalized_name, label)`` → canonical_name so that
"Sauron" as a Character and "Sauron" as a Faction are tracked independently,
and near-duplicates like "Aragorn" / "Aragorn " / "aragorn" collapse to the
first-seen form.

Threshold is read from ``settings.retrieval.fuzzy_match_threshold`` (default 85.0).
"""

import re
import string

from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import token_sort_ratio

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Registry: (normalized_name, label) → canonical display name
_registry: dict[tuple[str, str], str] = {}

# Index for fast fuzzy lookup: label → list[normalized_names]
_label_index: dict[str, list[str]] = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """
    Return a normalised version of *name* for fuzzy comparison.

    Lowercases, strips leading/trailing whitespace, and removes punctuation
    (except hyphens inside words, which are meaningful in Tolkien names).
    """
    name = name.strip().lower()
    # Remove punctuation except internal hyphens
    # Strategy: remove all non-alphanumeric/non-hyphen/non-space chars,
    # then collapse multiple spaces.
    name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _best_match(norm: str, label: str) -> str | None:
    """
    Find the best fuzzy match for *norm* among names already in the registry
    for the same *label*. Returns the matching normalized name or None.
    """
    candidates = _label_index.get(label, [])
    if not candidates:
        return None

    result = rf_process.extractOne(
        norm,
        candidates,
        scorer=token_sort_ratio,
        score_cutoff=settings.retrieval.fuzzy_match_threshold,
    )
    if result is not None:
        matched_norm, score, _ = result
        log.debug(
            "Fuzzy match: %r → %r (score=%.1f, label=%s)", norm, matched_norm, score, label
        )
        return matched_norm
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def canonical(name: str, label: str) -> str:
    """
    Return the canonical display name for *name* under *label*.

    If an existing registry entry is within the fuzzy threshold, the existing
    canonical name is returned (deduplication). Otherwise *name* is registered
    as a new canonical entry and returned as-is.

    Args:
        name:  Display name as extracted by the LLM (e.g. "Aragorn").
        label: Entity type label (e.g. "Character", "Location").

    Returns:
        Canonical display name string.
    """
    if not name or not name.strip():
        return name

    norm = normalize(name)
    key = (norm, label)

    # Exact normalised match — return the stored canonical name directly
    if key in _registry:
        return _registry[key]

    # Fuzzy match against existing entries for this label
    matched_norm = _best_match(norm, label)
    if matched_norm is not None:
        matched_key = (matched_norm, label)
        existing_canonical = _registry[matched_key]
        # Register the new variant pointing to the same canonical
        _registry[key] = existing_canonical
        _label_index.setdefault(label, []).append(norm)
        log.debug(
            "Deduplicated %r → %r (label=%s)", name, existing_canonical, label
        )
        return existing_canonical

    # New entry — register using the original display name as canonical
    _registry[key] = name
    _label_index.setdefault(label, []).append(norm)
    log.debug("Registered new canonical: %r (label=%s)", name, label)
    return name


def reset() -> None:
    """Clear the registry and index. Useful for tests and re-runs."""
    global _registry, _label_index
    _registry = {}
    _label_index = {}
    log.debug("Deduplicator registry reset.")