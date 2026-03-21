"""
retrieval/graph_retriever.py — Graph-based retrieval for the query pipeline.

Flow:
  1. Extract candidate entity names from the user question using a simple
     noun-phrase heuristic (no LLM call needed here — fast and cheap).
  2. Fuzzy-match each candidate against all node names in Neo4j using
     RapidFuzz at the configured threshold.
  3. Pass matched canonical names to graph_traversal.get_triples_for_entities()
  4. Return serialized triple sentences ready for context assembly.

Public functions:
    retrieve(question) -> list[str]   # serialized triple sentences
    match_entities(candidates)        -> list[str]   # canonical Neo4j names
    extract_candidates(question)      -> list[str]   # raw noun phrases
"""

import re

from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import token_sort_ratio

from graph.neo4j_client import run_query
from graph.graph_traversal import get_triples_for_entities, serialize_triples
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# ── Stopwords for candidate extraction ───────────────────────────────────────
# Common question words and English function words that should never be treated
# as entity candidates, even if they are capitalized or title-cased.

_STOPWORDS = {
    "Who", "What", "Where", "When", "Why", "How", "Which", "Is", "Are",
    "Was", "Were", "Did", "Do", "Does", "The", "A", "An", "In", "Of",
    "And", "Or", "But", "To", "From", "By", "For", "With", "At", "On",
    "That", "This", "Their", "There", "They", "Have", "Been", "Will",
    "About", "Tell", "Deal", "Role", "Like", "What", "Does", "Know",
    "More", "Also", "Very", "Just", "Some", "Into", "Than", "Then",
    "Really", "Actually", "Basically", "Generally", "Specifically",
}

# ── Neo4j name index (lazy, cached per process) ───────────────────────────────
# We load all node names once and reuse across calls to avoid repeated DB hits.

_node_index: list[str] | None = None


def _get_node_index() -> list[str]:
    """
    Return a flat list of all node names currently in Neo4j.
    Cached in module-level _node_index after first call.
    """
    global _node_index
    if _node_index is None:
        try:
            records = run_query("MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name")
            _node_index = [r["name"] for r in records if r.get("name")]
            log.debug("Loaded %d node names from Neo4j into index.", len(_node_index))
        except Exception as exc:
            log.error("Failed to load node index from Neo4j: %s", exc)
            _node_index = []
    return _node_index


def invalidate_node_index() -> None:
    """Force reload of the node name index on next call. Call after ingestion."""
    global _node_index
    _node_index = None
    log.debug("Node index invalidated.")


# ── Candidate extraction ──────────────────────────────────────────────────────

def extract_candidates(question: str) -> list[str]:
    """
    Extract candidate entity names from a natural language question.

    Uses two strategies:
      1. Capitalized word sequences — high precision for properly typed names
         e.g. "Helm's Deep", "One Ring", "Gil-galad"
      2. All words > 3 chars title-cased — catches lowercase entity names
         e.g. "gandalf" → "Gandalf", "mordor" → "Mordor"

    Both strategies filter against _STOPWORDS to remove common English words.
    No LLM call needed — fast and cheap.

    Examples:
        "Who is Aragorn's father?"           → ["Aragorn"]
        "what is the deal with gandalf"      → ["Gandalf"]
        "Where was the One Ring forged?"     → ["One Ring"]
        "tell me about mordor"               → ["Mordor"]

    Args:
        question: The user's raw question string.

    Returns:
        List of candidate strings (may include false positives).
    """
    if not question:
        return []

    candidates = []
    seen: set[str] = set()

    # Strategy 1 — capitalized word sequences (high precision)
    # Matches sequences like "Helm's Deep", "Gil-galad", "One Ring"
    pattern = r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)*)\b"
    for m in re.findall(pattern, question):
        if m not in _STOPWORDS and len(m) > 1 and m not in seen:
            candidates.append(m)
            seen.add(m)

    # Strategy 2 — all words > 3 chars, title-cased for matching
    # This catches lowercase entity names like "gandalf", "mordor", "frodo"
    # Title-casing aligns with how names are stored in Neo4j
    for word in re.findall(r"\b[a-zA-Z'\-]{4,}\b", question):
        titled = word.strip("'").title()
        if titled not in _STOPWORDS and titled not in seen:
            candidates.append(titled)
            seen.add(titled)

    log.debug("extract_candidates(%r) → %r", question[:80], candidates)
    return candidates


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def match_entities(candidates: list[str]) -> list[str]:
    """
    Fuzzy-match candidate strings against all node names in Neo4j.

    Uses RapidFuzz token_sort_ratio at the configured threshold so that
    partial matches like "Aragorn" → "Aragorn" and near-matches like
    "Gollum" → "Gollum" are found even with minor variations.

    Args:
        candidates: Raw candidate strings from extract_candidates().

    Returns:
        Deduplicated list of canonical Neo4j node names that matched.
    """
    if not candidates:
        return []

    node_names = _get_node_index()
    if not node_names:
        log.warning("Node index is empty — no graph entities to match against.")
        return []

    threshold = settings.retrieval.fuzzy_match_threshold
    matched: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        result = rf_process.extractOne(
            candidate,
            node_names,
            scorer=token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            canonical_name, score, _ = result
            if canonical_name not in seen:
                seen.add(canonical_name)
                matched.append(canonical_name)
                log.debug(
                    "Matched %r → %r (score=%.1f)", candidate, canonical_name, score
                )
        else:
            log.debug("No match for candidate %r at threshold %.1f", candidate, threshold)

    log.debug("match_entities: %d candidates → %d matched", len(candidates), len(matched))
    return matched


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve(question: str) -> list[str]:
    """
    Full graph retrieval pipeline for a user question.

    Extracts entity candidates from the question, fuzzy-matches them to
    Neo4j nodes, traverses the graph neighborhood, and returns serialized
    triple sentences ready for the context assembler.

    Args:
        question: The user's natural language question.

    Returns:
        List of human-readable triple sentences, e.g.:
        ["Aragorn is the heir of Isildur.", "Aragorn rules over Gondor."]
        Empty list if no entities matched or graph traversal returns nothing.
    """
    if not question or not question.strip():
        log.warning("graph_retriever.retrieve() called with empty question.")
        return []

    # Step 1 — extract candidate entity names from question
    candidates = extract_candidates(question)
    if not candidates:
        log.debug("No candidates extracted from question — skipping graph retrieval.")
        return []

    # Step 2 — fuzzy match candidates to canonical Neo4j node names
    matched = match_entities(candidates)
    if not matched:
        log.debug("No entities matched in Neo4j — skipping graph traversal.")
        return []

    # Step 3 — traverse the graph for matched entities
    triples = get_triples_for_entities(matched)
    if not triples:
        log.debug("Graph traversal returned no triples for entities: %r", matched)
        return []

    # Step 4 — serialize triples to human-readable sentences
    sentences = serialize_triples(triples)
    log.debug(
        "graph_retriever.retrieve() → %d sentences for entities %r",
        len(sentences), matched,
    )
    return sentences