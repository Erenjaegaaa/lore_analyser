"""
graph/graph_traversal.py — Neo4j graph traversal for the query pipeline.

Given entity names extracted from a user question, this module:
  1. Fetches the immediate neighborhood of each entity (depth ≤ 2)
  2. Optionally finds shortest paths between pairs of entities
  3. Serializes raw Cypher records into human-readable triple sentences

These sentences are passed to context_assembler.py which combines them
with ChromaDB chunks before sending to the LLM.

Public functions:
    get_neighbors(entity_name, depth) -> list[dict]
    get_paths(entity_a, entity_b)     -> list[dict]
    serialize_triples(records)        -> list[str]
"""

from graph.neo4j_client import run_query
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ── Predicate → human-readable verb map ──────────────────────────────────────
# Used by serialize_triples() to turn raw predicate names into prose.

_PREDICATE_VERBS: dict[str, str] = {
    "CHILD_OF":        "is a child of",
    "SIBLING_OF":      "is a sibling of",
    "SPOUSE_OF":       "is the spouse of",
    "HEIR_OF":         "is the heir of",
    "ALLY_OF":         "is an ally of",
    "ENEMY_OF":        "is an enemy of",
    "SERVANT_OF":      "is a servant of",
    "MEMBER_OF":       "is a member of",
    "RULES_OVER":      "rules over",
    "CREATED":         "created",
    "FORGED_BY":       "was forged by",
    "WIELDED":         "wielded",
    "BORN_IN":         "was born in",
    "PART_OF":         "is part of",
    "LOCATED_IN":      "is located in",
    "OCCURRED_AT":     "occurred at",
    "PARTICIPATED_IN": "participated in",
    "RESULTED_IN":     "resulted in",
}


# ── Neighborhood traversal ────────────────────────────────────────────────────

def get_neighbors(
    entity_name: str,
    depth: int | None = None,
) -> list[dict]:
    """
    Return all triples reachable from *entity_name* within *depth* hops.

    Traverses both outgoing and incoming relationships so the caller doesn't
    need to know the direction. Results are deduplicated by (subject, predicate,
    object) before returning.

    Args:
        entity_name: Canonical node name to start traversal from.
        depth:       Maximum traversal depth. Defaults to
                     ``settings.retrieval.graph_traversal_depth``.

    Returns:
        List of triple dicts:
        ``{"subject": str, "predicate": str, "object": str}``
        Capped at ``settings.retrieval.graph_max_triples`` results.
    """
    depth = depth or settings.retrieval.graph_traversal_depth
    max_triples = settings.retrieval.graph_max_triples

    # Match the named node then collect all relationships within depth hops.
    # We use OPTIONAL MATCH in both directions so we capture the full
    # neighborhood regardless of edge direction.
    cypher = """
        MATCH (start {name: $name})
        CALL apoc.path.subgraphNodes(start, {maxLevel: $depth}) YIELD node
        MATCH (a)-[r]->(b)
        WHERE (a = start OR b = start OR a = node OR b = node)
          AND a <> b
        RETURN a.name AS subject,
               type(r)  AS predicate,
               b.name   AS object
        LIMIT $limit
    """

    try:
        records = run_query(cypher, {
            "name":  entity_name,
            "depth": depth,
            "limit": max_triples,
        })
        log.debug(
            "get_neighbors(%r, depth=%d) → %d triples",
            entity_name, depth, len(records),
        )
        return _deduplicate(records)

    except Exception as exc:
        # APOC may not be installed — fall back to a simpler variable-length
        # path query that doesn't require any plugins.
        log.warning(
            "APOC unavailable (%s) — falling back to simple neighbor query.", exc
        )
        return _get_neighbors_simple(entity_name, depth, max_triples)


def _get_neighbors_simple(
    entity_name: str,
    depth: int,
    max_triples: int,
) -> list[dict]:
    """
    Fallback neighborhood query that works without APOC.

    Fetches direct (depth-1) and two-hop (depth-2) neighbors using two
    separate Cypher queries and merges the results.
    """
    results: list[dict] = []

    # Depth-1: direct relationships in both directions
    cypher_d1 = """
        MATCH (start {name: $name})-[r]-(neighbor)
        MATCH (a)-[r2]->(b)
        WHERE (a = start AND b = neighbor) OR (a = neighbor AND b = start)
        RETURN a.name AS subject,
               type(r2) AS predicate,
               b.name   AS object
        LIMIT $limit
    """
    try:
        d1 = run_query(cypher_d1, {"name": entity_name, "limit": max_triples})
        results.extend(d1)
        log.debug("Simple d1 query for %r → %d triples", entity_name, len(d1))
    except Exception as exc:
        log.error("Simple d1 query failed for %r: %s", entity_name, exc)
        return []

    # Depth-2: two-hop neighbors (only if depth ≥ 2 and budget remains)
    if depth >= 2 and len(results) < max_triples:
        remaining = max_triples - len(results)
        cypher_d2 = """
            MATCH (start {name: $name})-[*2]-(hop2)
            MATCH (a)-[r]->(b)
            WHERE (a = start OR b = start)
               OR (a = hop2  OR b = hop2)
            RETURN a.name AS subject,
                   type(r) AS predicate,
                   b.name  AS object
            LIMIT $limit
        """
        try:
            d2 = run_query(cypher_d2, {"name": entity_name, "limit": remaining})
            results.extend(d2)
            log.debug("Simple d2 query for %r → %d triples", entity_name, len(d2))
        except Exception as exc:
            log.warning("Simple d2 query failed for %r: %s", entity_name, exc)

    deduped = _deduplicate(results)[:max_triples]
    log.debug(
        "get_neighbors(%r) simple fallback → %d triples (after dedup)",
        entity_name, len(deduped),
    )
    return deduped


# ── Path queries ──────────────────────────────────────────────────────────────

def get_paths(
    entity_a: str,
    entity_b: str,
    max_hops: int = 4,
) -> list[dict]:
    """
    Find shortest paths between two named entities and return all triples
    along those paths.

    Useful for questions like "How is Frodo related to Bilbo?" where the
    answer requires traversing a chain of relationships.

    Args:
        entity_a: Name of the first entity.
        entity_b: Name of the second entity.
        max_hops: Maximum path length to consider.

    Returns:
        List of triple dicts along the shortest paths found.
        Empty list if no path exists or either entity is not in the graph.
    """
    cypher = """
        MATCH (a {name: $name_a}), (b {name: $name_b}),
              path = shortestPath((a)-[*1..$max_hops]-(b))
        UNWIND relationships(path) AS r
        RETURN startNode(r).name AS subject,
               type(r)           AS predicate,
               endNode(r).name   AS object
    """
    try:
        records = run_query(cypher, {
            "name_a":   entity_a,
            "name_b":   entity_b,
            "max_hops": max_hops,
        })
        log.debug(
            "get_paths(%r, %r) → %d triples", entity_a, entity_b, len(records)
        )
        return _deduplicate(records)
    except Exception as exc:
        log.error("get_paths(%r, %r) failed: %s", entity_a, entity_b, exc)
        return []


# ── Multi-entity entry point ──────────────────────────────────────────────────

def get_triples_for_entities(
    entity_names: list[str],
    depth: int | None = None,
) -> list[dict]:
    """
    Collect graph triples for a list of entity names.

    Runs ``get_neighbors`` for each entity and merges the results, deduplicating
    across entities. Caps the total at ``settings.retrieval.graph_max_triples``.

    This is the primary entry point called by ``graph_retriever.py``.

    Args:
        entity_names: List of canonical entity names from the question.
        depth:        Traversal depth (defaults to config value).

    Returns:
        Deduplicated list of triple dicts, capped at graph_max_triples.
    """
    if not entity_names:
        log.debug("get_triples_for_entities called with empty list — returning []")
        return []

    max_triples = settings.retrieval.graph_max_triples
    all_triples: list[dict] = []

    for name in entity_names:
        triples = get_neighbors(name, depth)
        all_triples.extend(triples)
        if len(all_triples) >= max_triples:
            break

    deduped = _deduplicate(all_triples)[:max_triples]
    log.debug(
        "get_triples_for_entities(%r) → %d triples total (after dedup)",
        entity_names, len(deduped),
    )
    return deduped


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_triples(records: list[dict]) -> list[str]:
    """
    Convert raw triple dicts into human-readable sentences.

    Each dict must have ``subject``, ``predicate``, and ``object`` keys.
    Unknown predicates fall back to a generic "is related to" phrasing.

    Example:
        {"subject": "Aragorn", "predicate": "HEIR_OF", "object": "Isildur"}
        → "Aragorn is the heir of Isildur."

    Args:
        records: List of triple dicts from ``get_neighbors`` or ``get_paths``.

    Returns:
        List of sentence strings, one per triple.
    """
    sentences: list[str] = []

    for rec in records:
        subject   = (rec.get("subject")   or "").strip()
        predicate = (rec.get("predicate") or "").strip().upper()
        obj       = (rec.get("object")    or "").strip()

        if not subject or not predicate or not obj:
            log.debug("Skipping incomplete triple: %r", rec)
            continue

        verb = _PREDICATE_VERBS.get(predicate, "is related to")
        sentence = f"{subject} {verb} {obj}."
        sentences.append(sentence)

    log.debug("serialize_triples: %d records → %d sentences", len(records), len(sentences))
    return sentences


# ── Internal helpers ──────────────────────────────────────────────────────────

def _deduplicate(records: list[dict]) -> list[dict]:
    """
    Remove duplicate triples, keeping first occurrence.
    A triple is considered duplicate if (subject, predicate, object) matches.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for rec in records:
        key = (
            (rec.get("subject")   or "").strip(),
            (rec.get("predicate") or "").strip(),
            (rec.get("object")    or "").strip(),
        )
        if key not in seen and all(key):
            seen.add(key)
            unique.append(rec)
    return unique