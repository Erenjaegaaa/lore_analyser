"""
graph/graph_builder.py — Writes entities and relationships to Neo4j.

Uses MERGE (never CREATE) so every run is idempotent.
All names pass through deduplicator.canonical() before being written.
Every node and relationship carries a ``source_chunk_id`` property.

Supported entity labels  : Character, Location, Event, Artifact, Faction
Supported relationship types: all 18 predicates defined in the project schema.
"""

from graph import neo4j_client
from graph import deduplicator
from utils.logger import get_logger

log = get_logger(__name__)

# ── Valid schema constants ────────────────────────────────────────────────────

_VALID_LABELS: frozenset[str] = frozenset(
    {"Character", "Location", "Event", "Artifact", "Faction"}
)

_VALID_PREDICATES: frozenset[str] = frozenset({
    "CHILD_OF", "SIBLING_OF", "SPOUSE_OF", "HEIR_OF",
    "ALLY_OF", "ENEMY_OF", "SERVANT_OF",
    "MEMBER_OF", "RULES_OVER",
    "CREATED", "FORGED_BY", "WIELDED",
    "BORN_IN", "PART_OF", "LOCATED_IN",
    "OCCURRED_AT", "PARTICIPATED_IN", "RESULTED_IN",
})


# ── Entity writer ─────────────────────────────────────────────────────────────

def merge_entity(entity: dict) -> None:
    """
    MERGE a single entity node into Neo4j.

    Args:
        entity: Dict with keys ``name``, ``type``, ``aliases``,
                ``source_chunk_id``.  Unknown types are skipped.
    """
    raw_name: str = entity.get("name", "").strip()
    label: str = entity.get("type", "").strip()
    aliases: list[str] = entity.get("aliases", []) or []
    source_chunk_id: str = entity.get("source_chunk_id", "")

    if not raw_name or not label:
        log.debug("Skipping entity with missing name or type: %r", entity)
        return

    if label not in _VALID_LABELS:
        log.warning("Unknown entity label %r — skipping %r.", label, raw_name)
        return

    name = deduplicator.canonical(raw_name, label)

    # Cypher: MERGE on (label, name); SET aliases and source_chunk_id.
    # The label is interpolated directly (it's validated against the whitelist above).
    cypher = (
        f"MERGE (n:{label} {{name: $name}}) "
        "SET n.aliases = $aliases, n.source_chunk_id = $source_chunk_id"
    )
    params = {
        "name": name,
        "aliases": aliases,
        "source_chunk_id": source_chunk_id,
    }
    try:
        neo4j_client.run_query(cypher, params)
        log.debug("Merged %s node: %r", label, name)
    except Exception as exc:
        log.error("Failed to merge entity %r (%s): %s", name, label, exc)


# ── Relation writer ───────────────────────────────────────────────────────────

def merge_relation(relation: dict) -> None:
    """
    MERGE a single relationship between two nodes.

    Both subject and object are resolved through the deduplicator.
    The node labels are inferred from what's already in the graph (MATCH
    on name only), so relationships can span different label types naturally.

    Args:
        relation: Dict with keys ``subject``, ``predicate``, ``object``,
                  ``source_chunk_id``.
    """
    subject_name: str = (relation.get("subject") or "").strip()
    predicate: str = (relation.get("predicate") or "").strip().upper()
    object_name: str = (relation.get("object") or "").strip()
    source_chunk_id: str = relation.get("source_chunk_id", "")

    if not subject_name or not predicate or not object_name:
        log.debug("Skipping relation with missing fields: %r", relation)
        return

    if predicate not in _VALID_PREDICATES:
        log.warning("Unknown predicate %r — skipping relation %r→%r.", predicate, subject_name, object_name)
        return

    # Resolve canonical names. We don't know the label at this point, so we
    # try to find an existing node by name. Use a special sentinel label
    # "_any_" to do a cross-label lookup by searching all label buckets.
    sub_canonical = _resolve_canonical(subject_name)
    obj_canonical = _resolve_canonical(object_name)

    # MERGE the relationship; MATCH nodes by name across any label.
    # We use WHERE to match both endpoints flexibly.
    cypher = (
        "MATCH (a {name: $subject}), (b {name: $object}) "
        f"MERGE (a)-[r:{predicate}]->(b) "
        "SET r.source_chunk_id = $source_chunk_id"
    )
    params = {
        "subject": sub_canonical,
        "object": obj_canonical,
        "source_chunk_id": source_chunk_id,
    }
    try:
        neo4j_client.run_query(cypher, params)
        log.debug("Merged relation: (%r)-[%s]->(%r)", sub_canonical, predicate, obj_canonical)
    except Exception as exc:
        log.error(
            "Failed to merge relation (%r)-[%s]->(%r): %s",
            sub_canonical, predicate, obj_canonical, exc,
        )


def _resolve_canonical(name: str) -> str:
    """
    Attempt to resolve *name* to a canonical form by checking each label
    bucket in the deduplicator. Falls back to the raw name if not found.
    """
    from graph.deduplicator import normalize, _registry, _label_index  # noqa: PLC0415
    norm = normalize(name)
    for label in ("Character", "Location", "Event", "Artifact", "Faction"):
        key = (norm, label)
        if key in _registry:
            return _registry[key]
    # Not in registry yet — return original (graph_builder will still try to MATCH)
    return name


# ── Batch writer ──────────────────────────────────────────────────────────────

def merge_batch(entities: list[dict], relations: list[dict]) -> None:
    """
    Merge a batch of entities followed by a batch of relations.

    Entities are always written before relations so the MATCH in
    ``merge_relation`` can find the nodes.

    Args:
        entities: List of entity dicts.
        relations: List of relation dicts.
    """
    for entity in entities:
        merge_entity(entity)
    for relation in relations:
        merge_relation(relation)
    log.debug(
        "Batch complete: %d entities, %d relations merged.",
        len(entities), len(relations),
    )