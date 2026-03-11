"""
graph/neo4j_client.py — Thin Neo4j driver wrapper.

Lazy singleton pattern: the driver is created once on first use and reused
for all subsequent calls. Credentials are read from ``settings`` (which reads
from ``.env`` via python-dotenv).

Usage:
    from graph.neo4j_client import run_query, close

    results = run_query("MATCH (n:Character) RETURN n.name LIMIT 5")
"""

import neo4j
from neo4j import GraphDatabase

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_driver: neo4j.Driver | None = None


# ── Driver lifecycle ──────────────────────────────────────────────────────────

def get_driver() -> neo4j.Driver:
    """
    Return the module-level Neo4j driver, creating it on first call.

    Reads ``settings.neo4j.uri``, ``settings.neo4j.username``, and
    ``settings.neo4j.password``.
    """
    global _driver
    if _driver is None:
        uri = settings.neo4j.uri
        username = settings.neo4j.username
        password = settings.neo4j.password
        log.debug("Connecting to Neo4j at %s as %s", uri, username)
        _driver = GraphDatabase.driver(uri, auth=(username, password))
        # Verify connectivity immediately so we fail fast on bad credentials
        _driver.verify_connectivity()
        log.info("Neo4j driver connected: %s", uri)
    return _driver


def close() -> None:
    """Close the driver and reset the singleton."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        log.info("Neo4j driver closed.")


# ── Query helper ──────────────────────────────────────────────────────────────

def run_query(cypher: str, params: dict | None = None) -> list[dict]:
    """
    Execute a Cypher query and return all records as plain dicts.

    Args:
        cypher: Cypher query string.
        params: Optional parameter dict for parameterised queries.

    Returns:
        List of record dicts (may be empty).
    """
    driver = get_driver()
    params = params or {}
    database = settings.neo4j.database

    try:
        with driver.session(database=database) as session:
            result = session.run(cypher, params)
            records = [record.data() for record in result]
            return records
    except Exception as exc:
        log.error("Neo4j query failed: %s | cypher=%r | params=%r", exc, cypher[:120], params)
        raise