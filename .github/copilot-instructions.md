# GitHub Copilot Instructions — GraphRAG Lore Assistant

## What this project is
Hybrid GraphRAG question-answering system over Tolkien/Middle-earth lore.
Combines a Neo4j knowledge graph with ChromaDB vector search, answered by Gemini.

## Language & runtime
Python 3.11+. All type hints use the 3.10+ union syntax (`X | Y`, `list[str]`, etc).

---

## Completed modules (do not regenerate)

### embeddings/embedder.py
Module-level lazy singleton (`_model`). Public functions:
- `embed_text(text: str) -> list[float]`
- `embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]`

### embeddings/chroma_store.py
Module-level lazy singletons (`_client`, `_collection`). Public functions:
- `get_collection() -> chromadb.Collection`
- `store_chunks(chunks: list[dict]) -> int`
- `query(query_text: str, top_k: int | None = None) -> list[dict]`
- `count() -> int`
- `reset_collection() -> None`

ChromaDB query result shape:
```python
{"chunk_id": str, "text": str, "metadata": dict, "distance": float}
```

---

## Day 3 target modules

### extraction/prompt_templates.py
Returns a prompt string that instructs Gemini to extract entities and relationships
from a chunk of lore text and return **only valid JSON** — no markdown fences,
no preamble. The JSON schema must match exactly:

```json
{
  "entities": [
    {"name": "Aragorn", "type": "Character", "aliases": ["Strider"]}
  ],
  "relations": [
    {"subject": "Aragorn", "predicate": "HEIR_OF", "object": "Isildur"}
  ]
}
```

Valid entity types: `Character`, `Location`, `Event`, `Artifact`, `Faction`

Valid predicates (18 total — use ONLY these, no others):
```
# Kinship
CHILD_OF        — parent→child lineage
SIBLING_OF      — shared parentage
SPOUSE_OF       — married/bonded pair
HEIR_OF         — designated successor

# Alliance & Enmity
ALLY_OF         — allied characters or factions
ENEMY_OF        — opposed characters or factions
SERVANT_OF      — subject/thrall relationship (Grima→Saruman, Nazgûl→Sauron)

# Faction & Politics
MEMBER_OF       — character belongs to a faction
RULES_OVER      — character holds dominion over location or faction

# Craftsmanship & Artifacts
CREATED         — general creation (wrote a book, built a city)
FORGED_BY       — weapon/ring/jewel made by a character (One Ring→Sauron)
WIELDED         — character carried/used an artifact

# Geography
BORN_IN         — character's place of origin
PART_OF         — location contained within another location (Shire→Eriador)
LOCATED_IN      — artifact physically resides at a location

# Events
OCCURRED_AT     — event took place at a location
PARTICIPATED_IN — character took part in an event (or event involved a character)
RESULTED_IN     — event caused another event or produced an artifact
```

### extraction/entity_extractor.py
Calls Gemini (`gemini-1.5-flash`) with the prompt from `prompt_templates.py`.
Parses JSON response. Attaches `source_chunk_id` to every entity and relation.
Must handle: JSON parse failures, empty responses, Gemini rate limits (exponential
backoff with `time.sleep`). Returns `(entities: list[dict], relations: list[dict])`.

Public function:
```python
def extract_from_chunk(chunk: dict) -> tuple[list[dict], list[dict]]: ...
```

### graph/neo4j_client.py
Thin wrapper around the `neo4j` driver. Lazy singleton (`_driver`).
Reads `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` from `settings` (which reads
from `.env`). Public functions:
```python
def get_driver() -> neo4j.Driver: ...
def run_query(cypher: str, params: dict | None = None) -> list[dict]: ...
def close() -> None: ...
```

### graph/deduplicator.py
Normalises entity names before writing to Neo4j to avoid duplicate nodes.
Uses `rapidfuzz.process.extractOne` with `settings.retrieval.fuzzy_match_threshold`
(default 85.0). Maintains an in-memory canonical name registry `_registry: dict[str, str]`
that maps every seen name to its canonical form.
```python
def normalize(name: str) -> str: ...           # lowercases, strips punctuation
def canonical(name: str, label: str) -> str: ... # returns canonical name or registers new
def reset() -> None: ...                        # clears registry (for tests)
```

### graph/graph_builder.py
Writes entities and relations to Neo4j using `MERGE` (never `CREATE`) so the
script is idempotent. Uses `deduplicator.canonical()` on every name before MERGE.
Sets `source_chunk_id` on every node and relationship.
```python
def merge_entity(entity: dict) -> None: ...
def merge_relation(relation: dict) -> None: ...
def merge_batch(entities: list[dict], relations: list[dict]) -> None: ...
```

### run_day3.py
Orchestrates: load chunks → for each chunk: extract → merge into Neo4j.
Processes chunks with a small delay between Gemini calls to avoid rate limits.
Logs progress every 50 chunks. Idempotent (safe to re-run).

---

## Conventions — always follow these

- **Logging**: `log = get_logger(__name__)` at module top level. Never `print()`.
- **Config**: `from config import settings` — never hardcode paths, model names, or thresholds.
- **Lazy singletons**: use module-level `_var = None` + getter function for
  expensive resources (DB drivers, ML models). Do not use classes unless state
  is genuinely complex.
- **Idempotent writes**: always `MERGE`, never `CREATE` in Neo4j Cypher.
- **No markdown in LLM output**: Gemini prompts must explicitly say
  "Return only valid JSON. No markdown. No explanation."
- **source_chunk_id on everything**: every node and relationship written to Neo4j
  must carry the `source_chunk_id` of the chunk it came from.
- **RapidFuzz threshold**: always read from `settings.retrieval.fuzzy_match_threshold`.
  Never hardcode `85` or any other number.
- **Type hints**: all function signatures must have full type hints.
- **Error handling**: wrap Gemini calls in try/except; log the error and return
  empty lists rather than crashing the whole pipeline.

---

## Config reference
```python
settings.retrieval.fuzzy_match_threshold  # 85.0
settings.retrieval.vector_top_k           # 5
settings.retrieval.graph_max_triples      # 20
settings.retrieval.graph_traversal_depth  # 2
settings.embedding.model_name             # "all-MiniLM-L6-v2"
settings.chroma.persist_path              # "./data/chroma_db"
settings.chroma.collection_name           # "lore_chunks"
settings.data.chunks_file                 # "./data/chunks/chunks.json"
settings.data.raw_dir                     # "./data/raw"
# Neo4j credentials come from .env via settings:
settings.neo4j.uri                        # NEO4J_URI
settings.neo4j.username                   # NEO4J_USERNAME
settings.neo4j.password                   # NEO4J_PASSWORD
```

## .env variables
```
GEMINI_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

## Neo4j node labels and relationship types

### Labels
`Character`, `Location`, `Event`, `Artifact`, `Faction`

### Relationship types (18 total)

| Predicate        | Category           | Typical direction                        |
|------------------|--------------------|------------------------------------------|
| `CHILD_OF`       | Kinship            | Character → Character                    |
| `SIBLING_OF`     | Kinship            | Character → Character                    |
| `SPOUSE_OF`      | Kinship            | Character → Character                    |
| `HEIR_OF`        | Kinship            | Character → Character                    |
| `ALLY_OF`        | Alliance & Enmity  | Character → Character/Faction            |
| `ENEMY_OF`       | Alliance & Enmity  | Character → Character/Faction            |
| `SERVANT_OF`     | Alliance & Enmity  | Character → Character/Faction            |
| `MEMBER_OF`      | Faction & Politics | Character → Faction                      |
| `RULES_OVER`     | Faction & Politics | Character → Location/Faction             |
| `CREATED`        | Craftsmanship      | Character → Artifact/Location            |
| `FORGED_BY`      | Craftsmanship      | Artifact → Character                     |
| `WIELDED`        | Craftsmanship      | Character → Artifact                     |
| `BORN_IN`        | Geography          | Character → Location                     |
| `PART_OF`        | Geography          | Location → Location                      |
| `LOCATED_IN`     | Geography          | Artifact → Location                      |
| `OCCURRED_AT`    | Events             | Event → Location                         |
| `PARTICIPATED_IN`| Events             | Character → Event  /  Event → Character  |
| `RESULTED_IN`    | Events             | Event → Event  /  Event → Artifact       |

### Node properties
`name` (str), `aliases` (list[str]), `source_chunk_id` (str)

### Relationship properties
`source_chunk_id` (str)