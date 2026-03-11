# GraphRAG Lore Assistant — Project Context

## Purpose
Hybrid GraphRAG question-answering system over Tolkien/Middle-earth lore.
Combines a Neo4j knowledge graph with ChromaDB vector search for retrieval,
then uses an LLM to generate answers grounded in retrieved context.

## Data Source
Wikipedia (en.wikipedia.org) — switched from lotr.fandom.com due to Cloudflare blocking.
Wikipedia uses MediaWiki HTML structure (mw-parser-output div) — same as Fandom.

## Current Dataset
- 71 Wikipedia pages scraped (characters, locations, events, artifacts, books)
- 1115 chunks produced and embedded
- Token stats: min 99, max 499, avg 464 (target 500, overlap 100)
- Raw pages: data/raw/ (one JSON per page, HTML preserved)
- Chunks: data/chunks/chunks.json
- ChromaDB: data/chroma_db/ (1115 chunks, cosine similarity, all-MiniLM-L6-v2)

---

## Tech Stack
- Language: Python 3.11+
- LLM API: Gemini (google-generativeai), model: gemini-1.5-flash
- Graph DB: Neo4j local (Neo4j Desktop), driver: neo4j==5.x
- Vector DB: ChromaDB (persistent), collection: lore_chunks
- Embeddings: sentence-transformers, model: all-MiniLM-L6-v2 (384 dims)
- Fuzzy matching: RapidFuzz (threshold 85.0)
- Scraping: httpx + BeautifulSoup4 + lxml
- Chunking: LangChain RecursiveCharacterTextSplitter
- Config: config.py (dataclasses) + .env (python-dotenv)
- Logging: utils/logger.py → get_logger(__name__)

---

## Project File Structure

```
graphrag-lore-assistant/
│
├── run_day1.py               ← Day 1 runner (scrape → clean → chunk → save)
├── run_day2.py               ← Day 2 runner (embed → ChromaDB) ✅
├── run_day3.py               ← Day 3 runner (extract → Neo4j)
├── main.py                   ← CLI: python main.py ingest / query / eval
├── config.py                 ← All settings via dataclasses + .env
├── requirements.txt
├── .env                      ← Credentials (not committed)
├── .env.example              ← Template for .env
├── .gitignore
├── PROJECT_CONTEXT.md        ← This file
│
├── ingestion/
│   ├── scraper.py            ← httpx scraper, Wikipedia URLs, DEFAULT_URLS list
│   └── document_loader.py    ← save/load raw pages + chunks.json to disk
│
├── chunking/
│   ├── text_cleaner.py       ← strips HTML boilerplate, returns plain prose
│   └── chunker.py            ← overlapping token chunks, assigns chunk_id
│
├── embeddings/
│   ├── embedder.py           ← sentence-transformers wrapper          ✅
│   │                            embed_text(str) -> list[float]
│   │                            embed_batch(list[str]) -> list[list[float]]
│   │                            Lazy singleton: _model loaded once on first call
│   └── chroma_store.py       ← ChromaDB persistent collection         ✅
│                                get_collection() -> chromadb.Collection
│                                store_chunks(chunks) -> int (new added)
│                                query(query_text, top_k) -> list[dict]
│                                count() -> int
│                                reset_collection() -> None
│
├── extraction/
│   ├── prompt_templates.py   ← LLM prompts for entity+relationship extraction
│   └── entity_extractor.py   ← calls Gemini, returns structured entities+relations
│
├── graph/
│   ├── neo4j_client.py       ← Neo4j driver wrapper, run_query()
│   ├── deduplicator.py       ← normalize + RapidFuzz before MERGE
│   ├── graph_builder.py      ← writes nodes/relationships to Neo4j via MERGE
│   └── graph_traversal.py    ← Cypher query templates for neighborhood/path queries
│
├── retrieval/
│   ├── vector_retriever.py   ← ChromaDB similarity search, returns top-K chunks
│   ├── graph_retriever.py    ← entity extraction from question → fuzzy match → traversal
│   └── context_assembler.py  ← merges graph facts + chunks into LLM prompt context
│
├── pipeline/
│   ├── ingestion_pipeline.py ← orchestrates full ingest flow
│   └── query_pipeline.py     ← orchestrates full query flow
│
├── evaluation/
│   ├── eval_runner.py        ← runs questions.json through query pipeline
│   └── metrics.py            ← summarises results files
│
├── utils/
│   ├── logger.py             ← get_logger(__name__) used in every module
│   └── helpers.py            ← normalize_entity_name, serialize_triple, chunk_id
│
├── tests/
│   ├── test_chunker.py
│   ├── test_extractor.py
│   └── test_retriever.py
│
└── data/
    ├── raw/                  ← scraped HTML as JSON, one file per page
    ├── cleaned/              ← (optional) cleaned text files
    ├── chunks/
    │   └── chunks.json       ← 1115 chunks, primary input for Day 2+
    ├── chroma_db/            ← ChromaDB persisted vector index (1115 docs) ✅
    └── evaluation/
        └── questions.json    ← 10 typed evaluation questions
```

---

## Key Config Values (config.py → settings)

```python
settings.chunking.chunk_size        = 500
settings.chunking.chunk_overlap     = 100
settings.chunking.min_chunk_size    = 50

settings.retrieval.vector_top_k         = 5
settings.retrieval.graph_max_triples    = 20
settings.retrieval.graph_traversal_depth = 2
settings.retrieval.fuzzy_match_threshold = 85.0

settings.embedding.model_name   = "all-MiniLM-L6-v2"
settings.chroma.persist_path    = "./data/chroma_db"
settings.chroma.collection_name = "lore_chunks"
settings.data.chunks_file       = "./data/chunks/chunks.json"
settings.data.raw_dir           = "./data/raw"
```

---

## Data Schemas

### Chunk dict (chunks.json)
```json
{
  "chunk_id":    "aragorn_003",
  "page_slug":   "aragorn",
  "source_url":  "https://en.wikipedia.org/wiki/Aragorn",
  "page_title":  "Aragorn - Wikipedia",
  "text":        "...",
  "token_count": 491,
  "chunk_index": 3
}
```

### ChromaDB query result dict
```python
{
    "chunk_id":  "aragorn_003",
    "text":      "...",
    "metadata":  {
        "page_slug":   "aragorn",
        "source_url":  "https://en.wikipedia.org/wiki/Aragorn",
        "page_title":  "Aragorn - Wikipedia",
        "chunk_index": 3,
        "token_count": 491,
    },
    "distance":  0.1231,   # cosine distance — lower = more similar
}
```

### Neo4j Node Schema
```
(:Character  {name, aliases, source_chunk_id})
(:Location   {name, aliases, source_chunk_id})
(:Event      {name, aliases, source_chunk_id})
(:Artifact   {name, aliases, source_chunk_id})
(:Faction    {name, aliases, source_chunk_id})
```

### Neo4j Relationship Schema

#### Kinship
```
(:Character)-[:CHILD_OF]->(:Character)
(:Character)-[:SIBLING_OF]->(:Character)
(:Character)-[:SPOUSE_OF]->(:Character)
(:Character)-[:HEIR_OF]->(:Character)
```

#### Alliance & Enmity
```
(:Character)-[:ALLY_OF]->(:Character)
(:Character)-[:ENEMY_OF]->(:Character)
(:Character)-[:SERVANT_OF]->(:Character)
(:Character)-[:SERVANT_OF]->(:Faction)
```

#### Faction & Politics
```
(:Character)-[:MEMBER_OF]->(:Faction)
(:Character)-[:RULES_OVER]->(:Location)
(:Character)-[:RULES_OVER]->(:Faction)
```

#### Craftsmanship & Artifacts
```
(:Character)-[:CREATED]->(:Artifact)
(:Character)-[:FORGED_BY]->(:Artifact)   # inverse: Artifact forged by Character
(:Artifact)-[:FORGED_BY]->(:Character)
(:Character)-[:WIELDED]->(:Artifact)
```

#### Geography
```
(:Character)-[:BORN_IN]->(:Location)
(:Location)-[:PART_OF]->(:Location)
(:Artifact)-[:LOCATED_IN]->(:Location)
```

#### Events
```
(:Event)-[:OCCURRED_AT]->(:Location)
(:Event)-[:PARTICIPATED_IN]->(:Character)   # replaces INVOLVED
(:Character)-[:PARTICIPATED_IN]->(:Event)
(:Event)-[:RESULTED_IN]->(:Event)
(:Event)-[:RESULTED_IN]->(:Artifact)
```

All relationships carry a `source_chunk_id` property for traceability.

**Total: 18 relationship types** (expanded from 10).
Predicates removed: `INVOLVED` (superseded by `PARTICIPATED_IN`).
No inverse duplicates — pick the direction that reads most naturally and stay consistent.

### Entity/Relation dict (output of entity_extractor.py)
```python
# Entity
{
    "name":           "Aragorn",
    "type":           "Character",        # Character | Location | Event | Artifact | Faction
    "aliases":        ["Strider", "Elessar"],
    "source_chunk_id": "aragorn_003",
}

# Relation
{
    "subject":        "Aragorn",
    "predicate":      "HEIR_OF",          # must be one of the 18 valid predicates
    "object":         "Isildur",
    "source_chunk_id": "aragorn_003",
}

# More relation examples (showing expanded types)
{"subject": "Frodo",        "predicate": "CHILD_OF",         "object": "Drogo Baggins",              "source_chunk_id": "..."}
{"subject": "Shire",        "predicate": "PART_OF",          "object": "Eriador",                    "source_chunk_id": "..."}
{"subject": "Grima",        "predicate": "SERVANT_OF",       "object": "Saruman",                    "source_chunk_id": "..."}
{"subject": "Aragorn",      "predicate": "RULES_OVER",       "object": "Gondor",                     "source_chunk_id": "..."}
{"subject": "One Ring",     "predicate": "FORGED_BY",        "object": "Sauron",                     "source_chunk_id": "..."}
{"subject": "Aragorn",      "predicate": "PARTICIPATED_IN",  "object": "Battle of Helm's Deep",      "source_chunk_id": "..."}
{"subject": "Battle of Pelennor Fields", "predicate": "RESULTED_IN", "object": "Defeat of Sauron's Army", "source_chunk_id": "..."}
```

---

## Module Conventions

- **Logging**: every module does `log = get_logger(__name__)` at top level.
- **Config**: always imported as `from config import settings`.
- **No globals except lazy singletons**: `_model`, `_client`, `_collection` patterns
  (see embedder.py and chroma_store.py) are acceptable for expensive resources.
- **Function-based modules preferred** over classes where state is minimal
  (embedder.py and chroma_store.py use module-level lazy singletons, not classes).
- **Idempotent runners**: all `run_dayN.py` scripts must be safely re-runnable
  without duplicating data.

---

## Ingestion Pipeline
```
chunks.json (1115 chunks)
  → Branch A: embed chunk → store in ChromaDB (chunk_id as key)          ✅ Day 2
  → Branch B: LLM extract entities+relationships → MERGE into Neo4j       ← Day 3
               with source_chunk_id
```

## Query Pipeline
```
User question
  → extract entities (Gemini)
  → fuzzy match entities to Neo4j node names (RapidFuzz)
  → graph traversal (Cypher, depth ≤ 2, max 20 triples)
  → vector similarity search (ChromaDB, top 5 chunks)
  → serialize triples to sentences: "Aragorn is heir of Isildur."
  → assemble context:
      GRAPH FACTS
      - ...
      TEXT CONTEXT
      [Source: page_title]
      chunk text...
  → Gemini answer generation (context-only, with citations)
```

---

## Day Progress
- Day 1 ✅ COMPLETE — scraping, cleaning, chunking, chunks.json (1115 chunks)
- Day 2 ✅ COMPLETE — embedder.py + chroma_store.py, 1115 chunks in ChromaDB
- Day 3 🔲 TODO — extraction/prompt_templates.py + extraction/entity_extractor.py
                   + graph/neo4j_client.py + graph/deduplicator.py
                   + graph/graph_builder.py + run_day3.py
- Day 4 🔲 TODO — graph/graph_traversal.py + serialization
- Day 5 🔲 TODO — full query pipeline + context assembly
- Day 6 🔲 TODO — LLM answer generation + debug logging
- Day 7 🔲 TODO — evaluation + README