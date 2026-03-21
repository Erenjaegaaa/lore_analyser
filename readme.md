# 🧙 GraphRAG Lore Assistant

A **Hybrid GraphRAG** question-answering system over Tolkien's Middle-earth lore. Combines a **Neo4j knowledge graph** with **ChromaDB vector search** for retrieval, then uses **Gemini** to generate grounded, cited answers.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│           Query Pipeline                │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Graph        │  │ Vector          │  │
│  │ Retriever    │  │ Retriever       │  │
│  │              │  │                 │  │
│  │ question     │  │ question        │  │
│  │ → extract    │  │ → embed         │  │
│  │   entities   │  │ → ChromaDB      │  │
│  │ → fuzzy match│  │   similarity    │  │
│  │ → Neo4j      │  │   search        │  │
│  │   traversal  │  │ → top-5 chunks  │  │
│  │ → triples    │  │                 │  │
│  └──────────────┘  └─────────────────┘  │
│          │                  │           │
│          └────────┬─────────┘           │
│                   ▼                     │
│          Context Assembler              │
│          GRAPH FACTS + TEXT CONTEXT     │
│                   │                     │
│                   ▼                     │
│          Gemini Answer Generation       │
└─────────────────────────────────────────┘
      │
      ▼
  Cited Answer
```

### Why Hybrid?
- **Graph path** — captures explicit relationships (kinship, allegiance, geography) that pure semantic search misses
- **Vector path** — handles semantic similarity, typos, paraphrasing, and questions without named entities
- **Always both** — no query classifier, both paths run unconditionally on every question

---

## 📊 System Performance

Evaluated on 10 typed questions covering factual, relationship, event, location, artifact, and thematic queries:

| Metric | Result |
|---|---|
| Answer rate | 80% (2 failures due to API quota, not pipeline) |
| Graph hit rate | 100% |
| Vector hit rate | 100% |
| Both paths used | 100% |
| Average latency | ~7 seconds |

---

## 🗂️ Dataset

- **Source**: English Wikipedia (71 pages)
- **Coverage**: Characters, locations, events, artifacts, factions across LotR, The Hobbit, The Silmarillion
- **Chunks**: 1,115 overlapping text chunks (500 tokens, 100 overlap)
- **Graph**: ~3,130 nodes, 18 relationship types, 651 chunks extracted

### Neo4j Schema

**Node labels**: `Character`, `Location`, `Event`, `Artifact`, `Faction`

**Relationship types** (18 total):

| Category | Predicates |
|---|---|
| Kinship | `CHILD_OF`, `SIBLING_OF`, `SPOUSE_OF`, `HEIR_OF` |
| Alliance & Enmity | `ALLY_OF`, `ENEMY_OF`, `SERVANT_OF` |
| Faction & Politics | `MEMBER_OF`, `RULES_OVER` |
| Craftsmanship | `CREATED`, `FORGED_BY`, `WIELDED` |
| Geography | `BORN_IN`, `PART_OF`, `LOCATED_IN` |
| Events | `OCCURRED_AT`, `PARTICIPATED_IN`, `RESULTED_IN` |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Knowledge Graph | Neo4j (AuraDB Free) |
| Vector DB | ChromaDB (persistent) |
| Embeddings | `all-MiniLM-L6-v2` (384 dims) |
| Extraction LLM | Groq (Llama 3.1 8B / Qwen3 32B) |
| Answer LLM | Google Gemini 2.5 Flash |
| Fuzzy Matching | RapidFuzz (threshold 85.0) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Scraping | httpx + BeautifulSoup4 |
| Chunking | LangChain RecursiveCharacterTextSplitter |

---

## 📁 Project Structure

```
graphrag-lore-assistant/
│
├── app.py                    ← FastAPI backend
├── streamlit_app.py          ← Streamlit chat UI
├── main.py                   ← CLI entry point
├── config.py                 ← All settings via dataclasses + .env
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── scraper.py            ← Wikipedia scraper
│   └── document_loader.py    ← Save/load raw pages + chunks
│
├── chunking/
│   ├── text_cleaner.py       ← HTML boilerplate stripper
│   └── chunker.py            ← Overlapping token chunker
│
├── embeddings/
│   ├── embedder.py           ← sentence-transformers wrapper
│   └── chroma_store.py       ← ChromaDB persistent collection
│
├── extraction/
│   ├── prompt_templates.py   ← LLM extraction prompts
│   └── entity_extractor.py   ← Groq entity/relation extractor
│
├── graph/
│   ├── neo4j_client.py       ← Neo4j driver wrapper
│   ├── deduplicator.py       ← RapidFuzz canonical name registry
│   ├── graph_builder.py      ← MERGE entities/relations into Neo4j
│   └── graph_traversal.py    ← Cypher neighborhood/path queries
│
├── retrieval/
│   ├── vector_retriever.py   ← ChromaDB similarity search
│   ├── graph_retriever.py    ← Entity extraction + fuzzy match + traversal
│   └── context_assembler.py  ← Merges graph + vector context
│
├── pipeline/
│   ├── ingestion_pipeline.py ← Full ingestion orchestration
│   └── query_pipeline.py     ← Full query orchestration + Gemini
│
├── evaluation/
│   ├── eval_runner.py        ← Runs questions.json through pipeline
│   ├── metrics.py            ← Computes evaluation summary
│   └── questions.json        ← 10 typed evaluation questions (auto-generated)
│       └── results.json      ← Evaluation results
│
├── utils/
│   ├── logger.py             ← Structured logging
│   └── helpers.py            ← Shared utilities
│
└── data/
    ├── raw/                  ← Scraped Wikipedia pages (JSON)
    ├── chunks/
    │   └── chunks.json       ← 1,115 text chunks
    ├── chroma_db/            ← ChromaDB vector index
    └── evaluation/
        └── questions.json    ← Evaluation questions
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Neo4j AuraDB Free account ([console.neo4j.io](https://console.neo4j.io))
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))
- Groq API key ([console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://github.com/Erenjaegaaa/lore_analyser
cd lore_analyser
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
CHROMA_PERSIST_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=lore_chunks
LOG_LEVEL=INFO
```

---

## 💻 Usage

### Interactive CLI
```bash
python main.py interactive
```

### Single query
```bash
python main.py query "Who forged the One Ring?"
```

### Streamlit UI
```bash
streamlit run streamlit_app.py
```

### FastAPI backend
```bash
uvicorn app:app --reload --port 8000
```

### Run evaluation
```bash
python -m evaluation.eval_runner
python -m evaluation.metrics
```

---

## 🔄 Ingestion Pipeline

If you want to rebuild the knowledge graph from scratch:

```bash
# Day 1: Scrape Wikipedia and chunk
python run_day1.py

# Day 2: Embed chunks into ChromaDB
python run_day2.py

# Day 3: Extract entities and build Neo4j graph
# Process in batches due to API rate limits
python run_day3.py --delay 2 --start 0 --limit 250

# Resume from where you left off
python run_day3.py --delay 2 --start 250
```

---

## 🌐 Deployment

### Backend (Railway)
1. Push to GitHub
2. Connect Railway to your repo
3. Set environment variables in Railway dashboard
4. Deploy — Railway auto-detects FastAPI

### Frontend (Hugging Face Spaces)
1. Create a new Space (Streamlit template)
2. Push `streamlit_app.py` and `requirements.txt`
3. Set `BACKEND_URL` secret to your Railway URL
4. Deploy

### Database
- Neo4j: Already on AuraDB Free (cloud-hosted)
- ChromaDB: Copy `data/chroma_db/` folder to your deployment server

---

## 📝 Design Decisions

**Why hybrid retrieval?** Pure vector search misses explicit relationships ("Who is Aragorn's father?"). Pure graph search misses semantic queries ("Tell me about the corruption of power in Tolkien"). Hybrid always runs both.

**Why no query classifier?** Classifiers add latency and failure modes. Running both paths unconditionally is simpler, more robust, and the cost is negligible at this scale.

**Why Wikipedia over fan wikis?** Tolkien Gateway and lotr.fandom.com use Cloudflare protection that blocks scrapers. Wikipedia has the same MediaWiki structure and comprehensive coverage.

**Why Groq for extraction?** Gemini free tier is 20 requests/day — too slow for 1,115 chunks. Groq offers 500K tokens/day free across multiple models, making full-dataset extraction feasible without cost.

**Why RapidFuzz deduplication?** LLMs extract the same entity with slight variations ("Aragorn", "Aragorn son of Arathorn", "Strider"). Fuzzy matching at 85% threshold collapses these to a canonical form before Neo4j writes.

---

## ⚠️ Known Limitations

- **Graph noise**: LLM extraction introduces ~10-15% incorrect relations. The vector path compensates for this in most cases.
- **Latency**: ~7-16 seconds per query due to Gemini API + cloud Neo4j round trips.
- **Coverage**: 651/1,115 chunks extracted into the graph (58%). Remaining chunks are accessible via vector search only.
- **Real-world noise**: Wikipedia adaptation/reception sections introduce non-lore entities (directors, actors, scholars) into the graph.

---

## 🔮 Future Improvements

- Complete graph extraction for all 1,115 chunks
- Add relation frequency filtering to reduce graph noise
- Implement Gemini streaming for lower perceived latency
- Add conversation memory for multi-turn Q&A
- Expand dataset to include Unfinished Tales, History of Middle-earth
- Fine-tune entity extraction prompt for better predicate adherence

---

## 👤 Author

Built as a research project exploring hybrid GraphRAG architectures for domain-specific knowledge bases.

GitHub: [Erenjaegaaa/lore_analyser](https://github.com/Erenjaegaaa/lore_analyser)