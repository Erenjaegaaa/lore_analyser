#
``
######## ##     ## ########    ##        #######  ########  ######## 
   ##    ##     ## ##          ##       ##     ## ##     ## ##       
   ##    ##     ## ##          ##       ##     ## ##     ## ##       
   ##    ######### ######      ##       ##     ## ########  ######   
   ##    ##     ## ##          ##       ##     ## ##   ##   ##       
   ##    ##     ## ##          ##       ##     ## ##    ##  ##       
   ##    ##     ## ########    ########  #######  ##     ## ######## 

##    ## ######## ######## ########  ######## ########  
##   ##  ##       ##       ##     ## ##       ##     ## 
##  ##   ##       ##       ##     ## ##       ##     ## 
#####    ######   ######   ########  ######   ########  
##  ##   ##       ##       ##        ##       ##   ##   
##   ##  ##       ##       ##        ##       ##    ##  
##    ## ######## ######## ##        ######## ##     ## 
```

> *Ask anything of the archives of Middle-earth.*

A **Hybrid GraphRAG** question-answering system built over Tolkien's Middle-earth lore. Combines a **Neo4j knowledge graph** with **ChromaDB vector search** for retrieval, then uses **Gemini** to generate grounded, cited answers in natural language.

**🌐 Live Demo → [lorekeeper-ochre.vercel.app](https://lorekeeper-ochre.vercel.app)**

---

## ✨ What it does

Ask any question about Tolkien's legendarium in plain English:

```
"Who forged the One Ring?"
"What is the relationship between Aragorn and Isildur?"
"What happened at the Battle of Helm's Deep?"
"How are Frodo and Bilbo related?"
"What are the Silmarils and why do they matter?"
```

The system retrieves facts from both a structured knowledge graph and semantic vector search, then synthesizes a cited, accurate answer using Gemini.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────┐
│              Query Pipeline                  │
│                                             │
│  ┌───────────────┐    ┌──────────────────┐  │
│  │ Graph         │    │ Vector           │  │
│  │ Retriever     │    │ Retriever        │  │
│  │               │    │                  │  │
│  │ → extract     │    │ → embed query    │  │
│  │   entities    │    │ → ChromaDB       │  │
│  │ → fuzzy match │    │   cosine search  │  │
│  │ → Neo4j       │    │ → top-5 chunks   │  │
│  │   traversal   │    │                  │  │
│  │ → triples     │    │                  │  │
│  └───────────────┘    └──────────────────┘  │
│          │                    │              │
│          └──────────┬─────────┘              │
│                     ▼                        │
│            Context Assembler                 │
│       GRAPH FACTS + TEXT CONTEXT             │
│                     │                        │
│                     ▼                        │
│          Gemini Answer Generation            │
└─────────────────────────────────────────────┘
      │
      ▼
  Cited Answer
```

### Why Hybrid?

| Retrieval Type | Strength |
|---|---|
| **Graph path** | Explicit relationships — kinship, allegiance, geography, events |
| **Vector path** | Semantic similarity — handles typos, paraphrasing, vague questions |
| **Always both** | No query classifier — both paths run unconditionally on every question |

---

## 📊 Evaluation

Tested on 10 typed questions covering factual, relationship, event, location, artifact, character, and thematic queries:

| Metric | Result |
|---|---|
| **Answer rate** | 100% (10/10) |
| **Graph hit rate** | 100% |
| **Vector hit rate** | 100% |
| **Both paths used** | 100% |
| **Average latency** | ~7 seconds |

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Source | English Wikipedia (71 pages) |
| Coverage | LotR, The Hobbit, The Silmarillion, Unfinished Tales |
| Chunks | 1,115 overlapping text passages (500 tokens, 100 overlap) |
| Vectors | 1,115 embeddings (all-MiniLM-L6-v2, 384 dims) |
| Graph nodes | ~3,130 |
| Graph edges | 18 relationship types |
| Chunks extracted | 651 / 1,115 |

### Neo4j Knowledge Graph Schema

**Node labels:** `Character` · `Location` · `Event` · `Artifact` · `Faction`

**Relationship types:**

| Category | Predicates |
|---|---|
| Kinship | `CHILD_OF` · `SIBLING_OF` · `SPOUSE_OF` · `HEIR_OF` |
| Alliance & Enmity | `ALLY_OF` · `ENEMY_OF` · `SERVANT_OF` |
| Faction & Politics | `MEMBER_OF` · `RULES_OVER` |
| Craftsmanship | `CREATED` · `FORGED_BY` · `WIELDED` |
| Geography | `BORN_IN` · `PART_OF` · `LOCATED_IN` |
| Events | `OCCURRED_AT` · `PARTICIPATED_IN` · `RESULTED_IN` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Knowledge Graph** | Neo4j AuraDB Free |
| **Vector DB** | ChromaDB (persistent) |
| **Embeddings** | `all-MiniLM-L6-v2` (384 dims) |
| **Extraction LLM** | Groq — Llama 3.1 8B / Qwen3 32B |
| **Answer LLM** | Google Gemini 2.5 Flash |
| **Fuzzy Matching** | RapidFuzz (threshold 85.0) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Next.js + Tailwind CSS |
| **Backend Hosting** | Railway |
| **Frontend Hosting** | Vercel |
| **Scraping** | httpx + BeautifulSoup4 |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Language** | Python 3.11+ / TypeScript |

---

## 📁 Repository Structure

```
lore_analyser/                         ← Backend (Python)
│
├── app.py                             ← FastAPI REST API
├── main.py                            ← CLI entry point
├── config.py                          ← Centralised settings
├── requirements.txt
│
├── ingestion/
│   ├── scraper.py                     ← Wikipedia scraper
│   └── document_loader.py
│
├── chunking/
│   ├── text_cleaner.py
│   └── chunker.py
│
├── embeddings/
│   ├── embedder.py                    ← sentence-transformers wrapper
│   └── chroma_store.py                ← ChromaDB interface
│
├── extraction/
│   ├── prompt_templates.py            ← LLM extraction prompts
│   └── entity_extractor.py           ← Groq entity/relation extractor
│
├── graph/
│   ├── neo4j_client.py               ← Neo4j driver wrapper
│   ├── deduplicator.py               ← RapidFuzz name canonicalisation
│   ├── graph_builder.py              ← MERGE nodes/edges into Neo4j
│   └── graph_traversal.py            ← Cypher neighborhood queries
│
├── retrieval/
│   ├── vector_retriever.py           ← ChromaDB similarity search
│   ├── graph_retriever.py            ← Entity extraction + fuzzy match
│   └── context_assembler.py          ← Merges graph + vector context
│
├── pipeline/
│   ├── ingestion_pipeline.py
│   └── query_pipeline.py             ← Full query orchestration
│
├── evaluation/
│   ├── eval_runner.py
│   ├── metrics.py
│   └── questions.json                ← 10 typed eval questions
│
└── data/
    ├── chunks/chunks.json            ← 1,115 text chunks
    ├── chroma_db/                    ← ChromaDB vector index
    └── evaluation/results.json      ← Evaluation results
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Neo4j AuraDB Free — [console.neo4j.io](https://console.neo4j.io)
- Google Gemini API key — [aistudio.google.com](https://aistudio.google.com)
- Groq API key — [console.groq.com](https://console.groq.com)

### Backend

```bash
git clone https://github.com/Erenjaegaaa/lore_analyser
cd lore_analyser
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env`:
```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
CHROMA_PERSIST_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=lore_chunks
LOG_LEVEL=INFO
```

Run:
```bash
uvicorn app:app --reload --port 8000
```

### Frontend

```bash
git clone https://github.com/Erenjaegaaa/LORE_KEEPER_FRONTEND
cd LORE_KEEPER_FRONTEND
npm install
```

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Run:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 💻 CLI Usage

```bash
# Single query
python main.py query "Who forged the One Ring?"

# Interactive session
python main.py interactive
```

---

## 🔄 Rebuilding the Knowledge Graph

```bash
# Day 1 — scrape and chunk
python run_day1.py

# Day 2 — embed into ChromaDB
python run_day2.py

# Day 3 — extract entities and build Neo4j graph
python run_day3.py --delay 2 --start 0 --limit 250

# Resume from a checkpoint
python run_day3.py --delay 2 --start 250
```

---

## 📋 Running Evaluation

```bash
python -m evaluation.eval_runner
python -m evaluation.metrics
```

---

## 🌐 Deployment

| Component | Platform | URL |
|---|---|---|
| Frontend | Vercel | [lorekeeper-ochre.vercel.app](https://lorekeeper-ochre.vercel.app) |
| Backend | Railway | https://web-production-4955d.up.railway.app |
| Graph DB | Neo4j AuraDB Free | Cloud hosted |
| Vector DB | ChromaDB | Bundled with backend |

---

## 🎨 Frontend

Built with Next.js, designed with a dark fantasy aesthetic. Features:

- **Cinematic hero** — AI-generated video of a wizard reading scrolls, with scroll-driven crossfade into the chat interface
- **Dark parchment theme** — gold accents, Cinzel and Cormorant Garamond fonts
- **Chat interface** — message history, animated loading states, source citation tags
- **Sidebar** — example questions, about section, debug info toggle
- **Debug mode** — shows graph sentences used, vector chunks retrieved, and latency per query

---

## 🧠 Design Decisions

**Why hybrid retrieval?**
Pure vector search misses explicit relationships ("Who is Aragorn's father?"). Pure graph search misses semantic queries ("Tell me about the corruption of power in Tolkien"). Running both unconditionally gives the best of both worlds with no routing complexity or failure modes.

**Why Wikipedia over fan wikis?**
Tolkien Gateway and lotr.fandom.com use Cloudflare protection that blocks scrapers. Wikipedia has comprehensive, well-structured coverage with the same MediaWiki HTML structure.

**Why Groq for extraction?**
Gemini free tier allows only 20 requests/day — far too slow for 1,115 chunks. Groq provides 500K tokens/day free across multiple models, making full-dataset extraction feasible at zero cost.

**Why RapidFuzz deduplication?**
LLMs extract the same entity with slight name variations across chunks. Fuzzy matching at 85% threshold collapses variants like "Aragorn", "Aragorn son of Arathorn", and "Strider" to a single canonical node before writing to Neo4j.

**Why no query classifier?**
Classifiers add latency and failure modes. Both retrieval paths are cheap enough to run unconditionally, and Gemini synthesizes whichever context is most relevant automatically.

---

## ⚠️ Known Limitations

- **Graph coverage** — 651/1,115 chunks extracted into the graph (58%). Remaining chunks are accessible via vector search only.
- **Graph noise** — LLM extraction introduces ~10–15% incorrect relations. The vector path and Gemini's grounding instruction compensate in most cases.
- **Latency** — ~7–16 seconds per query due to Gemini API + cloud Neo4j round trips.
- **Real-world noise** — Wikipedia's adaptation/reception sections introduce non-lore entities (directors, actors, scholars) into the graph.

---

## 🔮 Future Work

- Complete graph extraction for all 1,115 chunks
- Add relation frequency filtering to reduce graph noise
- Implement Gemini streaming for lower perceived latency
- Add conversation memory for multi-turn Q&A
- Expand dataset to Unfinished Tales, The History of Middle-earth series
- Fine-tune a small model on the extraction task for better predicate adherence

---

## 👤 Author

**Sahi** · Computer Science Student
GitHub: [@Erenjaegaaa](https://github.com/Erenjaegaaa)

---

*"Not all those who wander are lost." — J.R.R. Tolkien*