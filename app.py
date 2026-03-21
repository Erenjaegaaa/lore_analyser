
"""
app.py — FastAPI backend for the GraphRAG Lore Assistant.

Exposes a single POST /query endpoint that takes a question and returns
the answer along with debug metadata.

Run locally:
    uvicorn app:app --reload --port 8000

Environment variables required (same as main pipeline):
    GEMINI_API_KEY, GROQ_API_KEY
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
    CHROMA_PERSIST_PATH, CHROMA_COLLECTION_NAME
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.query_pipeline import query
from utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="GraphRAG Lore Assistant",
    description="Hybrid GraphRAG question-answering system over Tolkien/Middle-earth lore.",
    version="1.0.0",
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    vector_top_k: int | None = None


class QueryResponse(BaseModel):
    question:        str
    answer:          str
    graph_sentences: list[str]
    chunk_ids:       list[str]
    graph_count:     int
    vector_count:    int
    latency_ms:      float
    error:           str | None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "GraphRAG Lore Assistant is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Run a question through the full GraphRAG pipeline and return the answer.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    log.info("Query received: %r", request.question[:80])

    result = query(
        question=request.question,
        vector_top_k=request.vector_top_k,
    )

    return QueryResponse(
        question        = result["question"],
        answer          = result["answer"],
        graph_sentences = result["graph_sentences"],
        chunk_ids       = result["chunk_ids"],
        graph_count     = len(result["graph_sentences"]),
        vector_count    = len(result["chunk_ids"]),
        latency_ms      = result["latency_ms"],
        error           = result.get("error"),
    )