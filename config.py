"""
config.py — Central configuration for the GraphRAG Lore Assistant.

All tuneable parameters live here. Import this module anywhere in the
project instead of hardcoding values or reading env vars directly.

Usage:
    from config import settings
    print(settings.chunking.chunk_size)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ── Neo4j ─────────────────────────────────────────────────────────────────────

@dataclass
class Neo4jConfig:
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))


# ── LLM ───────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    extraction_model: str = "qwen/qwen3-32b"
    answer_model: str = "gemini-2.5-flash"
    max_output_tokens: int = 8192
    temperature: float = 0.2



# ── Embeddings ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    expected_dimension: int = 384


# ── ChromaDB ──────────────────────────────────────────────────────────────────

@dataclass
class ChromaConfig:
    persist_path: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "lore_chunks"))


# ── Chunking ──────────────────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50


# ── Retrieval ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    vector_top_k: int = 5
    graph_max_triples: int = 20
    graph_traversal_depth: int = 2
    fuzzy_match_threshold: float = 85.0


# ── Data paths ────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    raw_dir: str = "./data/raw"
    cleaned_dir: str = "./data/cleaned"
    chunks_file: str = "./data/chunks/chunks.json"
    evaluation_dir: str = "./data/evaluation"
    logs_dir: str = "./logs"


# ── Logging ───────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "DEBUG"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "./logs/graphrag.log"))


# ── Master settings object ────────────────────────────────────────────────────

@dataclass
class Settings:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Single importable instance
settings = Settings()