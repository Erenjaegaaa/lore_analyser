"""
extraction/entity_extractor.py — Groq-based entity + relationship extractor.

Calls Groq (Llama) with the prompt from prompt_templates.py, parses the JSON
response, and attaches ``source_chunk_id`` to every entity and relation.

Rate-limit handling: exponential backoff on 429 / rate-limit errors.
On any unrecoverable error the function logs and returns ([], []) so the
pipeline never crashes.
"""

import json
import time

from groq import Groq, RateLimitError, APIStatusError

from config import settings
from extraction.prompt_templates import build_extraction_prompt
from utils.logger import get_logger

log = get_logger(__name__)

# ── Groq client (lazy singleton) ──────────────────────────────────────────────

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.llm.groq_api_key)
        log.debug("Groq client initialised. Model: %s", settings.llm.extraction_model)
    return _client


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """
    Strip any accidental markdown fences and parse JSON.
    Returns the parsed dict, or raises ValueError on failure.
    """
    text = raw.strip()

    # Strip <think>...</think> blocks produced by reasoning models like Qwen3.
    # The model still does full reasoning internally — stripping only removes
    # the chain-of-thought wrapper so the JSON parser sees clean output.
    # Safe for all other models: the condition is False if no <think> block.
    if "<think>" in text:
        end = text.find("</think>")
        if end != -1:
            text = text[end + len("</think>"):].strip()

    # Remove ```json ... ``` or ``` ... ``` fences if the model added them
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    return json.loads(text)


def _validate_structure(data: dict) -> bool:
    """Return True if the parsed dict has the expected top-level keys."""
    return isinstance(data, dict) and "entities" in data and "relations" in data


# ── Public API ────────────────────────────────────────────────────────────────

def extract_from_chunk(
    chunk: dict,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> tuple[list[dict], list[dict]]:
    """
    Extract entities and relations from a single chunk using Groq (Llama).

    Attaches ``source_chunk_id`` (taken from ``chunk["chunk_id"]``) to every
    entity and relation dict before returning.

    Args:
        chunk:       Chunk dict with at least ``chunk_id`` and ``text`` keys.
        max_retries: Maximum number of retry attempts on rate-limit errors.
        base_delay:  Initial back-off delay in seconds (doubles each retry).

    Returns:
        Tuple of (entities, relations) — both are lists of dicts.
        Returns ([], []) on any unrecoverable error.
    """
    chunk_id: str = chunk.get("chunk_id", "<unknown>")
    text: str = chunk.get("text", "").strip()

    if not text:
        log.warning("[%s] Empty text — skipping extraction.", chunk_id)
        return [], []

    prompt = build_extraction_prompt(text)
    client = _get_client()

    attempt = 0
    delay = base_delay

    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model=settings.llm.extraction_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_output_tokens,
            )
            raw = response.choices[0].message.content

        except RateLimitError as exc:
            if attempt < max_retries:
                attempt += 1
                log.warning(
                    "[%s] Rate-limit hit (attempt %d/%d). Sleeping %.1fs.",
                    chunk_id, attempt, max_retries, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            log.error("[%s] Groq rate-limit — exhausted retries: %s", chunk_id, exc)
            return [], []

        except APIStatusError as exc:
            log.error("[%s] Groq API error: %s", chunk_id, exc)
            return [], []

        except Exception as exc:
            log.error("[%s] Unexpected error calling Groq: %s", chunk_id, exc)
            return [], []

        # ── Parse response ────────────────────────────────────────────────
        try:
            data = _parse_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("[%s] JSON parse error: %s | raw=%r", chunk_id, exc, raw[:200])
            return [], []

        if not _validate_structure(data):
            log.error("[%s] Unexpected JSON structure: %r", chunk_id, list(data.keys()))
            return [], []

        # ── Attach source_chunk_id ────────────────────────────────────────
        entities: list[dict] = []
        for ent in data.get("entities", []):
            if not isinstance(ent, dict) or not ent.get("name") or not ent.get("type"):
                log.debug("[%s] Skipping malformed entity: %r", chunk_id, ent)
                continue
            ent["source_chunk_id"] = chunk_id
            ent.setdefault("aliases", [])
            entities.append(ent)

        relations: list[dict] = []
        for rel in data.get("relations", []):
            if not isinstance(rel, dict):
                continue
            if not all(rel.get(k) for k in ("subject", "predicate", "object")):
                log.debug("[%s] Skipping malformed relation: %r", chunk_id, rel)
                continue
            rel["source_chunk_id"] = chunk_id
            relations.append(rel)

        log.debug(
            "[%s] Extracted %d entities, %d relations.",
            chunk_id, len(entities), len(relations),
        )
        return entities, relations

    log.error("[%s] Exhausted all retries — returning empty.", chunk_id)
    return [], []