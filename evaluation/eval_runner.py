"""
evaluation/eval_runner.py — Runs evaluation questions through the query pipeline.

Loads questions from questions.json, runs each through the full GraphRAG
pipeline, and saves results to data/evaluation/results.json.

Usage:
    python -m evaluation.eval_runner
    python -m evaluation.eval_runner --questions data/evaluation/questions.json
"""

import argparse
import json
import time
from pathlib import Path

from pipeline.query_pipeline import query
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_QUESTIONS_FILE = "./data/evaluation/questions.json"
DEFAULT_RESULTS_FILE   = "./data/evaluation/results.json"


def load_questions(path: str) -> list[dict]:
    """Load evaluation questions from a JSON file."""
    p = Path(path)
    if not p.exists():
        log.error("Questions file not found: %s", path)
        raise FileNotFoundError(f"Questions file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d evaluation questions from %s", len(questions), path)
    return questions


def save_results(results: list[dict], path: str) -> None:
    """Save evaluation results to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", path)


def run_evaluation(
    questions_file: str = DEFAULT_QUESTIONS_FILE,
    results_file: str = DEFAULT_RESULTS_FILE,
    delay: float = 3.0,
) -> list[dict]:
    """
    Run all evaluation questions through the pipeline and save results.

    Args:
        questions_file: Path to questions.json
        results_file:   Path to save results.json
        delay:          Seconds to wait between questions (avoid rate limits)

    Returns:
        List of result dicts with answers and metadata.
    """
    questions = load_questions(questions_file)
    results = []
    total = len(questions)

    log.info("Starting evaluation: %d questions", total)
    print(f"\n{'='*60}")
    print(f"GraphRAG Lore Assistant — Evaluation Run")
    print(f"Questions: {total}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions):
        q_id      = q.get("id", f"q{i+1:02d}")
        q_type    = q.get("type", "unknown")
        question  = q.get("question", "")
        notes     = q.get("notes", "")

        print(f"[{i+1}/{total}] {q_id} ({q_type}): {question}")

        if not question:
            log.warning("Skipping empty question at index %d", i)
            continue

        # Run the full pipeline
        start = time.time()
        result = query(question)
        elapsed = time.time() - start

        # Build result record
        record = {
            "id":              q_id,
            "type":            q_type,
            "question":        question,
            "answer":          result.get("answer", ""),
            "graph_sentences": result.get("graph_sentences", []),
            "chunk_ids":       result.get("chunk_ids", []),
            "graph_count":     len(result.get("graph_sentences", [])),
            "vector_count":    len(result.get("chunk_ids", [])),
            "latency_ms":      result.get("latency_ms", elapsed * 1000),
            "error":           result.get("error"),
            "notes":           notes,
        }

        results.append(record)

        # Print summary for this question
        status = "✅" if not result.get("error") else "❌"
        print(f"  {status} graph={record['graph_count']} sentences | "
              f"vector={record['vector_count']} chunks | "
              f"latency={record['latency_ms']:.0f}ms")
        print(f"  Answer: {record['answer'][:120]}{'...' if len(record['answer']) > 120 else ''}\n")

        # Save incrementally so progress isn't lost on crash
        save_results(results, results_file)

        # Delay between questions to avoid Gemini rate limits
        if i < total - 1 and delay > 0:
            time.sleep(delay)

    print(f"{'='*60}")
    print(f"Evaluation complete. Results saved to {results_file}")
    print(f"{'='*60}\n")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation questions through the GraphRAG pipeline."
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS_FILE,
        help=f"Path to questions JSON file (default: {DEFAULT_QUESTIONS_FILE})",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to save results JSON file (default: {DEFAULT_RESULTS_FILE})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between questions (default: 3.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        questions_file=args.questions,
        results_file=args.results,
        delay=args.delay,
    )