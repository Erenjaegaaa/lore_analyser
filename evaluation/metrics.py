"""
evaluation/metrics.py — Summarises evaluation results from results.json.

Reads the output of eval_runner.py and computes aggregate statistics:
  - Answer rate (how many questions got a non-empty answer)
  - Graph hit rate (how many questions retrieved graph context)
  - Vector hit rate (how many questions retrieved vector chunks)
  - Average latency
  - Breakdown by question type

Usage:
    python -m evaluation.metrics
    python -m evaluation.metrics --results data/evaluation/results.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_RESULTS_FILE = "./data/evaluation/results.json"


def load_results(path: str) -> list[dict]:
    """Load evaluation results from a JSON file."""
    p = Path(path)
    if not p.exists():
        log.error("Results file not found: %s", path)
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run eval_runner.py first to generate results."
        )
    with p.open("r", encoding="utf-8") as f:
        results = json.load(f)
    log.info("Loaded %d results from %s", len(results), path)
    return results


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics from a list of result dicts.

    Returns:
        Dict of metric names to values.
    """
    total = len(results)
    if total == 0:
        return {}

    answered        = sum(1 for r in results if r.get("answer") and not r.get("error"))
    errored         = sum(1 for r in results if r.get("error"))
    graph_hits      = sum(1 for r in results if r.get("graph_count", 0) > 0)
    vector_hits     = sum(1 for r in results if r.get("vector_count", 0) > 0)
    both_hits       = sum(1 for r in results if r.get("graph_count", 0) > 0 and r.get("vector_count", 0) > 0)
    latencies       = [r["latency_ms"] for r in results if r.get("latency_ms")]
    avg_latency     = sum(latencies) / len(latencies) if latencies else 0
    avg_graph       = sum(r.get("graph_count", 0) for r in results) / total
    avg_vector      = sum(r.get("vector_count", 0) for r in results) / total

    # Per-type breakdown
    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "answered": 0, "graph_hits": 0})
    for r in results:
        t = r.get("type", "unknown")
        by_type[t]["total"]      += 1
        by_type[t]["answered"]   += 1 if r.get("answer") and not r.get("error") else 0
        by_type[t]["graph_hits"] += 1 if r.get("graph_count", 0) > 0 else 0

    return {
        "total":            total,
        "answered":         answered,
        "errored":          errored,
        "answer_rate":      answered / total,
        "graph_hit_rate":   graph_hits / total,
        "vector_hit_rate":  vector_hits / total,
        "both_hit_rate":    both_hits / total,
        "avg_latency_ms":   avg_latency,
        "avg_graph_sentences": avg_graph,
        "avg_vector_chunks":   avg_vector,
        "by_type":          dict(by_type),
    }


def print_metrics(metrics: dict, results: list[dict]) -> None:
    """Pretty-print metrics to stdout."""
    print(f"\n{'='*60}")
    print(f"GraphRAG Lore Assistant — Evaluation Summary")
    print(f"{'='*60}")

    print(f"\n── Overall ────────────────────────────────────────────────")
    print(f"Total questions  : {metrics['total']}")
    print(f"Answered         : {metrics['answered']} ({metrics['answer_rate']*100:.0f}%)")
    print(f"Errors           : {metrics['errored']}")

    print(f"\n── Retrieval ──────────────────────────────────────────────")
    print(f"Graph hit rate   : {metrics['graph_hit_rate']*100:.0f}%  "
          f"(avg {metrics['avg_graph_sentences']:.1f} sentences/query)")
    print(f"Vector hit rate  : {metrics['vector_hit_rate']*100:.0f}%  "
          f"(avg {metrics['avg_vector_chunks']:.1f} chunks/query)")
    print(f"Both paths used  : {metrics['both_hit_rate']*100:.0f}%")

    print(f"\n── Latency ────────────────────────────────────────────────")
    print(f"Average latency  : {metrics['avg_latency_ms']:.0f}ms")

    print(f"\n── By Question Type ───────────────────────────────────────")
    for q_type, stats in sorted(metrics["by_type"].items()):
        answer_pct = stats["answered"] / stats["total"] * 100 if stats["total"] else 0
        graph_pct  = stats["graph_hits"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {q_type:<12} : {stats['total']} questions | "
              f"answered={answer_pct:.0f}% | graph_hits={graph_pct:.0f}%")

    print(f"\n── Individual Results ─────────────────────────────────────")
    for r in results:
        graph_icon  = "📊" if r.get("graph_count", 0) > 0 else "  "
        vector_icon = "📄" if r.get("vector_count", 0) > 0 else "  "
        error_icon  = "❌" if r.get("error") else "✅"
        print(f"  {error_icon} {graph_icon}{vector_icon} [{r.get('id','?')}] "
              f"({r.get('type','?')}) {r.get('question','')[:60]}")
        if r.get("error"):
            print(f"       ERROR: {r['error']}")
        else:
            answer_preview = (r.get("answer") or "")[:100]
            print(f"       {answer_preview}{'...' if len(r.get('answer','')) > 100 else ''}")

    print(f"\n{'='*60}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise GraphRAG evaluation results."
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to results JSON file (default: {DEFAULT_RESULTS_FILE})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = load_results(args.results)
    metrics = compute_metrics(results)
    print_metrics(metrics, results)