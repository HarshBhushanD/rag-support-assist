import argparse
import json
import math
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.rag_pipeline import RAGPipeline, normalize_text
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
DEFAULT_OUTPUT = EVAL_DIR / "performance_benchmark.json"
FALLBACK_ANSWER = (
    "I don't have enough information in the provided documents to answer that question."
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def percentile(values: list[float], percent: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def latency_summary(latencies_ms: list[float]) -> dict[str, float | int]:
    """Summarize query latency in milliseconds."""
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def cache_snapshot(pipeline: RAGPipeline) -> dict[str, float]:
    """Return cache keys and expiry times for diagnostic purposes."""
    return {
        key: expires_at for key, (expires_at, _) in pipeline._response_cache.items()
    }


def is_fallback(response: dict[str, Any]) -> bool:
    """Check whether the pipeline returned its standard fallback answer."""
    answer = str(response.get("answer", ""))
    return normalize_text(answer) == normalize_text(FALLBACK_ANSWER)


def time_query(
    pipeline: RAGPipeline,
    question: str,
    phase: str,
    iteration: int,
) -> dict[str, Any]:
    """Measure one query and infer cache status from the pipeline cache."""
    before = cache_snapshot(pipeline)
    key = normalize_text(question)
    cache_hit_before_call = key in before and time.monotonic() < before[key]

    started = time.perf_counter()

    try:
        response = pipeline.ask(question)
    except Exception as exc:
        return {
            "phase": phase,
            "question": question,
            "iteration": iteration,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "success": False,
            "cache_status": "unknown",
            "error": f"{type(exc).__name__}: {exc}",
        }

    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "phase": phase,
        "question": question,
        "iteration": iteration,
        "latency_ms": elapsed_ms,
        "success": True,
        "cache_status": ("hit" if cache_hit_before_call else "miss_or_not_cached"),
        "fallback": is_fallback(response),
        "answerable": None,
    }


def run_benchmark(
    pipeline: RAGPipeline,
    questions: list[dict[str, Any]],
    repeats: int,
) -> dict[str, Any]:
    """Measure cold-cache and warm-cache performance per question."""
    observations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    successful_warmups: set[str] = set()

    for question_item in questions:
        question = question_item["question"]

        for iteration in range(1, repeats + 1):
            pipeline.clear_cache()
            result = time_query(pipeline, question, "cold", iteration)
            result["answerable"] = question_item.get("answerable")
            observations.append(result)

            if not result["success"]:
                errors.append(
                    {
                        "phase": "cold",
                        "question": question,
                        "error": result["error"],
                    }
                )

    pipeline.clear_cache()

    for question_item in questions:
        question = question_item["question"]
        result = time_query(pipeline, question, "warmup", 1)

        if (
            result["success"]
            and not result.get("fallback", False)
            and normalize_text(question) in pipeline._response_cache
        ):
            successful_warmups.add(normalize_text(question))

        if not result["success"]:
            errors.append(
                {
                    "phase": "warmup",
                    "question": question,
                    "error": result["error"],
                }
            )

    for question_item in questions:
        question = question_item["question"]

        for iteration in range(1, repeats + 1):
            result = time_query(pipeline, question, "warm", iteration)
            result["answerable"] = question_item.get("answerable")
            observations.append(result)

            if not result["success"]:
                errors.append(
                    {
                        "phase": "warm",
                        "question": question,
                        "error": result["error"],
                    }
                )

    cold_results = [
        item for item in observations if item["phase"] == "cold" and item["success"]
    ]
    warm_results = [
        item for item in observations if item["phase"] == "warm" and item["success"]
    ]

    cold_latencies = [item["latency_ms"] for item in cold_results]
    warm_latencies = [item["latency_ms"] for item in warm_results]

    warm_hits = sum(item["cache_status"] == "hit" for item in warm_results)
    warm_non_hits = len(warm_results) - warm_hits

    cold_summary = latency_summary(cold_latencies)
    warm_summary = latency_summary(warm_latencies)

    cold_mean = cold_summary["mean_ms"]
    warm_mean = warm_summary["mean_ms"]

    return {
        "questions": len(questions),
        "repeats_per_phase": repeats,
        "expected_timed_queries": len(questions) * repeats * 2,
        "successful_timed_queries": len(cold_results) + len(warm_results),
        "failed_operations": errors,
        "cold_cache": cold_summary,
        "warm_cache": warm_summary,
        "warm_cache_speedup": (
            cold_mean / warm_mean if cold_mean > 0 and warm_mean > 0 else None
        ),
        "warm_phase_cache_hit_observations": warm_hits,
        "warm_phase_cache_miss_or_uncached_observations": warm_non_hits,
        "questions_cached_after_warmup": len(successful_warmups),
        "per_question_observations": observations,
        "note": (
            "Cache status is inferred from the pipeline's private response "
            "cache immediately before each query. Cold runs clear the response "
            "cache, but model weights remain loaded after their first load. "
            "Fallback responses use a separate, shorter cache TTL."
        ),
    }


def estimate_cost(
    queries: int,
    hourly_rate: float,
    mean_latency_ms: float | None,
) -> dict[str, float | int | str | None]:
    """Estimate serial compute cost from a supplied hourly rate."""
    if mean_latency_ms is None:
        compute_hours = None
        cost = None
    else:
        compute_hours = queries * mean_latency_ms / 3_600_000
        cost = compute_hours * hourly_rate

    return {
        "queries": queries,
        "assumed_compute_cost_per_hour": hourly_rate,
        "mean_cold_query_latency_ms": mean_latency_ms,
        "estimated_compute_hours": compute_hours,
        "estimated_cost": cost,
        "currency": "user-supplied; not inferred",
        "note": (
            "Approximation assumes serial queries at the measured mean latency. "
            "A zero hourly rate means unpriced, not necessarily free."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RAG cold-cache and warm-cache latency."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compute-cost-per-hour",
        type=float,
        default=0.0,
        help="Assumed compute cost per hour in your currency.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not math.isfinite(args.compute_cost_per_hour) or args.compute_cost_per_hour < 0:
        parser.error("--compute-cost-per-hour must be finite and non-negative")

    dataset = load_json(EVAL_DIR / "golden_dataset.json")
    chunks = load_json(EVAL_DIR / "evaluation_corpus.json")
    print(f"Questions: {len(dataset)}")
    print(f"Corpus chunks: {len(chunks)}")
    print("Building retrieval engine...")

    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="day19_performance_benchmark",
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        retrieval_mode="hybrid",
        top_k=2,
        max_new_tokens=160,
        cache_maxsize=max(128, len(dataset)),
        cache_ttl_seconds=3600,
    )

    print("Running cold-cache and warm-cache measurements...")
    measurements = run_benchmark(pipeline, dataset, args.repeats)

    cold = measurements["cold_cache"]
    warm = measurements["warm_cache"]

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "retrieval_mode": "hybrid",
            "top_k": 2,
        },
        "dataset": "golden_dataset.json",
        "corpus": "evaluation_corpus.json",
        "measurements": measurements,
        "cost_estimate": estimate_cost(
            queries=1000,
            hourly_rate=args.compute_cost_per_hour,
            mean_latency_ms=cold["mean_ms"],
        ),
        "limitations": [
            "The evaluation dataset has only eight questions.",
            "Cold-cache means the response cache is cleared; model weights "
            "may remain loaded.",
            "Warm-cache measurements represent repeated identical questions.",
            "Fallback responses use a separate, shorter cache TTL.",
            "Cache diagnostics inspect a private pipeline attribute.",
            "The cost estimate is approximate and uses the supplied hourly rate.",
            "Failed operations are excluded from latency summaries.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\n--- Cold-cache latency (ms) ---")
    print(f"Successful queries: {cold['count']}")
    print(f"Mean: {cold['mean_ms']:.2f}")
    print(f"p50: {cold['p50_ms']:.2f}")
    print(f"p95: {cold['p95_ms']:.2f}")

    print("\n--- Warm-cache latency (ms) ---")
    print(f"Successful queries: {warm['count']}")
    print(f"Mean: {warm['mean_ms']:.2f}")
    print(f"p50: {warm['p50_ms']:.2f}")
    print(f"p95: {warm['p95_ms']:.2f}")
    print(
        "Known cache-hit observations: "
        f"{measurements['warm_phase_cache_hit_observations']}"
    )
    print(
        "Known cache-miss/uncached observations: "
        f"{measurements['warm_phase_cache_miss_or_uncached_observations']}"
    )
    print(
        "Questions cached after warmup: "
        f"{measurements['questions_cached_after_warmup']}"
    )

    print("\n--- Cost estimate ---")
    estimated_cost = result["cost_estimate"]["estimated_cost"]

    if estimated_cost is None:
        print("Estimated cost: unavailable")
    else:
        print(f"Estimated cost for 1,000 queries: {estimated_cost:.4f}")
    if args.compute_cost_per_hour == 0:
        print(
            "Note: hourly rate is zero; set --compute-cost-per-hour "
            "for a non-zero illustrative estimate."
        )

    print(f"Results saved to: {args.output}")

    if measurements["failed_operations"]:
        print(
            f"Warning: {len(measurements['failed_operations'])} operations "
            "failed; inspect failed_operations in the report."
        )


if __name__ == "__main__":
    main()
