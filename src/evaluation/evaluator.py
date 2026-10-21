from typing import Any, Literal

from src.retrieval.retriever import RetrievalEngine

RetrievalMode = Literal["dense", "bm25", "hybrid"]


def evaluate_retrieval(
    retriever: RetrievalEngine,
    dataset: list[dict[str, Any]],
    top_k: int = 2,
) -> dict[str, Any]:
    """Evaluate retrieval using expected source filenames."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    search_methods = {
        "dense": retriever.search_dense,
        "bm25": retriever.search_bm25,
        "hybrid": retriever.search_hybrid,
    }

    results: dict[str, Any] = {}

    for mode, search in search_methods.items():
        query_results = []

        for item in dataset:
            expected_sources = set(item.get("expected_sources", []))

            # Questions without known supporting sources cannot be
            # scored with source-based retrieval metrics.
            if not expected_sources:
                continue

            retrieved = search(item["question"], top_k=top_k)
            retrieved_sources = [chunk.get("source", "unknown") for chunk in retrieved]

            relevant_ranks = [
                rank
                for rank, source in enumerate(retrieved_sources, start=1)
                if source in expected_sources
            ]

            hit = bool(relevant_ranks)
            recall = len(set(retrieved_sources) & expected_sources) / len(
                expected_sources
            )
            reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

            query_results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_sources": sorted(expected_sources),
                    "retrieved_sources": retrieved_sources,
                    "hit": hit,
                    "recall": recall,
                    "reciprocal_rank": reciprocal_rank,
                }
            )

        count = len(query_results)

        results[mode] = {
            "top_k": top_k,
            "queries_evaluated": count,
            "hit_rate_at_k": (
                sum(result["hit"] for result in query_results) / count if count else 0.0
            ),
            "recall_at_k": (
                sum(result["recall"] for result in query_results) / count
                if count
                else 0.0
            ),
            "mrr": (
                sum(result["reciprocal_rank"] for result in query_results) / count
                if count
                else 0.0
            ),
            "details": query_results,
        }

    return results
