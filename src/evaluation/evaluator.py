from typing import Any, Literal

from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

RetrievalMode = Literal["dense", "bm25", "hybrid"]


def evaluate_retrieval(
    retriever: RetrievalEngine,
    dataset: list[dict[str, Any]],
    top_k: int = 2,
    reranker: CrossEncoderReranker | None = None,
    candidate_k: int = 6,
) -> dict[str, Any]:
    """Evaluate retrieval against expected source filenames."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if reranker is not None and candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    search_methods = {
        "dense": retriever.search_dense,
        "bm25": retriever.search_bm25,
        "hybrid": retriever.search_hybrid,
    }

    if reranker is not None:

        def search_reranked_hybrid(
            question: str,
            top_k: int,
        ) -> list[dict[str, Any]]:
            candidates = retriever.search_hybrid(
                question,
                top_k=candidate_k,
            )
            return reranker.rerank(
                question,
                candidates,
                top_k=top_k,
            )

        search_methods["reranked_hybrid"] = search_reranked_hybrid

    results: dict[str, Any] = {}

    for mode, search in search_methods.items():
        query_results: list[dict[str, Any]] = []

        for item in dataset:
            expected_sources = set(item.get("expected_sources", []))
            retrieved = search(item["question"], top_k=top_k)
            retrieved_sources = [chunk.get("source", "unknown") for chunk in retrieved]
            retrieved_source_set = set(retrieved_sources)

            # Respect explicit answerability labels in the evaluation dataset.
            is_answerable = item.get("answerable", bool(expected_sources))

            if not is_answerable:
                query_results.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "expected_sources": sorted(expected_sources),
                        "retrieved_sources": retrieved_sources,
                        "is_unanswerable": True,
                        "returned_results": bool(retrieved),
                        "empty_retrieval": not retrieved,
                    }
                )
                continue

            if not expected_sources:
                raise ValueError(
                    f"Answerable query {item['id']} must have expected_sources"
                )

            relevant_ranks = [
                rank
                for rank, source in enumerate(retrieved_sources, start=1)
                if source in expected_sources
            ]

            matched_sources = retrieved_source_set & expected_sources
            hit = bool(matched_sources)
            recall = len(matched_sources) / len(expected_sources)
            reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

            query_results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_sources": sorted(expected_sources),
                    "retrieved_sources": retrieved_sources,
                    "is_unanswerable": False,
                    "hit": hit,
                    "all_sources_found": expected_sources.issubset(
                        retrieved_source_set
                    ),
                    "recall": recall,
                    "reciprocal_rank": reciprocal_rank,
                }
            )

        answerable_results = [
            result for result in query_results if not result["is_unanswerable"]
        ]
        unanswerable_results = [
            result for result in query_results if result["is_unanswerable"]
        ]

        answerable_count = len(answerable_results)
        unanswerable_count = len(unanswerable_results)

        results[mode] = {
            "top_k": top_k,
            "queries_evaluated": len(query_results),
            "answerable_queries": answerable_count,
            "unanswerable_queries": unanswerable_count,
            "hit_rate_at_k": (
                sum(result["hit"] for result in answerable_results) / answerable_count
                if answerable_count
                else 0.0
            ),
            "recall_at_k": (
                sum(result["recall"] for result in answerable_results)
                / answerable_count
                if answerable_count
                else 0.0
            ),
            "mrr": (
                sum(result["reciprocal_rank"] for result in answerable_results)
                / answerable_count
                if answerable_count
                else 0.0
            ),
            "all_sources_recall_at_k": (
                sum(result["all_sources_found"] for result in answerable_results)
                / answerable_count
                if answerable_count
                else 0.0
            ),
            "unanswerable_empty_retrieval_rate": (
                sum(result["empty_retrieval"] for result in unanswerable_results)
                / unanswerable_count
                if unanswerable_count
                else 0.0
            ),
            "unanswerable_returned_results_rate": (
                sum(result["returned_results"] for result in unanswerable_results)
                / unanswerable_count
                if unanswerable_count
                else 0.0
            ),
            "details": query_results,
        }

    return results
