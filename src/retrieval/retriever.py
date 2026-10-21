import re
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi

from src.ingest.vector_store import QdrantVectorStore


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms."""
    return re.findall(r"\b\w+\b", text.lower())


def document_key(document: dict[str, Any]) -> tuple[str, int, str]:
    """Create a stable identity for a document chunk."""
    return (
        str(document.get("source", "")),
        int(document.get("chunk_index", 0)),
        str(document.get("text", "")),
    )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[dict[str, Any]]],
    top_k: int = 3,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Combine ranked lists using Reciprocal Rank Fusion."""
    if top_k < 1 or k < 1:
        raise ValueError("top_k and k must be at least 1")

    fused: dict[tuple[str, int, str], dict[str, Any]] = {}

    for ranking in rankings:
        seen: set[tuple[str, int, str]] = set()

        for rank, document in enumerate(ranking, start=1):
            key = document_key(document)
            if key in seen:
                continue
            seen.add(key)

            if key not in fused:
                fused[key] = {
                    field: value
                    for field, value in document.items()
                    if field
                    not in {
                        "score",
                        "dense_score",
                        "bm25_score",
                        "rrf_score",
                        "retrieval_methods",
                    }
                }
                fused[key]["rrf_score"] = 0.0
                fused[key]["retrieval_methods"] = []

            fused[key]["rrf_score"] += 1.0 / (k + rank)

            method = document.get("retrieval_method")
            if method and method not in fused[key]["retrieval_methods"]:
                fused[key]["retrieval_methods"].append(method)

            if method == "dense":
                fused[key]["dense_score"] = document.get("score")
            elif method == "bm25":
                fused[key]["bm25_score"] = document.get("score")

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]


class RetrievalEngine:
    """Provide dense, BM25, and hybrid retrieval."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        collection_name: str = "support_chunks",
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = [dict(chunk) for chunk in chunks]
        corpus = [tokenize(chunk["text"]) for chunk in self.chunks]

        if any(not tokens for tokens in corpus):
            raise ValueError("Every chunk must contain searchable text")

        self.bm25 = BM25Okapi(corpus)
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
        )
        self.vector_store.add_chunks(self.chunks)

    def search_dense(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search using embedding similarity."""
        if not query.strip() or top_k < 1:
            raise ValueError("Query must be non-empty and top_k at least 1")

        return [
            {
                **result,
                "dense_score": result["score"],
                "retrieval_method": "dense",
            }
            for result in self.vector_store.search(query, top_k=top_k)
        ]

    def search_bm25(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search using BM25 keyword ranking."""
        if not query.strip() or top_k < 1:
            raise ValueError("Query must be non-empty and top_k at least 1")

        terms = tokenize(query)
        if not terms:
            raise ValueError("Query must contain searchable terms")

        scores = self.bm25.get_scores(terms)
        indices = sorted(
            range(len(self.chunks)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        return [
            {
                **self.chunks[index],
                "score": float(scores[index]),
                "bm25_score": float(scores[index]),
                "retrieval_method": "bm25",
            }
            for index in indices
        ]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Combine dense and BM25 results using RRF."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1 or rrf_k < 1:
            raise ValueError("top_k and rrf_k must be at least 1")

        candidate_k = min(len(self.chunks), top_k * 3)
        dense = self.search_dense(query, top_k=candidate_k)
        lexical = self.search_bm25(query, top_k=candidate_k)

        return reciprocal_rank_fusion(
            [dense, lexical],
            top_k=top_k,
            k=rrf_k,
        )
