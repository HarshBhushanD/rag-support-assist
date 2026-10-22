"""CrossEncoder-based reranking for retrieved document chunks."""

from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved chunks by query-document relevance."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: Any | None = None,
    ) -> None:
        """Initialize the CrossEncoder, allowing model injection in tests."""
        self.model = model or CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the top-k candidates ordered by CrossEncoder relevance."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not candidates:
            return []

        if any(
            not isinstance(chunk.get("text"), str) or not chunk["text"].strip()
            for chunk in candidates
        ):
            raise ValueError("Every candidate must contain non-empty text")
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = self.model.predict(pairs)

        reranked = []

        for chunk, score in zip(candidates, scores, strict=True):
            result = chunk.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda chunk: chunk["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]
