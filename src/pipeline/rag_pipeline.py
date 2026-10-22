from typing import Any, Literal

from src.generation.generator import generate_answer
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class RAGPipeline:
    """Connect retrieval, optional reranking, and answer generation."""

    def __init__(
        self,
        retriever: RetrievalEngine,
        retrieval_mode: RetrievalMode = "hybrid",
        top_k: int = 3,
        max_new_tokens: int = 160,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 6,
    ) -> None:
        if retrieval_mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("retrieval_mode must be 'dense', 'bm25', or 'hybrid'")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")

        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve candidates and optionally rerank them."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        retrieval_k = self.candidate_k if self.reranker is not None else self.top_k

        if self.retrieval_mode == "dense":
            chunks = self.retriever.search_dense(
                question,
                top_k=retrieval_k,
            )
        elif self.retrieval_mode == "bm25":
            chunks = self.retriever.search_bm25(
                question,
                top_k=retrieval_k,
            )
        else:
            chunks = self.retriever.search_hybrid(
                question,
                top_k=retrieval_k,
            )

        if self.reranker is not None and chunks:
            return self.reranker.rerank(
                question,
                chunks,
                top_k=self.top_k,
            )

        return chunks[: self.top_k]

    def ask(self, question: str) -> dict[str, Any]:
        """Retrieve context and generate a grounded answer."""
        retrieved_chunks = self.retrieve(question)

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": (
                    "I don't have enough information in the provided "
                    "documents to answer that question."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "retrieval_mode": self.retrieval_mode,
            }

        generation_result = generate_answer(
            question=question,
            context_chunks=retrieved_chunks,
            max_new_tokens=self.max_new_tokens,
        )

        return {
            "question": question,
            "answer": generation_result["answer"],
            "sources": generation_result["sources"],
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": self.retrieval_mode,
        }
