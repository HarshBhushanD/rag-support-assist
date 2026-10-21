from typing import Any, Literal

from src.generation.generator import generate_answer
from src.retrieval.retriever import RetrievalEngine

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class RAGPipeline:
    """Connect document retrieval with grounded answer generation."""

    def __init__(
        self,
        retriever: RetrievalEngine,
        retrieval_mode: RetrievalMode = "hybrid",
        top_k: int = 3,
        max_new_tokens: int = 160,
    ) -> None:
        if retrieval_mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("retrieval_mode must be 'dense', 'bm25', or 'hybrid'")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks for a question."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        if self.retrieval_mode == "dense":
            return self.retriever.search_dense(
                question,
                top_k=self.top_k,
            )

        if self.retrieval_mode == "bm25":
            return self.retriever.search_bm25(
                question,
                top_k=self.top_k,
            )

        return self.retriever.search_hybrid(
            question,
            top_k=self.top_k,
        )

    def ask(self, question: str) -> dict[str, Any]:
        """Retrieve context and generate an answer with source metadata."""
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
