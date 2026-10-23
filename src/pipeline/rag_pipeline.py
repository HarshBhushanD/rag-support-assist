import re
from math import isfinite
from typing import Any, Literal

from src.generation.generator import generate_answer
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

RetrievalMode = Literal["dense", "bm25", "hybrid"]

FALLBACK_ANSWER = (
    "I don't have enough information in the provided documents to answer that question."
)

ABSTENTION_PHRASES = (
    "does not provide this information",
    "doesn't provide this information",
    "not enough information",
    "cannot determine the answer",
    "can't determine the answer",
    "unable to determine the answer",
    "not specified in the provided documents",
    "not mentioned in the provided documents",
    "not mentioned in the given references",
    "not provided in the context",
    "does not provide any information",
    "doesn't provide any information",
    "cannot determine if",
    "can't determine if",
    "cannot determine this fact",
    "cannot determine this",
    "cannot be determined from the context",
    "not explicitly stated in the given documents",
    "not explicitly stated in the provided documents",
    "cannot provide a specific",
)

STOPWORDS = frozenset(
    """
    a an the and or but if then to of in on at by for from with without
    is are was were be been being do does did have has had can could
    would should will may might must it its this that these those
    i we you they he she them our your their my me us
    what which who whom where when why how
    please tell show explain give find check verify
    """.split()
)

WORD_PATTERN = re.compile(r"\b[a-z]+(?:['-][a-z]+)*\b|\b\d+(?::\d+)?\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?::\d+)?\b")


def normalize_text(text: str) -> str:
    """Normalize whitespace and casing for comparisons."""
    return " ".join(text.casefold().split())


def is_abstention_response(answer: str) -> bool:
    """Recognize fallback answers and common abstention phrasings."""
    normalized = normalize_text(answer)
    compact_answer = normalized.replace(" ", "")
    compact_fallback = FALLBACK_ANSWER.casefold().replace(" ", "")

    if compact_fallback in compact_answer:
        return True

    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def answer_is_lexically_supported(
    answer: str,
    chunks: list[dict[str, Any]],
    min_token_coverage: float = 0.5,
) -> bool:
    """
    Apply a conservative lexical grounding check.

    This is a heuristic, not a semantic entailment or fact-verification model.
    """
    if not answer or not answer.strip() or not chunks:
        return False

    context = " ".join(
        str(chunk.get("text", ""))
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    )

    if not context.strip():
        return False

    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)

    # Reject numeric details that do not appear in the evidence.
    answer_numbers = NUMBER_PATTERN.findall(normalized_answer)
    context_numbers = set(NUMBER_PATTERN.findall(normalized_context))

    if any(number not in context_numbers for number in answer_numbers):
        return False

    answer_tokens = {
        token
        for token in WORD_PATTERN.findall(normalized_answer)
        if token not in STOPWORDS and not token.isdigit()
    }
    context_tokens = {
        token
        for token in WORD_PATTERN.findall(normalized_context)
        if token not in STOPWORDS and not token.isdigit()
    }

    if not answer_tokens:
        return False

    overlap = answer_tokens & context_tokens
    coverage = len(overlap) / len(answer_tokens)

    return coverage >= min_token_coverage


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
        min_reranker_score: float | None = None,
    ) -> None:
        if retrieval_mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("retrieval_mode must be 'dense', 'bm25', or 'hybrid'")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")
        if min_reranker_score is not None and reranker is None:
            raise ValueError("min_reranker_score requires a reranker")
        if min_reranker_score is not None and not isfinite(min_reranker_score):
            raise ValueError("min_reranker_score must be finite")

        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.min_reranker_score = min_reranker_score

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

    def _fallback_result(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a consistent abstention response."""
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": self.retrieval_mode,
        }

    def ask(self, question: str) -> dict[str, Any]:
        """Retrieve evidence and return an answer only if it passes checks."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        retrieved_chunks = self.retrieve(question)

        if not retrieved_chunks:
            return self._fallback_result(question, [])

        if self.min_reranker_score is not None:
            scores = [chunk.get("reranker_score") for chunk in retrieved_chunks]

            valid_scores = all(
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and isfinite(score)
                for score in scores
            )

            if not valid_scores or max(scores) < self.min_reranker_score:
                return self._fallback_result(
                    question,
                    retrieved_chunks,
                )

        generation_result = generate_answer(
            question=question,
            context_chunks=retrieved_chunks,
            max_new_tokens=self.max_new_tokens,
        )

        answer = generation_result.get("answer", "")

        if is_abstention_response(answer):
            return self._fallback_result(question, retrieved_chunks)

        if not answer_is_lexically_supported(answer, retrieved_chunks):
            return self._fallback_result(question, retrieved_chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": generation_result.get("sources", []),
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": self.retrieval_mode,
        }
