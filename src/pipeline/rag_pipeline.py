import re
from collections import OrderedDict
from copy import deepcopy
from math import isfinite
from time import monotonic
from typing import Any, Literal

from src.generation.generator import generate_answer
from src.ingest.embedder import embed_texts
from src.pipeline.retry import run_with_retries
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
    """Apply a conservative lexical grounding heuristic.
    This is not a semantic entailment or fact-verification model.
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
    """Connect retrieval, optional reranking, answer generation, and caching."""

    def __init__(
        self,
        retriever: RetrievalEngine,
        retrieval_mode: RetrievalMode = "hybrid",
        top_k: int = 3,
        max_new_tokens: int = 160,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 6,
        min_reranker_score: float | None = None,
        cache_maxsize: int = 128,
        cache_ttl_seconds: float = 300.0,
        fallback_cache_ttl_seconds: float = 30.0,
        semantic_cache_enabled: bool = False,
        semantic_cache_similarity_threshold: float = 0.92,
        semantic_cache_ttl_seconds: float = 300.0,
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
        if cache_maxsize < 0:
            raise ValueError("cache_maxsize must be non-negative")
        if not isfinite(cache_ttl_seconds) or cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be finite and non-negative")
        if (
            not isinstance(fallback_cache_ttl_seconds, (int, float))
            or isinstance(fallback_cache_ttl_seconds, bool)
            or not isfinite(fallback_cache_ttl_seconds)
            or fallback_cache_ttl_seconds < 0
        ):
            raise ValueError(
                "fallback_cache_ttl_seconds must be finite and non-negative"
            )
        if not isinstance(semantic_cache_enabled, bool):
            raise ValueError("semantic_cache_enabled must be a boolean")
        if (
            not isinstance(semantic_cache_similarity_threshold, (int, float))
            or isinstance(semantic_cache_similarity_threshold, bool)
            or not isfinite(semantic_cache_similarity_threshold)
            or not 0.0 <= semantic_cache_similarity_threshold <= 1.0
        ):
            raise ValueError(
                "semantic_cache_similarity_threshold must be between 0 and 1"
            )
        if (
            not isinstance(semantic_cache_ttl_seconds, (int, float))
            or isinstance(semantic_cache_ttl_seconds, bool)
            or not isfinite(semantic_cache_ttl_seconds)
            or semantic_cache_ttl_seconds < 0
        ):
            raise ValueError(
                "semantic_cache_ttl_seconds must be finite and non-negative"
            )
        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.min_reranker_score = min_reranker_score
        self.cache_maxsize = cache_maxsize
        self.cache_ttl_seconds = cache_ttl_seconds
        self.fallback_cache_ttl_seconds = fallback_cache_ttl_seconds
        self.semantic_cache_enabled = semantic_cache_enabled
        self.semantic_cache_similarity_threshold = float(
            semantic_cache_similarity_threshold
        )
        self.semantic_cache_ttl_seconds = float(semantic_cache_ttl_seconds)
        self._response_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = (
            OrderedDict()
        )
        self._semantic_cache: OrderedDict[
            str, tuple[float, list[float], dict[str, Any]]
        ] = OrderedDict()

    def clear_cache(self) -> None:
        """Remove all exact-match and semantic cached responses."""
        self._response_cache.clear()
        self._semantic_cache.clear()

    def _get_cached_response(self, question: str) -> dict[str, Any] | None:
        """Return an unexpired exact-match cached response, if available."""
        if self.cache_maxsize == 0:
            return None
        key = normalize_text(question)
        entry = self._response_cache.get(key)
        if entry is None:
            return None
        expires_at, response = entry
        if monotonic() >= expires_at:
            del self._response_cache[key]
            return None
        self._response_cache.move_to_end(key)
        cached_response = deepcopy(response)
        cached_response["question"] = question
        return cached_response

    def _cache_response(
        self,
        question: str,
        response: dict[str, Any],
        ttl_seconds: float | None = None,
    ) -> None:
        """Cache a response with its own TTL and LRU eviction."""
        if ttl_seconds is None:
            ttl_seconds = self.cache_ttl_seconds
        if self.cache_maxsize == 0 or ttl_seconds == 0:
            return
        key = normalize_text(question)
        self._response_cache[key] = (monotonic() + ttl_seconds, deepcopy(response))
        self._response_cache.move_to_end(key)
        while len(self._response_cache) > self.cache_maxsize:
            self._response_cache.popitem(last=False)

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve candidates with retries, then optionally rerank them."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        retrieval_k = self.candidate_k if self.reranker is not None else self.top_k

        def search() -> list[dict[str, Any]]:
            if self.retrieval_mode == "dense":
                return self.retriever.search_dense(question, top_k=retrieval_k)
            if self.retrieval_mode == "bm25":
                return self.retriever.search_bm25(question, top_k=retrieval_k)
            return self.retriever.search_hybrid(question, top_k=retrieval_k)

        chunks = run_with_retries(
            search,
            retryable_exceptions=(TimeoutError,),
            max_attempts=2,
            delay_seconds=0.1,
        )
        if self.reranker is not None and chunks:
            return self.reranker.rerank(question, chunks, top_k=self.top_k)
        return chunks[: self.top_k]

    def _get_semantic_cached_response(self, question: str) -> dict[str, Any] | None:
        """Return a response from a sufficiently similar cached question.
        Embedding errors deliberately become cache misses; optional caching must
        never prevent the normal retrieval-and-generation path from running.
        """
        if not self.semantic_cache_enabled or self.cache_maxsize == 0:
            return None
        now = monotonic()
        expired_keys = [
            key
            for key, (expires_at, _, _) in self._semantic_cache.items()
            if now >= expires_at
        ]
        for key in expired_keys:
            del self._semantic_cache[key]
        if not self._semantic_cache:
            return None
        try:
            query_embedding = embed_texts([question])[0]
        except Exception:
            return None
        best_key: str | None = None
        best_similarity = -1.0
        for key, (expires_at, cached_embedding, _) in self._semantic_cache.items():
            if now >= expires_at:
                continue
            # Embeddings are normalized by embed_texts, so their dot product
            # is cosine similarity (within floating-point precision).
            try:
                similarity = sum(
                    left * right
                    for left, right in zip(
                        query_embedding, cached_embedding, strict=True
                    )
                )
            except (TypeError, ValueError):
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key
        if (
            best_key is None
            or best_similarity < self.semantic_cache_similarity_threshold
        ):
            return None
        _, _, response = self._semantic_cache[best_key]
        self._semantic_cache.move_to_end(best_key)
        cached_response = deepcopy(response)
        cached_response["question"] = question
        return cached_response

    def _cache_semantic_response(self, question: str, response: dict[str, Any]) -> None:
        """Cache only successful, grounded answers for semantic reuse."""
        if (
            not self.semantic_cache_enabled
            or self.cache_maxsize == 0
            or self.semantic_cache_ttl_seconds == 0
            or is_abstention_response(response.get("answer", ""))
        ):
            return
        # The semantic cache is an optimization: if embedding fails, keep the
        # generated answer and exact-match cache rather than raising here.
        try:
            embedding = embed_texts([question])[0]
        except Exception:
            return
        key = normalize_text(question)
        self._semantic_cache[key] = (
            monotonic() + self.semantic_cache_ttl_seconds,
            embedding,
            deepcopy(response),
        )
        self._semantic_cache.move_to_end(key)
        while len(self._semantic_cache) > self.cache_maxsize:
            self._semantic_cache.popitem(last=False)

    def _fallback_result(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build and briefly cache a consistent abstention response."""
        response = {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": self.retrieval_mode,
        }
        self._cache_response(
            question,
            response,
            ttl_seconds=self.fallback_cache_ttl_seconds,
        )
        return response

    def ask(self, question: str) -> dict[str, Any]:
        """Return a validated answer, using exact and optional semantic caches."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        # Exact-match cache has priority.
        cached_response = self._get_cached_response(question)
        if cached_response is not None:
            return cached_response
        # Semantic matching is opt-in and only reuses grounded cached answers.
        cached_response = self._get_semantic_cached_response(question)
        if cached_response is not None:
            return cached_response
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
                return self._fallback_result(question, retrieved_chunks)
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
        response = {
            "question": question,
            "answer": answer,
            "sources": generation_result.get("sources", []),
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": self.retrieval_mode,
        }
        self._cache_response(question, response)
        self._cache_semantic_response(question, response)
        return response
