from unittest.mock import Mock, patch

import pytest

from src.pipeline.rag_pipeline import RAGPipeline


@pytest.fixture
def retriever():
    return Mock()


@pytest.fixture
def chunks():
    return [
        {
            "text": "Reset your password from Account Settings.",
            "source": "account-help.md",
            "chunk_index": 0,
        }
    ]


def test_ask_uses_hybrid_retrieval_and_generates_answer(
    retriever,
    chunks,
):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset it from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever)
        result = pipeline.ask("How do I reset my password?")

    retriever.search_hybrid.assert_called_once_with(
        "How do I reset my password?",
        top_k=3,
    )
    mock_generate.assert_called_once_with(
        question="How do I reset my password?",
        context_chunks=chunks,
        max_new_tokens=160,
    )
    assert result["answer"] == "Reset it from Account Settings."
    assert result["sources"] == ["account-help.md"]
    assert result["retrieved_chunks"] == chunks
    assert result["retrieval_mode"] == "hybrid"


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("dense", "search_dense"),
        ("bm25", "search_bm25"),
        ("hybrid", "search_hybrid"),
    ],
)
def test_retrieve_uses_selected_mode(
    retriever,
    mode,
    method_name,
):
    expected_chunks = [{"text": "Example context.", "source": "guide.md"}]
    getattr(retriever, method_name).return_value = expected_chunks

    pipeline = RAGPipeline(
        retriever,
        retrieval_mode=mode,
    )

    assert pipeline.retrieve("Example question?") == expected_chunks
    getattr(retriever, method_name).assert_called_once_with(
        "Example question?",
        top_k=3,
    )


def test_ask_returns_fallback_when_no_chunks_are_retrieved(retriever):
    retriever.search_hybrid.return_value = []

    with patch("src.pipeline.rag_pipeline.generate_answer") as mock_generate:
        result = RAGPipeline(retriever).ask("Unknown question?")

    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []
    mock_generate.assert_not_called()


def test_retrieve_rejects_empty_question(retriever):
    pipeline = RAGPipeline(retriever)

    with pytest.raises(ValueError, match="question must not be empty"):
        pipeline.retrieve("   ")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retrieval_mode": "invalid"}, "retrieval_mode"),
        ({"top_k": 0}, "top_k"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
    ],
)
def test_pipeline_rejects_invalid_configuration(
    retriever,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        RAGPipeline(retriever, **kwargs)


def test_pipeline_reranks_candidates_before_generation(retriever):
    candidates = [
        {"text": "Password reset instructions.", "source": "account.md"},
        {"text": "Database hostname configuration.", "source": "database.md"},
        {"text": "API token authentication.", "source": "api.md"},
        {"text": "Other technical documentation.", "source": "other.md"},
        {"text": "Network troubleshooting.", "source": "network.md"},
        {"text": "Deployment instructions.", "source": "deploy.md"},
    ]

    reranked_chunks = [
        {**candidates[1], "reranker_score": 0.95},
        {**candidates[4], "reranker_score": 0.85},
    ]

    retriever.search_hybrid.return_value = candidates
    reranker = Mock()
    reranker.rerank.return_value = reranked_chunks

    question = "How do I connect to the database?"

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Check the database hostname.",
            "sources": ["database.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(
            retriever,
            retrieval_mode="hybrid",
            top_k=2,
            reranker=reranker,
            candidate_k=6,
            min_reranker_score=0.0,
        )

        result = pipeline.ask(question)

    retriever.search_hybrid.assert_called_once_with(
        question,
        top_k=6,
    )
    reranker.rerank.assert_called_once_with(
        question,
        candidates,
        top_k=2,
    )
    mock_generate.assert_called_once_with(
        question=question,
        context_chunks=reranked_chunks,
        max_new_tokens=160,
    )
    assert result["retrieved_chunks"] == reranked_chunks


def test_pipeline_rejects_candidate_k_smaller_than_top_k(retriever):
    with pytest.raises(ValueError, match="candidate_k"):
        RAGPipeline(
            retriever,
            top_k=3,
            candidate_k=2,
        )


def test_pipeline_without_reranker_keeps_existing_retrieval(retriever):
    chunks = [
        {"text": "First chunk.", "source": "first.md"},
        {"text": "Second chunk.", "source": "second.md"},
        {"text": "Third chunk.", "source": "third.md"},
    ]

    retriever.search_hybrid.return_value = chunks

    pipeline = RAGPipeline(
        retriever,
        top_k=2,
        reranker=None,
    )

    result = pipeline.retrieve("Example question?")

    assert result == chunks[:2]
    retriever.search_hybrid.assert_called_once_with(
        "Example question?",
        top_k=2,
    )


def test_pipeline_abstains_when_all_reranked_chunks_have_weak_scores(
    retriever,
):
    candidates = [
        {"text": "API request troubleshooting.", "source": "api-help.html"},
        {"text": "Database connection settings.", "source": "troubleshooting.md"},
    ]

    weak_chunks = [
        {**candidates[0], "reranker_score": -2.0},
        {**candidates[1], "reranker_score": -1.5},
    ]

    retriever.search_hybrid.return_value = candidates
    reranker = Mock()
    reranker.rerank.return_value = weak_chunks

    with patch("src.pipeline.rag_pipeline.generate_answer") as mock_generate:
        pipeline = RAGPipeline(
            retriever,
            reranker=reranker,
            top_k=2,
            candidate_k=6,
            min_reranker_score=0.0,
        )
        result = pipeline.ask("What is the support team's phone number?")

    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
    assert result["retrieved_chunks"] == weak_chunks
    mock_generate.assert_not_called()


def test_pipeline_generates_answer_when_best_score_meets_threshold(
    retriever,
):
    candidates = [
        {"text": "Relevant support instructions.", "source": "support.md"},
        {"text": "Other information.", "source": "other.md"},
    ]

    reranked_chunks = [
        {**candidates[0], "reranker_score": 0.8},
        {**candidates[1], "reranker_score": -0.5},
    ]

    retriever.search_hybrid.return_value = candidates
    reranker = Mock()
    reranker.rerank.return_value = reranked_chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "See the support instructions.",
            "sources": ["support.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(
            retriever,
            reranker=reranker,
            top_k=2,
            candidate_k=2,
            min_reranker_score=0.0,
        )

        result = pipeline.ask("Where are the support instructions?")

    assert result["answer"] == "See the support instructions."
    mock_generate.assert_called_once()


def test_pipeline_abstains_when_generator_cannot_answer(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": (
                "The context does not provide this information. "
                "I cannot determine the answer."
            ),
            "sources": ["account-help.md"],
        },
    ):
        result = RAGPipeline(retriever).ask("What is the support team's phone number?")

    assert result["answer"] == (
        "I don't have enough information in the provided documents "
        "to answer that question."
    )
    assert result["sources"] == []


def test_pipeline_normalizes_malformed_fallback(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": (
                "I don't have enough information in the provided "
                "documentsto answer that question."
            ),
            "sources": ["account-help.md"],
        },
    ):
        result = RAGPipeline(retriever).ask("What is the support team's phone number?")

    assert result["answer"] == (
        "I don't have enough information in the provided documents "
        "to answer that question."
    )
    assert result["sources"] == []


@pytest.mark.parametrize(
    "question,unsupported_answer",
    [
        (
            "What are the support team's working hours?",
            "The support team works on weekdays from 8:00 AM to 5:00 PM.",
        ),
        (
            "Is support available on Saturday?",
            "Support is not available on Saturdays.",
        ),
        (
            "What is the support team's phone number?",
            "The support team's phone number is 1234567890.",
        ),
    ],
)
def test_pipeline_does_not_accept_unsupported_claims(
    retriever,
    chunks,
    question,
    unsupported_answer,
):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": unsupported_answer,
            "sources": ["auth-help.md"],
        },
    ):
        result = RAGPipeline(retriever).ask(question)

    assert result["answer"] == (
        "I don't have enough information in the provided documents "
        "to answer that question."
    )
    assert result["sources"] == []


def test_ask_reuses_cached_response(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset your password from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever)

        first = pipeline.ask("How do I reset my password?")
        second = pipeline.ask("How do I reset my password?")

    assert first == second
    retriever.search_hybrid.assert_called_once()
    mock_generate.assert_called_once()


def test_ask_cache_is_case_and_whitespace_insensitive(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset your password from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever)

        pipeline.ask("How do I reset my password?")
        second = pipeline.ask("  HOW DO I RESET MY PASSWORD?  ")

    assert second["question"] == "  HOW DO I RESET MY PASSWORD?  "
    retriever.search_hybrid.assert_called_once()
    mock_generate.assert_called_once()


def test_ask_reuses_cached_fallback_response(retriever):
    retriever.search_hybrid.return_value = []

    pipeline = RAGPipeline(retriever)

    first = pipeline.ask("Unknown question?")
    second = pipeline.ask("Unknown question?")

    assert first["answer"] == second["answer"]
    assert first["sources"] == []
    assert retriever.search_hybrid.call_count == 1


def test_clear_cache_forces_new_retrieval(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset your password from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever)

        pipeline.ask("How do I reset my password?")
        pipeline.clear_cache()
        pipeline.ask("How do I reset my password?")

    assert retriever.search_hybrid.call_count == 2
    assert mock_generate.call_count == 2


def test_cache_maxsize_zero_disables_caching(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset your password from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever, cache_maxsize=0)

        pipeline.ask("How do I reset my password?")
        pipeline.ask("How do I reset my password?")

    assert retriever.search_hybrid.call_count == 2
    assert mock_generate.call_count == 2


def test_cache_evicts_least_recently_used_entry(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset your password from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever, cache_maxsize=2)

        pipeline.ask("Question one?")
        pipeline.ask("Question two?")
        pipeline.ask("Question one?")  # Refresh question one's LRU position.
        pipeline.ask("Question three?")  # Evicts question two.
        pipeline.ask("Question one?")  # Still cached.
        pipeline.ask("Question two?")  # Must be recomputed.

    assert mock_generate.call_count == 4
    assert retriever.search_hybrid.call_count == 4


def test_cache_rejects_invalid_configuration(retriever):
    with pytest.raises(ValueError, match="cache_maxsize"):
        RAGPipeline(retriever, cache_maxsize=-1)

    with pytest.raises(ValueError, match="cache_ttl_seconds"):
        RAGPipeline(retriever, cache_ttl_seconds=-1)

    with pytest.raises(ValueError, match="cache_ttl_seconds"):
        RAGPipeline(retriever, cache_ttl_seconds=float("inf"))

    with pytest.raises(ValueError, match="fallback_cache_ttl_seconds"):
        RAGPipeline(retriever, fallback_cache_ttl_seconds=-1)

    with pytest.raises(ValueError, match="fallback_cache_ttl_seconds"):
        RAGPipeline(retriever, fallback_cache_ttl_seconds=float("inf"))


def test_expired_cache_entry_is_recomputed(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with patch(
        "src.pipeline.rag_pipeline.generate_answer",
        return_value={
            "answer": "Reset your password from Account Settings.",
            "sources": ["account-help.md"],
        },
    ) as mock_generate:
        pipeline = RAGPipeline(retriever, cache_ttl_seconds=0.01)

        pipeline.ask("How do I reset my password?")

        # Wait longer than the configured TTL.
        import time

        time.sleep(0.02)

        pipeline.ask("How do I reset my password?")

    assert retriever.search_hybrid.call_count == 2
    assert mock_generate.call_count == 2


def test_expired_fallback_cache_entry_is_recomputed(retriever):
    retriever.search_hybrid.return_value = []

    pipeline = RAGPipeline(
        retriever,
        fallback_cache_ttl_seconds=0.01,
    )

    pipeline.ask("Unknown question?")

    import time

    time.sleep(0.02)

    pipeline.ask("Unknown question?")

    assert retriever.search_hybrid.call_count == 2


def test_fallback_cache_ttl_zero_disables_fallback_caching(retriever):
    retriever.search_hybrid.return_value = []

    pipeline = RAGPipeline(
        retriever,
        fallback_cache_ttl_seconds=0,
    )

    pipeline.ask("Unknown question?")
    pipeline.ask("Unknown question?")

    assert retriever.search_hybrid.call_count == 2


def _mock_semantic_embedding(question: str) -> list[float]:
    """Return deterministic embeddings for semantic-cache unit tests."""
    normalized = question.casefold()

    if "database" in normalized:
        return [0.0, 1.0]

    return [1.0, 0.0]


def test_semantic_cache_reuses_answer_for_similar_question(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with (
        patch(
            "src.pipeline.rag_pipeline.embed_texts",
            side_effect=lambda texts: [
                _mock_semantic_embedding(text) for text in texts
            ],
        ),
        patch(
            "src.pipeline.rag_pipeline.generate_answer",
            return_value={
                "answer": "Reset your password from Account Settings.",
                "sources": ["account-help.md"],
            },
        ) as mock_generate,
    ):
        pipeline = RAGPipeline(
            retriever,
            semantic_cache_enabled=True,
            semantic_cache_similarity_threshold=0.92,
        )

        first = pipeline.ask("How do I reset my password?")
        second = pipeline.ask("How can I reset my password?")

    assert first["answer"] == second["answer"]
    assert second["question"] == "How can I reset my password?"
    retriever.search_hybrid.assert_called_once()
    mock_generate.assert_called_once()


def test_semantic_cache_does_not_reuse_unrelated_question(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with (
        patch(
            "src.pipeline.rag_pipeline.embed_texts",
            side_effect=lambda texts: [
                _mock_semantic_embedding(text) for text in texts
            ],
        ),
        patch(
            "src.pipeline.rag_pipeline.generate_answer",
            return_value={
                "answer": "Reset your password from Account Settings.",
                "sources": ["account-help.md"],
            },
        ) as mock_generate,
    ):
        pipeline = RAGPipeline(
            retriever,
            semantic_cache_enabled=True,
            semantic_cache_similarity_threshold=0.92,
        )

        pipeline.ask("How do I reset my password?")
        pipeline.ask("How do I configure the database?")

    assert retriever.search_hybrid.call_count == 2
    assert mock_generate.call_count == 2


def test_expired_semantic_cache_entry_is_not_reused(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with (
        patch(
            "src.pipeline.rag_pipeline.embed_texts",
            side_effect=lambda texts: [
                _mock_semantic_embedding(text) for text in texts
            ],
        ),
        patch(
            "src.pipeline.rag_pipeline.generate_answer",
            return_value={
                "answer": "Reset your password from Account Settings.",
                "sources": ["account-help.md"],
            },
        ) as mock_generate,
    ):
        pipeline = RAGPipeline(
            retriever,
            cache_ttl_seconds=0,
            semantic_cache_enabled=True,
            semantic_cache_ttl_seconds=300,
        )

        pipeline.ask("How do I reset my password?")

        # Force the semantic entry to expire without waiting.
        key = next(iter(pipeline._semantic_cache))
        expires_at, embedding, response = pipeline._semantic_cache[key]
        pipeline._semantic_cache[key] = (0.0, embedding, response)

        pipeline.ask("How can I reset my password?")

    assert retriever.search_hybrid.call_count == 2
    assert mock_generate.call_count == 2


def test_clear_cache_clears_semantic_cache(retriever, chunks):
    retriever.search_hybrid.return_value = chunks

    with (
        patch(
            "src.pipeline.rag_pipeline.embed_texts",
            side_effect=lambda texts: [
                _mock_semantic_embedding(text) for text in texts
            ],
        ),
        patch(
            "src.pipeline.rag_pipeline.generate_answer",
            return_value={
                "answer": "Reset your password from Account Settings.",
                "sources": ["account-help.md"],
            },
        ) as mock_generate,
    ):
        pipeline = RAGPipeline(
            retriever,
            semantic_cache_enabled=True,
        )

        pipeline.ask("How do I reset my password?")
        assert pipeline._semantic_cache

        pipeline.clear_cache()

        assert not pipeline._semantic_cache
        assert not pipeline._response_cache

        pipeline.ask("How can I reset my password?")

    assert retriever.search_hybrid.call_count == 2
    assert mock_generate.call_count == 2


def test_fallback_response_is_not_added_to_semantic_cache(retriever):
    retriever.search_hybrid.return_value = []

    with patch("src.pipeline.rag_pipeline.embed_texts") as mock_embed:
        pipeline = RAGPipeline(
            retriever,
            semantic_cache_enabled=True,
        )

        result = pipeline.ask("What is the support team's phone number?")

    assert "don't have enough information" in result["answer"]
    assert not pipeline._semantic_cache
    mock_embed.assert_not_called()


def test_semantic_cache_embedding_failure_falls_back_gracefully(
    retriever,
    chunks,
):
    retriever.search_hybrid.return_value = chunks

    with (
        patch(
            "src.pipeline.rag_pipeline.embed_texts",
            side_effect=RuntimeError("Embedding model unavailable"),
        ),
        patch(
            "src.pipeline.rag_pipeline.generate_answer",
            return_value={
                "answer": "Reset your password from Account Settings.",
                "sources": ["account-help.md"],
            },
        ) as mock_generate,
    ):
        pipeline = RAGPipeline(
            retriever,
            semantic_cache_enabled=True,
        )

        result = pipeline.ask("How do I reset my password?")

    assert result["answer"] == "Reset your password from Account Settings."
    assert result["sources"] == ["account-help.md"]
    mock_generate.assert_called_once()
    assert not pipeline._semantic_cache
