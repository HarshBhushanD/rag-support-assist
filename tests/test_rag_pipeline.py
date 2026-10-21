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


def test_ask_uses_hybrid_retrieval_and_generates_answer(retriever, chunks):
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
def test_retrieve_uses_selected_mode(retriever, mode, method_name):
    expected_chunks = [{"text": "Example context.", "source": "guide.md"}]
    getattr(retriever, method_name).return_value = expected_chunks

    pipeline = RAGPipeline(retriever, retrieval_mode=mode)

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
def test_pipeline_rejects_invalid_configuration(retriever, kwargs, message):
    with pytest.raises(ValueError, match=message):
        RAGPipeline(retriever, **kwargs)
