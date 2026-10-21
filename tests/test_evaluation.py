from unittest.mock import Mock

import pytest

from src.evaluation.evaluator import evaluate_retrieval


@pytest.fixture
def retriever():
    mock = Mock()
    mock.search_dense.return_value = [
        {"source": "account-help.md"},
        {"source": "database-troubleshooting.md"},
    ]
    mock.search_bm25.return_value = [
        {"source": "database-troubleshooting.md"},
        {"source": "account-help.md"},
    ]
    mock.search_hybrid.return_value = [
        {"source": "account-help.md"},
        {"source": "database-troubleshooting.md"},
    ]
    return mock


@pytest.fixture
def dataset():
    return [
        {
            "id": "q001",
            "question": "How do I reset my password?",
            "expected_sources": ["account-help.md"],
        },
        {
            "id": "q002",
            "question": "How do I troubleshoot a database?",
            "expected_sources": ["database-troubleshooting.md"],
        },
        {
            "id": "q003",
            "question": "What is the phone number?",
            "expected_sources": [],
        },
    ]


def test_evaluator_returns_metrics_for_all_modes(retriever, dataset):
    results = evaluate_retrieval(retriever, dataset, top_k=2)

    assert set(results) == {"dense", "bm25", "hybrid"}

    for metrics in results.values():
        assert metrics["queries_evaluated"] == 2
        assert metrics["hit_rate_at_k"] == 1.0
        assert metrics["recall_at_k"] == 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0


def test_evaluator_calculates_reciprocal_rank(retriever, dataset):
    results = evaluate_retrieval(retriever, dataset, top_k=2)

    assert results["bm25"]["details"][0]["reciprocal_rank"] == 0.5
    assert results["hybrid"]["details"][0]["reciprocal_rank"] == 1.0


def test_evaluator_rejects_invalid_top_k(retriever, dataset):
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(retriever, dataset, top_k=0)


def test_evaluator_skips_questions_without_expected_sources(retriever, dataset):
    results = evaluate_retrieval(retriever, dataset)

    for metrics in results.values():
        assert all(detail["id"] != "q003" for detail in metrics["details"])
        assert metrics["queries_evaluated"] == 2
