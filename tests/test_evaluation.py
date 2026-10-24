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
        assert metrics["queries_evaluated"] == 3
        assert metrics["answerable_queries"] == 2
        assert metrics["unanswerable_queries"] == 1
        assert metrics["hit_rate_at_k"] == 1.0
        assert metrics["recall_at_k"] == 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0


def test_evaluator_rejects_invalid_top_k(retriever, dataset):
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(retriever, dataset, top_k=0)


def test_evaluator_includes_unanswerable_questions(retriever, dataset):
    results = evaluate_retrieval(retriever, dataset)

    for mode_metrics in results.values():
        assert mode_metrics["queries_evaluated"] == 3
        assert mode_metrics["answerable_queries"] == 2
        assert mode_metrics["unanswerable_queries"] == 1

        unanswerable = next(
            detail for detail in mode_metrics["details"] if detail["id"] == "q003"
        )

        assert unanswerable["is_unanswerable"] is True
        assert unanswerable["expected_sources"] == []
        assert "empty_retrieval" in unanswerable
        assert "returned_results" in unanswerable


def test_evaluator_adds_reranked_hybrid_when_reranker_is_provided(
    retriever,
    dataset,
):
    reranker = Mock()

    def fake_rerank(question, candidates, top_k):
        # Return the candidate whose source supports this question.
        expected_source = (
            "account-help.md"
            if "password" in question.lower()
            else "database-troubleshooting.md"
        )
        matching = [
            chunk for chunk in candidates if chunk.get("source") == expected_source
        ]
        return matching[:top_k]

    retriever.search_hybrid.return_value = [
        {
            "text": "Relevant support documentation.",
            "source": "account-help.md",
        },
        {
            "text": "Database troubleshooting guide.",
            "source": "database-troubleshooting.md",
        },
    ]
    reranker.rerank.side_effect = fake_rerank

    results = evaluate_retrieval(
        retriever,
        dataset,
        top_k=1,
        reranker=reranker,
        candidate_k=6,
    )

    assert set(results) == {
        "dense",
        "bm25",
        "hybrid",
        "reranked_hybrid",
    }

    metrics = results["reranked_hybrid"]
    assert metrics["queries_evaluated"] == 3
    assert metrics["answerable_queries"] == 2
    assert metrics["unanswerable_queries"] == 1
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0

    retriever.search_hybrid.assert_any_call(
        "How do I reset my password?",
        top_k=6,
    )
    reranker.rerank.assert_called()


def test_evaluator_rejects_candidate_k_smaller_than_top_k(
    retriever,
    dataset,
):
    reranker = Mock()

    with pytest.raises(ValueError, match="candidate_k"):
        evaluate_retrieval(
            retriever,
            dataset,
            top_k=3,
            reranker=reranker,
            candidate_k=2,
        )


def test_unanswerable_retrieval_rates_are_calculated_correctly(
    retriever,
    dataset,
):
    # q003 is unanswerable, and the mock retriever returns results.
    results = evaluate_retrieval(retriever, dataset, top_k=2)

    for metrics in results.values():
        assert metrics["unanswerable_returned_results_rate"] == 1.0
        assert metrics["unanswerable_empty_retrieval_rate"] == 0.0

        unanswerable = next(
            detail for detail in metrics["details"] if detail["id"] == "q003"
        )
        assert unanswerable["returned_results"] is True
        assert unanswerable["empty_retrieval"] is False


def test_evaluator_handles_dataset_without_unanswerable_queries(retriever):
    dataset = [
        {
            "id": "q001",
            "question": "How do I reset my password?",
            "expected_sources": ["account-help.md"],
            "answerable": True,
        }
    ]

    results = evaluate_retrieval(retriever, dataset, top_k=2)

    for metrics in results.values():
        assert metrics["answerable_queries"] == 1
        assert metrics["unanswerable_queries"] == 0
        assert metrics["unanswerable_empty_retrieval_rate"] == 0.0
        assert metrics["unanswerable_returned_results_rate"] == 0.0


def test_evaluator_calculates_precision_at_k(retriever, dataset):
    retriever.search_dense.return_value = [
        {"source": "account-help.md"},
        {"source": "unrelated.md"},
    ]

    results = evaluate_retrieval(retriever, dataset, top_k=2)
    metrics = results["dense"]

    # q001 retrieves one relevant and one irrelevant chunk: precision = 1/2.
    q001 = next(detail for detail in metrics["details"] if detail["id"] == "q001")
    assert q001["precision"] == 0.5

    # q002 retrieves no matching source: precision = 0.
    q002 = next(detail for detail in metrics["details"] if detail["id"] == "q002")
    assert q002["precision"] == 0.0

    # Mean precision over the two answerable questions: (0.5 + 0.0) / 2.
    assert metrics["precision_at_k"] == 0.25


def test_precision_at_k_is_zero_when_no_results_are_retrieved(
    retriever,
    dataset,
):
    retriever.search_dense.return_value = []

    results = evaluate_retrieval(retriever, dataset, top_k=2)
    metrics = results["dense"]

    assert metrics["precision_at_k"] == 0.0

    for detail in metrics["details"]:
        if not detail["is_unanswerable"]:
            assert detail["precision"] == 0.0


def test_precision_at_k_is_zero_when_no_queries_are_answerable(retriever):
    dataset = [
        {
            "id": "q003",
            "question": "What is the phone number?",
            "expected_sources": [],
            "answerable": False,
        }
    ]

    results = evaluate_retrieval(retriever, dataset, top_k=2)

    for metrics in results.values():
        assert metrics["answerable_queries"] == 0
        assert metrics["precision_at_k"] == 0.0
