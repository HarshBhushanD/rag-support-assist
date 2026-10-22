import pytest

from src.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Deterministic model for unit tests."""

    def predict(self, pairs):
        scores = [0.2, 0.9, 0.5]
        return scores[: len(pairs)]


@pytest.fixture
def reranker():
    return CrossEncoderReranker(model=FakeCrossEncoder())


@pytest.fixture
def candidates():
    return [
        {
            "text": "Password reset instructions.",
            "source": "account.md",
        },
        {
            "text": "Database hostname configuration.",
            "source": "database.md",
        },
        {
            "text": "API token authentication.",
            "source": "api.md",
        },
    ]


def test_reranker_sorts_by_score(reranker, candidates):
    results = reranker.rerank(
        "How do I connect?",
        candidates,
        top_k=3,
    )

    assert [item["source"] for item in results] == [
        "database.md",
        "api.md",
        "account.md",
    ]
    assert [item["reranker_score"] for item in results] == pytest.approx(
        [0.9, 0.5, 0.2]
    )


def test_reranker_respects_top_k(reranker, candidates):
    results = reranker.rerank(
        "How do I connect?",
        candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["source"] == "database.md"


def test_reranker_preserves_metadata(reranker, candidates):
    results = reranker.rerank(
        "How do I connect?",
        candidates,
        top_k=1,
    )

    assert results[0]["source"] == "database.md"
    assert results[0]["text"] == "Database hostname configuration."
    assert results[0]["reranker_score"] == pytest.approx(0.9)


def test_reranker_does_not_modify_original_candidates(reranker, candidates):
    original = [chunk.copy() for chunk in candidates]

    reranker.rerank(
        "How do I connect?",
        candidates,
        top_k=2,
    )

    assert candidates == original


def test_reranker_handles_empty_candidates(reranker):
    results = reranker.rerank(
        "question",
        [],
        top_k=3,
    )

    assert results == []


@pytest.mark.parametrize("query", ["", "   "])
def test_reranker_rejects_empty_query(reranker, candidates, query):
    with pytest.raises(ValueError, match="Query"):
        reranker.rerank(query, candidates)


@pytest.mark.parametrize("top_k", [0, -1])
def test_reranker_rejects_invalid_top_k(
    reranker,
    candidates,
    top_k,
):
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank(
            "question",
            candidates,
            top_k=top_k,
        )


def test_reranker_rejects_missing_text(reranker):
    with pytest.raises(ValueError, match="non-empty text"):
        reranker.rerank(
            "question",
            [{"source": "missing.md"}],
        )


def test_reranker_rejects_empty_text(reranker):
    with pytest.raises(ValueError, match="non-empty text"):
        reranker.rerank(
            "question",
            [{"text": "   ", "source": "empty.md"}],
        )
