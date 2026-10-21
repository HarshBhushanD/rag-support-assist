import pytest

from src.retrieval.retriever import (
    RetrievalEngine,
    reciprocal_rank_fusion,
    tokenize,
)


@pytest.fixture
def chunks():
    return [
        {
            "text": "Reset your password from the account login page.",
            "source": "account.md",
            "file_type": "md",
            "chunk_index": 0,
        },
        {
            "text": "Database connection failures require checking the hostname.",
            "source": "database.md",
            "file_type": "md",
            "chunk_index": 0,
        },
        {
            "text": "HTTP authentication errors require checking your API token.",
            "source": "api.md",
            "file_type": "md",
            "chunk_index": 0,
        },
    ]


@pytest.fixture
def engine(monkeypatch, chunks):
    """Use deterministic embeddings for fast, repeatable tests."""

    def fake_embed_texts(texts):
        vectors = []

        for text in texts:
            normalized = text.lower()
            vector = [0.0] * 384

            if "password" in normalized:
                vector[0] = 1.0
            elif "database" in normalized or "hostname" in normalized:
                vector[1] = 1.0
            else:
                vector[2] = 1.0

            vectors.append(vector)

        return vectors

    monkeypatch.setattr(
        "src.ingest.vector_store.embed_texts",
        fake_embed_texts,
    )

    return RetrievalEngine(chunks)


def test_tokenize():
    assert tokenize("Password RESET!") == ["password", "reset"]


def test_dense_retrieval_preserves_source(engine):
    results = engine.search_dense("password reset", top_k=2)

    assert len(results) == 2
    assert results[0]["source"] == "account.md"
    assert results[0]["retrieval_method"] == "dense"


def test_bm25_retrieval_finds_keyword_match(engine):
    results = engine.search_bm25("hostname database", top_k=3)

    assert len(results) == 3
    assert results[0]["source"] == "database.md"
    assert results[0]["bm25_score"] > 0


def test_hybrid_retrieval_preserves_metadata(engine):
    results = engine.search_hybrid("database hostname", top_k=2)

    assert len(results) == 2
    assert results[0]["source"] == "database.md"
    assert "rrf_score" in results[0]
    assert "retrieval_methods" in results[0]
    assert results[0]["file_type"] == "md"


def test_rrf_combines_rankings():
    dense = [
        {"text": "A", "source": "a.md", "chunk_index": 0},
        {"text": "B", "source": "b.md", "chunk_index": 0},
    ]
    bm25 = [
        {"text": "B", "source": "b.md", "chunk_index": 0},
        {"text": "A", "source": "a.md", "chunk_index": 0},
    ]

    results = reciprocal_rank_fusion(
        [dense, bm25],
        top_k=2,
        k=60,
    )

    expected_score = (1 / 61) + (1 / 62)

    assert len(results) == 2
    assert results[0]["rrf_score"] == pytest.approx(expected_score)
    assert results[1]["rrf_score"] == pytest.approx(expected_score)


# def test_rrf_combines_rankings():
#     dense = [
#         {"text": "A", "source": "a.md", "chunk_index": 0},
#         {"text": "B", "source": "b.md", "chunk_index": 0},
#     ]
#     bm25 = [
#         {"text": "B", "source": "b.md", "chunk_index": 0},
#         {"text": "A", "source": "a.md", "chunk_index": 0},
#     ]

#     results = reciprocal_rank_fusion(
#         [dense, bm25],
#         top_k=2,
#         k=60,
#     )

#     assert len(results) == 2
#     assert results[0]["rrf_score"] == pytest.approx(2 / 61)
#     assert results[1]["rrf_score"] == pytest.approx(2 / 62)


def test_rrf_deduplicates_same_chunk():
    chunk = {
        "text": "Password reset",
        "source": "account.md",
        "chunk_index": 0,
    }

    results = reciprocal_rank_fusion(
        [[chunk, chunk]],
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["rrf_score"] == pytest.approx(1 / 61)


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 3),
        ("   ", 3),
        ("password", 0),
        ("password", -1),
    ],
)
def test_dense_rejects_invalid_arguments(engine, query, top_k):
    with pytest.raises(ValueError):
        engine.search_dense(query, top_k=top_k)


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 3),
        ("   ", 3),
        ("password", 0),
        ("password", -1),
    ],
)
def test_bm25_rejects_invalid_arguments(engine, query, top_k):
    with pytest.raises(ValueError):
        engine.search_bm25(query, top_k=top_k)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="At least one chunk"):
        RetrievalEngine([])


def test_rrf_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)
