import pytest

from src.ingest.chunker import chunk_document


def test_chunk_document_preserves_metadata():
    document = {
        "text": "abcdefghij",
        "source": "guide.md",
        "file_type": "md",
    }

    chunks = chunk_document(document, chunk_size=6, overlap=2)

    assert [chunk["text"] for chunk in chunks] == [
        "abcdef",
        "efghij",
    ]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(chunk["source"] == "guide.md" for chunk in chunks)
    assert all(chunk["file_type"] == "md" for chunk in chunks)


def test_empty_document_returns_no_chunks():
    document = {"text": "   ", "source": "empty.md", "file_type": "md"}

    assert chunk_document(document) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_settings_raise_error(chunk_size, overlap):
    document = {"text": "hello world", "source": "guide.md", "file_type": "md"}

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size, overlap)
