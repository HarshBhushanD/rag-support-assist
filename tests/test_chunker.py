import pytest

from src.ingest.chunker import (
    chunk_document,
    chunk_document_by_heading,
    chunk_document_recursive,
)


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


def test_recursive_chunking_preserves_metadata():
    document = {
        "text": (
            "First paragraph has useful information.\n\n"
            "Second paragraph has more useful information.\n\n"
            "Third paragraph adds further details."
        ),
        "source": "guide.md",
        "file_type": "md",
    }

    chunks = chunk_document_recursive(document, chunk_size=60, overlap=10)

    assert len(chunks) > 1
    assert all(chunk["source"] == "guide.md" for chunk in chunks)
    assert all(chunk["file_type"] == "md" for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_recursive_chunking_handles_long_text_without_spaces():
    document = {
        "text": "x" * 100,
        "source": "guide.md",
        "file_type": "md",
    }

    chunks = chunk_document_recursive(document, chunk_size=30, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 30 for chunk in chunks)


def test_recursive_chunking_empty_document():
    document = {"text": "   ", "source": "empty.md", "file_type": "md"}

    assert chunk_document_recursive(document) == []


def test_recursive_chunks_never_exceed_chunk_size():
    document = {
        "text": "a\n\n" + ("b" * 30),
        "source": "guide.md",
        "file_type": "md",
    }

    chunks = chunk_document_recursive(
        document,
        chunk_size=30,
        overlap=5,
    )

    assert chunks
    assert all(len(chunk["text"]) <= 30 for chunk in chunks)


def test_heading_chunking_preserves_heading_context():
    document = {
        "text": (
            "# Database Guide\n\n"
            "General database information.\n\n"
            "## Connection Errors\n\n"
            "Check the hostname and port.\n\n"
            "## Authentication Errors\n\n"
            "Verify the credentials."
        ),
        "source": "database.md",
        "file_type": "md",
    }

    chunks = chunk_document_by_heading(document, chunk_size=100, overlap=10)

    combined_text = "\n".join(chunk["text"] for chunk in chunks)

    assert any("Database Guide" in chunk["text"] for chunk in chunks)
    assert any(
        "Database Guide > Connection Errors" in chunk["text"] for chunk in chunks
    )
    assert any(
        "Database Guide > Authentication Errors" in chunk["text"] for chunk in chunks
    )
    assert "Check the hostname and port." in combined_text
    assert "Verify the credentials." in combined_text


def test_heading_chunking_preserves_metadata_and_size():
    document = {
        "text": "# Guide\n\n" + ("Troubleshooting details. " * 20),
        "source": "guide.md",
        "file_type": "md",
    }

    chunks = chunk_document_by_heading(document, chunk_size=80, overlap=10)

    assert chunks
    assert all(chunk["source"] == "guide.md" for chunk in chunks)
    assert all(chunk["file_type"] == "md" for chunk in chunks)
    assert all(len(chunk["text"]) <= 80 for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_heading_chunking_empty_document():
    document = {"text": "   ", "source": "empty.md", "file_type": "md"}

    assert chunk_document_by_heading(document) == []


def test_recursive_chunking_preserves_requested_overlap():
    document = {
        "text": "abcdefghijklmnopqrstuvwxyz" * 4,
        "source": "overlap-test.md",
        "file_type": "md",
    }

    chunks = chunk_document_recursive(
        document,
        chunk_size=30,
        overlap=5,
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert previous["text"][-5:] in current["text"]
