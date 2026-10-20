from src.ingest.cleaner import clean_text, process_text, redact_pii
from src.ingest.loaders import load_document, load_documents


def test_clean_text_normalizes_whitespace():
    assert clean_text("Hello\n\n world   !") == "Hello world !"


def test_redact_email():
    result = redact_pii("Contact john.doe@example.com")
    assert "john.doe@example.com" not in result
    assert "[EMAIL_REDACTED]" in result


def test_redact_indian_phone():
    result = redact_pii("Call +91 9876543210")
    assert "9876543210" not in result
    assert "[PHONE_REDACTED]" in result


def test_process_text_cleans_and_redacts():
    result = process_text("Email: user@example.com\n\nCall me")
    assert result == "Email: [EMAIL_REDACTED] Call me"


def test_load_markdown_document(tmp_path):
    file_path = tmp_path / "guide.md"
    file_path.write_text(
        "Hello\n\nContact user@example.com",
        encoding="utf-8",
    )

    document = load_document(file_path)

    assert document["source"] == "guide.md"
    assert document["file_type"] == "md"
    assert document["text"] == "Hello Contact [EMAIL_REDACTED]"


def test_load_documents_ignores_unsupported_files(tmp_path):
    (tmp_path / "guide.md").write_text("Hello", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Ignore me", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "guide.md"
