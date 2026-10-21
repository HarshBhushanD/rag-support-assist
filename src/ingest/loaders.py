from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.ingest.cleaner import process_text


class Document(TypedDict):
    text: str
    source: str
    file_type: str


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}


def load_document(file_path: str | Path) -> Document:
    """Load a PDF, HTML, or Markdown file into a common document format."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = process_text(raw_text)

    elif suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8-sig")
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        if soup.title:
            soup.title.decompose()

        raw_text = soup.get_text(separator="\n", strip=True)
        text = process_text(raw_text)

    elif suffix in {".md", ".markdown"}:
        raw_text = path.read_text(encoding="utf-8-sig")

        # Clean and redact each line independently.
        # Preserve Markdown headings and paragraph boundaries.
        lines = [process_text(line) for line in raw_text.splitlines() if line.strip()]

        text = "\n".join(line for line in lines if line)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return {
        "text": text,
        "source": path.name,
        "file_type": suffix.lstrip("."),
    }


def load_documents(directory: str | Path) -> list[Document]:
    """Load all supported documents in a directory, non-recursively."""
    root = Path(directory)

    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    documents = []

    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_document(path))

    return documents
