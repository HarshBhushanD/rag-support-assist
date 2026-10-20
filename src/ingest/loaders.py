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
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    elif suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)

    elif suffix in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return {
        "text": process_text(text),
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
