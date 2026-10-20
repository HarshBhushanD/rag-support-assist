from typing import TypedDict


class Chunk(TypedDict):
    text: str
    source: str
    file_type: str
    chunk_index: int


def chunk_document(
    document: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a document into character-sized chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = document["text"].strip()

    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": document["source"],
                    "file_type": document["file_type"],
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
        if end == len(text):
            break
        start += step

    return chunks
