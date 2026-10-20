import re
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


def _split_recursively(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """Split oversized text using progressively finer boundaries."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        return [
            text[index : index + chunk_size]
            for index in range(0, len(text), chunk_size)
        ]

    separator = separators[0]

    # An empty separator means splitting into individual characters.
    if separator == "":
        return [
            text[index : index + chunk_size]
            for index in range(0, len(text), chunk_size)
        ]

    if separator not in text:
        return _split_recursively(text, chunk_size, separators[1:])

    parts = text.split(separator)

    # Keep each separator attached to the preceding text.
    splits = [
        part + separator if index < len(parts) - 1 else part
        for index, part in enumerate(parts)
    ]

    result: list[str] = []

    for part in splits:
        if not part:
            continue

        if len(part) <= chunk_size:
            result.append(part)
        else:
            result.extend(_split_recursively(part, chunk_size, separators[1:]))

    return result


# def _split_recursively(
#     text: str,
#     chunk_size: int,
#     separators: list[str],
# ) -> list[str]:
#     """Split oversized text using progressively finer boundaries."""
#     if len(text) <= chunk_size:
#         return [text] if text else []

#     if not separators:
#         return list(text)

#     separator = separators[0]

#     if separator not in text:
#         return _split_recursively(text, chunk_size, separators[1:])

#     parts = text.split(separator)

#     # Keep each separator attached to the preceding text.
#     splits = [
#         part + separator if index < len(parts) - 1 else part
#         for index, part in enumerate(parts)
#     ]

#     result: list[str] = []

#     for part in splits:
#         if not part:
#             continue

#         if len(part) <= chunk_size:
#             result.append(part)
#         else:
#             result.extend(_split_recursively(part, chunk_size, separators[1:]))

#     return result


# def _merge_splits(
#     splits: list[str],
#     chunk_size: int,
#     overlap: int,
# ) -> list[str]:
#     """Merge smaller text pieces into chunks with character overlap."""
#     chunks: list[str] = []
#     current: list[str] = []
#     current_length = 0

#     for split in splits:
#         if not split:
#             continue

#         if current and current_length + len(split) > chunk_size:
#             chunk_text = "".join(current).strip()

#             if chunk_text:
#                 chunks.append(chunk_text)

#             # Retain a suffix of the previous content for overlap.
#             while current and current_length > overlap:
#                 removed = current.pop(0)
#                 current_length -= len(removed)

#         current.append(split)
#         current_length += len(split)

#     final_text = "".join(current).strip()

#     if final_text:
#         chunks.append(final_text)

#     return chunks


def _merge_splits(
    splits: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Merge text pieces while enforcing the chunk-size limit."""
    chunks: list[str] = []
    current = ""

    for split in splits:
        if not split:
            continue

        # Safety net: no individual piece may exceed the limit.
        pieces = [
            split[index : index + chunk_size]
            for index in range(0, len(split), chunk_size)
        ]

        for piece in pieces:
            if current and len(current) + len(piece) > chunk_size:
                completed = current.strip()

                if completed:
                    chunks.append(completed)

                # Keep as much overlap as fits alongside the next piece.
                keep = min(overlap, chunk_size - len(piece))
                current = current[-keep:] if keep else ""

            current += piece

    final_text = current.strip()

    if final_text:
        chunks.append(final_text)

    return chunks


def chunk_document_recursive(
    document: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a document recursively using natural text boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = document["text"].strip()

    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    splits = _split_recursively(text, chunk_size, separators)
    chunk_texts = _merge_splits(splits, chunk_size, overlap)

    return [
        {
            "text": chunk_text,
            "source": document["source"],
            "file_type": document["file_type"],
            "chunk_index": index,
        }
        for index, chunk_text in enumerate(chunk_texts)
    ]


def chunk_document_by_heading(
    document: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split Markdown documents into chunks with heading context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = document["text"].strip()

    if not text:
        return []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    sections: list[tuple[list[str], list[str]]] = []
    heading_stack: list[tuple[int, str]] = []
    body_lines: list[str] = []

    def save_section() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            context = [heading for _, heading in heading_stack]
            sections.append((context, [body]))

    for line in text.splitlines():
        match = heading_pattern.match(line.strip())

        if match:
            save_section()
            body_lines.clear()

            level = len(match.group(1))
            heading = match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading))
        else:
            body_lines.append(line)

    save_section()

    chunks: list[Chunk] = []

    for headings, bodies in sections:
        heading_context = " > ".join(headings)
        content = "\n".join(bodies).strip()

        prefix = f"{heading_context}\n" if heading_context else ""
        available_size = chunk_size - len(prefix)

        if available_size <= 0:
            raise ValueError(
                "Heading context is too long for the configured chunk_size"
            )

        section_document = {
            "text": content,
            "source": document["source"],
            "file_type": document["file_type"],
        }

        section_chunks = chunk_document_recursive(
            section_document,
            chunk_size=available_size,
            overlap=min(overlap, available_size - 1),
        )

        for section_chunk in section_chunks:
            chunk_text = prefix + section_chunk["text"]

            chunks.append(
                {
                    "text": chunk_text,
                    "source": document["source"],
                    "file_type": document["file_type"],
                    "chunk_index": len(chunks),
                }
            )

    return chunks
