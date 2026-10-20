from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """Load and cache an embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[list[float]]:
    """Convert text strings into normalized embedding vectors."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not contain empty or whitespace-only strings")

    model = get_embedding_model(model_name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()
