from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingest.embedder import embed_texts


class QdrantVectorStore:
    """Store and search document embeddings using Qdrant."""

    def __init__(
        self,
        collection_name: str = "support_chunks",
        vector_size: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """Embed document chunks and store them with their metadata."""
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=dict(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = embed_texts([query])[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": point.score,
                **(point.payload or {}),
            }
            for point in results.points
        ]
