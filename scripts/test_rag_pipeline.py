from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import RetrievalEngine

if __name__ == "__main__":
    chunks = [
        {
            "text": (
                "To reset your password, open Account Settings, "
                "select Security, and click Reset Password."
            ),
            "source": "account-help.md",
            "chunk_index": 0,
        },
        {
            "text": (
                "For database connection failures, verify the database "
                "hostname, port, credentials, and network connectivity."
            ),
            "source": "database-troubleshooting.md",
            "chunk_index": 0,
        },
        {
            "text": ("Support hours are Monday to Friday, 9 AM to 5 PM."),
            "source": "support-hours.md",
            "chunk_index": 0,
        },
    ]

    print("Initializing retrieval engine...")
    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="day6_smoke_test",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        retrieval_mode="hybrid",
        top_k=2,
        max_new_tokens=100,
    )

    question = "How do I reset my password?"

    print(f"\nQuestion: {question}")
    print("\nRunning end-to-end RAG pipeline...")

    result = pipeline.ask(question)

    print("\n--- Retrieved Chunks ---")
    for chunk in result["retrieved_chunks"]:
        print(f"Source: {chunk.get('source')}")
        print(f"Text: {chunk.get('text')}")
        print()

    print("--- Generated Answer ---")
    print(result["answer"])

    print("\n--- Sources ---")
    print(result["sources"])

    print("\n--- Retrieval Mode ---")
    print(result["retrieval_mode"])
