from pathlib import Path

from src.ingest.chunker import chunk_document_recursive
from src.ingest.loaders import load_documents
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

QUESTIONS = [
    "What should I check if login fails?",
    "What should I do if my account is inactive?",
    "How do I handle an invalid API token?",
    "What should I check if database connectivity fails?",
    "Why should database passwords not be stored in source code?",
    "How do I register a new account?",
    "What are the available API endpoints?",
    "How do I reset my password?",
]


def main():
    documents = load_documents(RAW_DIR)

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document_recursive(
            document,
            chunk_size=500,
            overlap=50,
        )
    ]

    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="retrieval_quality_test",
    )

    reranker = CrossEncoderReranker()

    print(f"\nDocuments: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        for mode in ["dense", "bm25", "hybrid"]:
            if mode == "dense":
                results = retriever.search_dense(question, top_k=6)
            elif mode == "bm25":
                results = retriever.search_bm25(question, top_k=6)
            else:
                results = retriever.search_hybrid(question, top_k=6)

            print(f"\n--- {mode.upper()} ---")

            for rank, result in enumerate(results, start=1):
                print(f"{rank}. {result['source']} | chunk={result['chunk_index']}")
                print(result["text"][:250].replace("\n", " "))

        candidates = retriever.search_hybrid(question, top_k=6)
        reranked = reranker.rerank(
            question,
            candidates,
            top_k=3,
        )

        print("\n--- HYBRID + RERANKER ---")

        for rank, result in enumerate(reranked, start=1):
            print(f"{rank}. {result['source']} | score={result['reranker_score']:.4f}")
            print(result["text"][:250].replace("\n", " "))


if __name__ == "__main__":
    main()
