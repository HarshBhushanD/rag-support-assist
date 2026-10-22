# ruff: noqa: E402, I001
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.chunker import chunk_document_recursive
from src.ingest.loaders import load_documents
from src.retrieval.retriever import RetrievalEngine

ROOT = PROJECT_ROOT
EVAL_DIR = ROOT / "data" / "eval"

documents = load_documents("data/raw")

chunks = [
    chunk
    for document in documents
    for chunk in chunk_document_recursive(
        document,
        chunk_size=500,
        overlap=50,
    )
]

if not chunks:
    raise SystemExit("No document chunks found in data/raw")

engine = RetrievalEngine(
    chunks,
    collection_name="day4_evaluation",
)

queries = [
    "How do I troubleshoot a database connection?",
    "How do I fix an API request error?",
    "What should I check when authentication fails?",
]

for query in queries:
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")

    strategies = [
        ("DENSE", engine.search_dense),
        ("BM25", engine.search_bm25),
        ("HYBRID", engine.search_hybrid),
    ]

    for name, search in strategies:
        print(f"\n--- {name} ---")

        for rank, result in enumerate(search(query, top_k=3), start=1):
            print(
                f"{rank}. source={result['source']} | "
                f"chunk={result['chunk_index']} | "
                f"text={result['text'][:180]}"
            )

            if name == "DENSE":
                print(f"   similarity={result['score']:.4f}")
            elif name == "BM25":
                print(f"   bm25_score={result['score']:.4f}")
            else:
                print(f"   rrf_score={result['rrf_score']:.6f}")
