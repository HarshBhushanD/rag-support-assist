# ruff: noqa: E402, I001
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_retrieval
from src.evaluation.mlflow_tracker import log_retrieval_experiment
from src.ingest.chunker import chunk_document_recursive
from src.ingest.loaders import load_documents
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DATASET_PATH = EVAL_DIR / "expanded_retrieval_dataset.json"
OUTPUT_PATH = EVAL_DIR / "expanded_retrieval_results.json"

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 2
CANDIDATE_K = 6


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    dataset = load_json(DATASET_PATH)
    documents = load_documents(PROJECT_ROOT / "data" / "raw")

    if not documents:
        raise SystemExit("No supported documents found in data/raw")

    if not dataset:
        raise SystemExit("The expanded evaluation dataset is empty")

    print(f"Loaded {len(documents)} raw documents.")
    print(f"Evaluating {len(dataset)} questions.")

    answerable_count = sum(bool(item.get("expected_sources")) for item in dataset)
    unanswerable_count = len(dataset) - answerable_count

    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document_recursive(
            document,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
        )
    ]

    if not chunks:
        raise SystemExit("No chunks generated from data/raw")

    chunk_counts = {}
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        chunk_counts[source] = chunk_counts.get(source, 0) + 1

    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks per source: {chunk_counts}")

    print("\nLoading CrossEncoder reranker...")
    reranker = CrossEncoderReranker()

    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name=f"day12_expanded_{uuid4().hex[:8]}",
    )

    candidate_k = min(CANDIDATE_K, len(chunks))
    if candidate_k < TOP_K:
        raise SystemExit(f"Need at least {TOP_K} chunks, found {len(chunks)}")

    metrics = evaluate_retrieval(
        retriever=retriever,
        dataset=dataset,
        top_k=TOP_K,
        reranker=reranker,
        candidate_k=candidate_k,
    )

    results = {
        "dataset": DATASET_PATH.name,
        "corpus": "current_data_raw",
        "chunking": {
            "strategy": "recursive",
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "total_chunks": len(chunks),
            "chunks_per_source": chunk_counts,
        },
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "notes": [
            "Retrieval relevance is assessed using expected source filenames.",
            (
                "Unanswerable retrieval metrics do not measure "
                "generated-answer abstention."
            ),
            "No existing baseline or Day 11 result file is overwritten.",
        ],
        "retrieval_metrics": metrics,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    run_id = log_retrieval_experiment(
        results=metrics,
        dataset_name=DATASET_PATH.name,
        corpus_name=(f"raw_recursive_size_{CHUNK_SIZE}_overlap_{OVERLAP}"),
    )

    print("\nRetrieval metrics:")
    for mode, result in metrics.items():
        print(f"\n{mode}")
        print(f"  Hit Rate@{TOP_K}: {result['hit_rate_at_k']:.3f}")
        print(f"  Recall@{TOP_K}: {result['recall_at_k']:.3f}")
        print(f"  MRR: {result['mrr']:.3f}")
        print(f"  All expected sources found: {result['all_sources_recall_at_k']:.3f}")
        print(
            "  Unanswerable empty retrieval rate: "
            f"{result['unanswerable_empty_retrieval_rate']:.3f}"
        )
        print(
            "  Unanswerable returned-results rate: "
            f"{result['unanswerable_returned_results_rate']:.3f}"
        )

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"MLflow run ID: {run_id}")
    print("\nDay 12 expanded retrieval evaluation completed.")


if __name__ == "__main__":
    main()
