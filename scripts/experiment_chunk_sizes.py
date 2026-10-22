import json
from pathlib import Path
from uuid import uuid4

from src.evaluation.evaluator import evaluate_retrieval
from src.evaluation.mlflow_tracker import log_retrieval_experiment
from src.ingest.chunker import chunk_document_recursive
from src.ingest.loaders import load_documents
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"

CHUNK_CONFIGS = [
    {"chunk_size": 200, "overlap": 40},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 800, "overlap": 160},
]

TOP_K = 2
CANDIDATE_K = 6


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    dataset = load_json(EVAL_DIR / "chunk_size_dataset.json")
    documents = load_documents(ROOT / "data" / "raw")

    if not documents:
        raise SystemExit("No supported documents found in data/raw")

    print(f"Loaded {len(documents)} raw documents.")
    print(f"Evaluating {len(dataset)} questions.")

    print("\nLoading CrossEncoder reranker...")
    reranker = CrossEncoderReranker()

    experiment_results = {
        "dataset": "chunk_size_dataset.json",
        "chunking_strategy": "recursive",
        "top_k": TOP_K,
        "configurations": [],
        "notes": [
            "Uses the current data/raw documents.",
            "Expected relevance is based on source filenames.",
            "The dataset contains six answerable retrieval questions.",
            "Answer generation is not evaluated by this experiment.",
        ],
    }

    for config in CHUNK_CONFIGS:
        chunk_size = config["chunk_size"]
        overlap = config["overlap"]

        print(f"\n{'=' * 70}\nChunk size: {chunk_size} | Overlap: {overlap}")

        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document_recursive(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        ]

        if not chunks:
            raise SystemExit(f"No chunks generated for chunk_size={chunk_size}")

        chunk_counts = {}
        for chunk in chunks:
            source = chunk["source"]
            chunk_counts[source] = chunk_counts.get(source, 0) + 1

        print(f"Total chunks: {len(chunks)}")
        print(f"Chunks per source: {chunk_counts}")

        # A unique collection name avoids reusing stale indexed chunks
        # from a previous run of this experiment.
        collection_name = f"day11_{chunk_size}_{uuid4().hex[:8]}"

        retriever = RetrievalEngine(
            chunks=chunks,
            collection_name=collection_name,
        )

        candidate_k = min(CANDIDATE_K, len(chunks))

        if candidate_k < TOP_K:
            raise SystemExit("At least two chunks are required for top_k=2.")

        metrics = evaluate_retrieval(
            retriever=retriever,
            dataset=dataset,
            top_k=TOP_K,
            reranker=reranker,
            candidate_k=candidate_k,
        )

        corpus_name = f"raw_recursive_size_{chunk_size}_overlap_{overlap}"

        run_id = log_retrieval_experiment(
            results=metrics,
            dataset_name="chunk_size_dataset.json",
            corpus_name=corpus_name,
        )

        experiment_results["configurations"].append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": len(chunks),
                "chunks_per_source": chunk_counts,
                "retrieval_metrics": metrics,
                "mlflow_run_id": run_id,
            }
        )

        print("\nRetrieval metrics:")
        for mode, result in metrics.items():
            print(
                f"{mode:18} "
                f"Hit Rate@{TOP_K}={result['hit_rate_at_k']:.3f} | "
                f"Recall@{TOP_K}={result['recall_at_k']:.3f} | "
                f"MRR={result['mrr']:.3f}"
            )

        print(f"MLflow run ID: {run_id}")

    output_path = EVAL_DIR / "chunk_size_experiment_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            experiment_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nResults saved to: {output_path}")
    print("\nDay 11 chunk-size experiment completed.")


if __name__ == "__main__":
    main()
