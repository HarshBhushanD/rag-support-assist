import json
from pathlib import Path

from src.evaluation.evaluator import evaluate_retrieval
from src.evaluation.mlflow_tracker import log_retrieval_experiment
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    dataset = load_json(EVAL_DIR / "golden_dataset.json")
    chunks = load_json(EVAL_DIR / "evaluation_corpus.json")

    print("Building evaluation retrieval engine...")
    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="day7_evaluation",
    )

    print("\nEvaluating retrieval modes...")
    retrieval_results = evaluate_retrieval(
        retriever=retriever,
        dataset=dataset,
        top_k=2,
    )

    print("\n--- Retrieval Metrics ---")
    for mode, metrics in retrieval_results.items():
        print(f"\nMode: {mode}")
        print(f"Queries evaluated: {metrics['queries_evaluated']}")
        print(f"Hit Rate@{metrics['top_k']}: {metrics['hit_rate_at_k']:.3f}")
        print(f"Recall@{metrics['top_k']}: {metrics['recall_at_k']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")

    run_id = log_retrieval_experiment(
        results=retrieval_results,
        dataset_name="golden_dataset.json",
        corpus_name="evaluation_corpus.json",
    )
    print(f"\nMLflow run ID: {run_id}")

    pipeline = RAGPipeline(
        retriever=retriever,
        retrieval_mode="hybrid",
        top_k=2,
        max_new_tokens=100,
    )

    answer_results = []

    print("\nGenerating hybrid RAG answers...")
    for item in dataset:
        print(f"\nEvaluating {item['id']}: {item['question']}")
        result = pipeline.ask(item["question"])

        fallback = (
            "i don't have enough information in the provided documents "
            "to answer that question."
        )
        abstained = fallback in result["answer"].lower()

        answer_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answerable": item["answerable"],
                "answer": result["answer"],
                "expected_sources": item["expected_sources"],
                "retrieved_sources": result["sources"],
                "abstained": abstained,
                "abstention_correct": (abstained == (not item["answerable"])),
                "manual_review_required": True,
            }
        )

        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")

    output = {
        "dataset": "golden_dataset.json",
        "corpus": "evaluation_corpus.json",
        "retrieval_metrics": retrieval_results,
        "answers": answer_results,
        "notes": [
            "This is a small controlled evaluation corpus.",
            "Answer correctness and groundedness require manual review.",
            "Abstention correctness uses the exact configured fallback phrase.",
        ],
    }

    output_path = EVAL_DIR / "baseline_results.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("Review answer correctness and groundedness manually.")


if __name__ == "__main__":
    main()
