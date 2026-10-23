import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import evaluate_retrieval
from src.evaluation.mlflow_tracker import log_retrieval_experiment
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"

RERANKER_THRESHOLDS = [1.5, 2.0, 2.5]

FALLBACK = (
    "i don't have enough information in the provided documents to answer that question."
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def is_abstained(answer: str) -> bool:
    return FALLBACK in answer.lower()


def summarize_abstention(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in results if item["answerable"]]
    unanswerable = [item for item in results if not item["answerable"]]

    answerable_answered = sum(not item["abstained"] for item in answerable)
    answerable_false_abstentions = sum(item["abstained"] for item in answerable)
    unanswerable_correctly_abstained = sum(item["abstained"] for item in unanswerable)
    unanswerable_false_acceptances = sum(not item["abstained"] for item in unanswerable)
    correct_decisions = answerable_answered + unanswerable_correctly_abstained

    total = len(results)

    return {
        "queries_evaluated": total,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "answerable_questions_answered": answerable_answered,
        "answerable_false_abstentions": answerable_false_abstentions,
        "unanswerable_questions_correctly_abstained": (
            unanswerable_correctly_abstained
        ),
        "unanswerable_false_acceptances": unanswerable_false_acceptances,
        "correct_abstention_decisions": correct_decisions,
        "decision_accuracy": (correct_decisions / total if total else 0.0),
        "answerable_coverage": (
            answerable_answered / len(answerable) if answerable else 0.0
        ),
    }


def main() -> None:
    dataset = load_json(EVAL_DIR / "golden_dataset.json")
    chunks = load_json(EVAL_DIR / "evaluation_corpus.json")

    print("Building evaluation retrieval engine...")
    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="day7_evaluation",
    )

    print("\nLoading CrossEncoder reranker...")
    reranker = CrossEncoderReranker()

    print("\nEvaluating retrieval modes...")
    retrieval_results = evaluate_retrieval(
        retriever=retriever,
        dataset=dataset,
        top_k=2,
        reranker=reranker,
        candidate_k=6,
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

    answer_results: list[dict[str, Any]] = []
    threshold_summaries: dict[str, dict[str, Any]] = {}

    print("\nCreating hybrid baseline pipeline...")
    hybrid_pipeline = RAGPipeline(
        retriever=retriever,
        retrieval_mode="hybrid",
        top_k=2,
        max_new_tokens=160,
    )

    print("\nEvaluating hybrid baseline...")
    hybrid_results = []

    for item in dataset:
        result = hybrid_pipeline.ask(item["question"])
        abstained = is_abstained(result["answer"])

        entry = {
            "id": item["id"],
            "question": item["question"],
            "mode": "hybrid",
            "expected_answer": item["expected_answer"],
            "answerable": item["answerable"],
            "answer": result["answer"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": result["sources"],
            "abstained": abstained,
            "abstention_correct": (abstained == (not item["answerable"])),
            "manual_review_required": True,
        }

        hybrid_results.append(entry)
        answer_results.append(entry)

        print(f"\nQuestion: {item['id']}")
        print(f"Answer: {result['answer']}")
        print(f"Abstained: {abstained}")

    threshold_summaries["hybrid"] = summarize_abstention(hybrid_results)

    print("\n--- Reranker Threshold Calibration ---")

    for threshold in RERANKER_THRESHOLDS:
        mode = f"reranked_hybrid_threshold_{threshold:.1f}"
        print(f"\nEvaluating threshold: {threshold:.1f}")

        pipeline = RAGPipeline(
            retriever=retriever,
            retrieval_mode="hybrid",
            top_k=2,
            max_new_tokens=160,
            reranker=reranker,
            candidate_k=6,
            min_reranker_score=threshold,
        )

        threshold_results = []

        for item in dataset:
            result = pipeline.ask(item["question"])
            abstained = is_abstained(result["answer"])

            entry = {
                "id": item["id"],
                "question": item["question"],
                "mode": mode,
                "threshold": threshold,
                "expected_answer": item["expected_answer"],
                "answerable": item["answerable"],
                "answer": result["answer"],
                "expected_sources": item["expected_sources"],
                "retrieved_sources": result["sources"],
                "abstained": abstained,
                "abstention_correct": (abstained == (not item["answerable"])),
                "manual_review_required": True,
            }

            threshold_results.append(entry)
            answer_results.append(entry)

            print(f"{item['id']}: abstained={abstained}, sources={result['sources']}")

        summary = summarize_abstention(threshold_results)
        threshold_summaries[mode] = summary

        print(f"\nSummary for threshold {threshold:.1f}:")
        print(
            "Answerable questions incorrectly rejected: "
            f"{summary['answerable_false_abstentions']}"
        )
        print(
            "Unanswerable questions incorrectly answered: "
            f"{summary['unanswerable_false_acceptances']}"
        )
        print(
            "Correct answer/abstain decisions: "
            f"{summary['correct_abstention_decisions']}/"
            f"{summary['queries_evaluated']}"
        )
        print(f"Answerable coverage: {summary['answerable_coverage']:.1%}")
        print(f"Decision accuracy: {summary['decision_accuracy']:.1%}")

    output = {
        "dataset": "golden_dataset.json",
        "corpus": "evaluation_corpus.json",
        "retrieval_metrics": retrieval_results,
        "abstention_summaries": threshold_summaries,
        "answers": answer_results,
        "thresholds_tested": RERANKER_THRESHOLDS,
        "notes": [
            "This is a small controlled evaluation corpus.",
            "Answer correctness and groundedness require manual review.",
            "Abstention detection checks for the configured fallback phrase.",
            "Decision metrics measure answering versus abstaining, "
            "not factual correctness or groundedness.",
            "Threshold results include both score-based rejection "
            "and any abstention performed by the generation pipeline.",
            "Reranker scores are relevance signals, not calibrated "
            "probabilities of answer correctness.",
        ],
    }

    output_path = EVAL_DIR / "baseline_results.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("Review answer correctness and groundedness manually.")


if __name__ == "__main__":
    main()
