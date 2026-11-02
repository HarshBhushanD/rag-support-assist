import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import evaluate_retrieval
from src.evaluation.evidence_evaluator import evaluate_evidence
from src.evaluation.mlflow_tracker import log_retrieval_experiment
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


def evaluate_evidence_dataset(
    retriever: RetrievalEngine,
    dataset: list[dict[str, Any]],
    top_k: int = 6,
) -> dict[str, Any]:
    """Evaluate evidence sufficiency for answerable and unanswerable questions."""

    results = []

    for item in dataset:
        retrieved = retriever.search_hybrid(
            item["question"],
            top_k=top_k,
        )

        context = "\n".join(
            chunk.get("text", "")
            for chunk in retrieved
            if isinstance(chunk.get("text"), str)
        )

        if item["id"] == "q015":
            print("\n--- Q015 RETRIEVED CONTEXT ---")
            print(context)
            print("--- END Q015 CONTEXT ---\n")

        evidence = evaluate_evidence(
            required_evidence=item["required_evidence"],
            context=context,
        )

        answerable = item["answerable"]

        if answerable:
            correct = evidence["sufficient"]
            status = "evidence_sufficient" if correct else "evidence_insufficient"
        else:
            correct = not evidence["supported_evidence"]
            status = (
                "requested_evidence_absent" if correct else "requested_evidence_found"
            )

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": answerable,
                "evidence_coverage": evidence["evidence_coverage"],
                "supported_evidence": evidence["supported_evidence"],
                "missing_evidence": evidence["missing_evidence"],
                "status": status,
                "correct": correct,
            }
        )

    answerable_results = [item for item in results if item["answerable"]]

    unanswerable_results = [item for item in results if not item["answerable"]]

    answerable_correct = sum(item["correct"] for item in answerable_results)

    unanswerable_correct = sum(item["correct"] for item in unanswerable_results)

    return {
        "retrieval_mode": "hybrid",
        "top_k": top_k,
        "queries_evaluated": len(results),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": len(unanswerable_results),
        "answerable_evidence_sufficiency": (
            answerable_correct / len(answerable_results) if answerable_results else 0.0
        ),
        "unanswerable_evidence_absence_rate": (
            unanswerable_correct / len(unanswerable_results)
            if unanswerable_results
            else 0.0
        ),
        "results": results,
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

    print("\n--- Evidence Sufficiency Evaluation ---")

    evidence_metrics = evaluate_evidence_dataset(
        retriever=retriever,
        dataset=dataset,
        top_k=2,
    )

    print(
        "Answerable evidence sufficiency: "
        f"{evidence_metrics['answerable_evidence_sufficiency']:.1%}"
    )

    print(
        "Unanswerable evidence absence rate: "
        f"{evidence_metrics['unanswerable_evidence_absence_rate']:.1%}"
    )
    print("\n--- Reranker Threshold Calibration (Score Only) ---")

    threshold_summaries = {}

    for threshold in RERANKER_THRESHOLDS:
        print(f"\nEvaluating threshold: {threshold:.1f}")

        threshold_results = []

        for item in dataset:
            candidates = retriever.search_hybrid(
                item["question"],
                top_k=6,
            )

            reranked = reranker.rerank(
                item["question"],
                candidates,
                top_k=2,
            )

            scores = [
                result["reranker_score"]
                for result in reranked
                if "reranker_score" in result
            ]

            best_score = max(scores) if scores else float("-inf")

            # A score above the threshold predicts that
            # sufficient retrieval evidence is available.
            predicted_answerable = best_score >= threshold

            abstained = not predicted_answerable

            entry = {
                "id": item["id"],
                "question": item["question"],
                "threshold": threshold,
                "best_reranker_score": best_score,
                "answerable": item["answerable"],
                "predicted_answerable": predicted_answerable,
                "abstained": abstained,
                "decision_correct": (predicted_answerable == item["answerable"]),
            }

            threshold_results.append(entry)

            print(
                f"{item['id']}: "
                f"score={best_score:.4f}, "
                f"answerable={item['answerable']}, "
                f"predicted_answerable={predicted_answerable}"
            )

        summary = summarize_abstention(
            [
                {
                    "answerable": item["answerable"],
                    "abstained": item["abstained"],
                }
                for item in threshold_results
            ]
        )

        threshold_summaries[f"threshold_{threshold:.1f}"] = {
            **summary,
            "results": threshold_results,
        }

        print(f"\nSummary for threshold {threshold:.1f}:")
        print(
            "Answerable questions incorrectly rejected: "
            f"{summary['answerable_false_abstentions']}"
        )
        print(
            "Unanswerable questions incorrectly accepted: "
            f"{summary['unanswerable_false_acceptances']}"
        )
        print(f"Decision accuracy: {summary['decision_accuracy']:.1%}")
        print(f"Answerable coverage: {summary['answerable_coverage']:.1%}")

    output = {
        "dataset": "golden_dataset.json",
        "corpus": "evaluation_corpus.json",
        "retrieval_metrics": retrieval_results,
        "evidence_metrics": evidence_metrics,
        "threshold_calibration": threshold_summaries,
        "thresholds_tested": RERANKER_THRESHOLDS,
        "notes": [
            "This is a small controlled evaluation corpus.",
            "Threshold calibration uses reranker scores without generation.",
            "A score above the threshold predicts answerability.",
            "Decision accuracy does not measure factual correctness.",
            "Reranker scores are relevance signals, not calibrated probabilities.",
            "Evidence evaluation uses lexical matching, not semantic verification.",
        ],
    }

    output_path = EVAL_DIR / "baseline_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("Review threshold decisions against the evaluation dataset.")


if __name__ == "__main__":
    main()
