import json
from pathlib import Path
from typing import Any

from src.evaluation.groundedness_evaluator import evaluate_groundedness
from src.evaluation.source_evaluator import source_was_retrieved
from src.ingest.chunker import chunk_document_recursive
from src.ingest.loaders import load_documents
from src.pipeline.rag_pipeline import FALLBACK_ANSWER, RAGPipeline
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
OUTPUT_PATH = EVAL_DIR / "real_corpus_results.json"

QUESTIONS = [
    {
        "id": "R001",
        "question": "How do I troubleshoot a database connection?",
        "answerable": True,
        "expected_source": "troubleshooting.md",
    },
    {
        "id": "R002",
        "question": "What should I check when an API request fails?",
        "answerable": True,
        "expected_source": "api-help.html",
    },
    {
        "id": "R003",
        "question": "What should I check when authentication fails?",
        "answerable": True,
        "expected_source": "auth-help.md",
    },
    {
        "id": "R004",
        "question": "What are the support team's working hours?",
        "answerable": False,
        "expected_source": None,
    },
    {
        "id": "R005",
        "question": "What is the support team's phone number?",
        "answerable": False,
        "expected_source": None,
    },
    {
        "id": "R006",
        "question": "Is support available on Saturday?",
        "answerable": False,
        "expected_source": None,
    },
]


def is_abstained(answer: str) -> bool:
    """Detect the canonical fallback, allowing missing whitespace."""
    normalized = " ".join(answer.casefold().split())
    fallback = " ".join(FALLBACK_ANSWER.casefold().split())

    return normalized.replace(" ", "") == fallback.replace(" ", "")


def main() -> None:
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

    if not chunks:
        raise SystemExit(f"No document chunks found in {RAW_DIR}")

    print(f"Loaded {len(documents)} documents and {len(chunks)} chunks.")
    print("Building real-corpus retrieval engine...")

    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="real_corpus_evaluation",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        retrieval_mode="hybrid",
        top_k=3,
        max_new_tokens=160,
    )

    results: list[dict[str, Any]] = []

    for item in QUESTIONS:
        result = pipeline.ask(item["question"])
        answer = result["answer"]
        abstained = is_abstained(answer)
        retrieved_sources = result["sources"]
        retrieved_chunks = result.get("retrieved_chunks", [])

        retrieved_context = " ".join(
            str(chunk.get("text", ""))
            for chunk in retrieved_chunks
            if str(chunk.get("text", "")).strip()
        )

        if abstained:
            groundedness: dict[str, bool | str | float | None] = {
                "grounded": None,
                "reason": "abstained",
                "token_coverage": None,
            }
        else:
            groundedness = evaluate_groundedness(answer, retrieved_context)

        expected_source_found = source_was_retrieved(
            item["expected_source"],
            retrieved_sources,
        )

        entry = {
            **item,
            "answer": answer,
            "abstained": abstained,
            "abstention_correct": abstained == (not item["answerable"]),
            "retrieved_sources": retrieved_sources,
            "expected_source_found": expected_source_found,
            "grounded": groundedness["grounded"],
            "groundedness_reason": groundedness["reason"],
            "groundedness_token_coverage": groundedness["token_coverage"],
            "manual_review_required": True,
        }
        results.append(entry)

        print(f"\n{'=' * 70}")
        print(f"{item['id']}: {item['question']}")
        print(f"Answer: {answer}")
        print(f"Abstained: {abstained}")
        print(f"Sources: {retrieved_sources}")

        if expected_source_found is not None:
            print(f"Expected source found: {expected_source_found}")

        print(f"Grounded: {groundedness['grounded']}")
        print(f"Groundedness reason: {groundedness['reason']}")
        print(f"Abstention decision correct: {entry['abstention_correct']}")

    total = len(results)
    correct_decisions = sum(result["abstention_correct"] for result in results)
    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]

    grounded_answers = [result for result in results if result["grounded"] is not None]
    grounded_count = sum(result["grounded"] is True for result in grounded_answers)

    groundedness_rate = (
        grounded_count / len(grounded_answers) if grounded_answers else 0.0
    )

    summary = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "queries_evaluated": total,
        "correct_abstention_decisions": correct_decisions,
        "decision_accuracy": correct_decisions / total if total else 0.0,
        "answerable_questions_answered": sum(
            not result["abstained"] for result in answerable
        ),
        "unanswerable_questions_correctly_abstained": sum(
            result["abstained"] for result in unanswerable
        ),
        "answers_evaluated_for_groundedness": len(grounded_answers),
        "grounded_answers": grounded_count,
        "groundedness_rate": groundedness_rate,
        "groundedness_method": (
            "Content-token coverage (threshold 0.8); not semantic entailment."
        ),
        "warning": (
            "Decision accuracy measures abstention behavior, not factual "
            "correctness. Groundedness is a lexical baseline, not proof of "
            "factual correctness. Review answers and sources manually."
        ),
    }

    output = {"summary": summary, "results": results}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
