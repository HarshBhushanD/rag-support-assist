import re
from statistics import mean
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize text for deterministic lexical comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def content_tokens(text: str) -> list[str]:
    """Return normalized meaningful tokens, preserving negation and order."""
    return [
        token
        for token in normalize_text(text).split()
        if token not in STOPWORDS or token in {"no", "not", "nor"}
    ]


def score_answer(
    answer: str,
    expected_answer: str,
) -> dict[str, float | bool]:
    """Measure lexical overlap with an expected answer.

    These metrics are proxies for answer coverage, not semantic
    correctness or proof that an answer is grounded.
    """
    normalized_answer = normalize_text(answer)
    normalized_expected = normalize_text(expected_answer)

    answer_tokens = content_tokens(answer)
    expected_tokens = content_tokens(expected_answer)

    answer_token_set = set(answer_tokens)
    expected_token_set = set(expected_tokens)
    overlap = answer_token_set & expected_token_set

    precision = len(overlap) / len(answer_token_set) if answer_token_set else 0.0
    recall = len(overlap) / len(expected_token_set) if expected_token_set else 0.0

    # Penalize a negation mismatch instead of letting shared words hide it.
    answer_negations = {"no", "not", "nor"} & answer_token_set
    expected_negations = {"no", "not", "nor"} & expected_token_set
    if answer_negations != expected_negations:
        recall = min(recall, 0.99)

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "exact_match": normalized_answer == normalized_expected,
        "expected_answer_coverage": recall,
        "answer_token_precision": precision,
        "token_f1": f1,
    }


def evaluate_answer_quality(
    dataset: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate answer coverage and abstention decisions."""
    dataset_by_id = {item["id"]: item for item in dataset}
    results_by_id = {item["id"]: item for item in results}

    if len(dataset_by_id) != len(dataset):
        raise ValueError("Dataset contains duplicate question IDs")

    if len(results_by_id) != len(results):
        raise ValueError("Results contain duplicate question IDs")

    missing_ids = dataset_by_id.keys() - results_by_id.keys()
    if missing_ids:
        raise ValueError(f"Missing results for question IDs: {sorted(missing_ids)}")

    extra_ids = results_by_id.keys() - dataset_by_id.keys()
    if extra_ids:
        raise ValueError(f"Unexpected result IDs: {sorted(extra_ids)}")

    details: list[dict[str, Any]] = []
    scored_answers: list[dict[str, float | bool]] = []

    answerable_count = 0
    answerable_answered = 0
    answerable_false_abstentions = 0
    unanswerable_count = 0
    unanswerable_correctly_abstained = 0
    unanswerable_false_acceptances = 0

    for item in dataset:
        result = results_by_id[item["id"]]
        answer = result.get("answer", "")
        abstained = bool(result.get("abstained", False))
        answerable = bool(item["answerable"])

        detail: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "answerable": answerable,
            "abstained": abstained,
            "abstention_correct": abstained == (not answerable),
        }

        if answerable:
            answerable_count += 1

            if abstained:
                answerable_false_abstentions += 1
            else:
                answerable_answered += 1
                scores = score_answer(answer, item["expected_answer"])
                detail.update(scores)
                scored_answers.append(scores)
        else:
            unanswerable_count += 1

            if abstained:
                unanswerable_correctly_abstained += 1
            else:
                unanswerable_false_acceptances += 1

        details.append(detail)

    total = len(dataset)
    scored_count = len(scored_answers)

    return {
        "queries_evaluated": total,
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "answerable_questions_answered": answerable_answered,
        "answerable_false_abstentions": answerable_false_abstentions,
        "unanswerable_questions_correctly_abstained": (
            unanswerable_correctly_abstained
        ),
        "unanswerable_false_acceptances": unanswerable_false_acceptances,
        "answerable_response_rate": (
            answerable_answered / answerable_count if answerable_count else 0.0
        ),
        "unanswerable_abstention_accuracy": (
            unanswerable_correctly_abstained / unanswerable_count
            if unanswerable_count
            else 0.0
        ),
        "mean_expected_answer_coverage": (
            mean(item["expected_answer_coverage"] for item in scored_answers)
            if scored_count
            else 0.0
        ),
        "mean_answer_token_precision": (
            mean(item["answer_token_precision"] for item in scored_answers)
            if scored_count
            else 0.0
        ),
        "mean_token_f1": (
            mean(item["token_f1"] for item in scored_answers) if scored_count else 0.0
        ),
        "answers_scored": scored_count,
        "notes": [
            "Lexical overlap is a proxy, not a semantic correctness metric.",
            "High token overlap does not prove that an answer is grounded.",
            "Review answer correctness and groundedness manually.",
        ],
        "details": details,
    }
