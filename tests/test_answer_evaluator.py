import pytest

from src.evaluation.answer_evaluator import (
    evaluate_answer_quality,
    score_answer,
)


def test_score_answer_detects_incomplete_answer():
    scores = score_answer(
        "Account Settings",
        "Open Account Settings and select Security.",
    )

    assert scores["exact_match"] is False
    assert scores["expected_answer_coverage"] == pytest.approx(0.4)
    assert scores["token_f1"] < 1.0


def test_score_answer_handles_punctuation_and_case():
    scores = score_answer(
        "VERIFY the hostname, port, credentials, and network connectivity!",
        "Verify the hostname, port, credentials, and network connectivity.",
    )

    assert scores["exact_match"] is True
    assert scores["expected_answer_coverage"] == pytest.approx(1.0)
    assert scores["answer_token_precision"] == pytest.approx(1.0)
    assert scores["token_f1"] == pytest.approx(1.0)


def test_score_answer_retains_negation():
    scores = score_answer(
        "Support is not available",
        "Support is available",
    )

    assert scores["expected_answer_coverage"] < 1.0
    assert scores["answer_token_precision"] < 1.0


def test_evaluator_summarizes_answerable_and_unanswerable_questions():
    dataset = [
        {
            "id": "q001",
            "question": "How do I reset my password?",
            "answerable": True,
            "expected_answer": "Open Account Settings and select Security.",
        },
        {
            "id": "q007",
            "question": "What is the support phone number?",
            "answerable": False,
            "expected_answer": (
                "The provided documents do not specify a support phone number."
            ),
        },
    ]

    results = [
        {
            "id": "q001",
            "answer": "Open Account Settings and select Security.",
            "abstained": False,
        },
        {
            "id": "q007",
            "answer": (
                "I don't have enough information in the provided documents "
                "to answer that question."
            ),
            "abstained": True,
        },
    ]

    summary = evaluate_answer_quality(dataset, results)

    assert summary["queries_evaluated"] == 2
    assert summary["answerable_questions_answered"] == 1
    assert summary["unanswerable_questions_correctly_abstained"] == 1
    assert summary["answerable_response_rate"] == pytest.approx(1.0)
    assert summary["unanswerable_abstention_accuracy"] == pytest.approx(1.0)
    assert summary["mean_expected_answer_coverage"] == pytest.approx(1.0)
    assert summary["mean_token_f1"] == pytest.approx(1.0)


def test_evaluator_rejects_missing_results():
    dataset = [
        {
            "id": "q001",
            "question": "Question?",
            "answerable": True,
            "expected_answer": "Expected answer",
        }
    ]

    with pytest.raises(ValueError, match="Missing results"):
        evaluate_answer_quality(dataset, [])


def test_evaluator_handles_no_answerable_questions():
    dataset = [
        {
            "id": "q007",
            "question": "What is the support phone number?",
            "answerable": False,
            "expected_answer": "Not specified.",
        }
    ]
    results = [{"id": "q007", "answer": "No information.", "abstained": True}]

    summary = evaluate_answer_quality(dataset, results)

    assert summary["answerable_questions"] == 0
    assert summary["answers_scored"] == 0
    assert summary["mean_token_f1"] == 0.0
    assert summary["unanswerable_abstention_accuracy"] == pytest.approx(1.0)
