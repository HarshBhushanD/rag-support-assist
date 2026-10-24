import pytest

from src.evaluation.groundedness_evaluator import evaluate_groundedness


def test_accepts_supported_answer_with_paraphrasing():
    result = evaluate_groundedness(
        answer="Verify the request URL, HTTP method, and authentication credentials.",
        context="Check the request URL, HTTP method, and authentication credentials.",
    )

    assert result["grounded"] is True
    assert result["reason"] == "sufficient_content_token_coverage"
    assert result["token_coverage"] == pytest.approx(6 / 7)


def test_rejects_unsupported_answer():
    result = evaluate_groundedness(
        answer="Call support at 1234567890.",
        context="The documentation covers database troubleshooting.",
    )

    assert result["grounded"] is False
    assert result["reason"] == "insufficient_content_token_coverage"
    assert result["token_coverage"] < 0.8


def test_normalizes_case_and_punctuation():
    result = evaluate_groundedness(
        answer="VERIFY the hostname and port!",
        context="Verify the hostname and port before connecting.",
    )

    assert result["grounded"] is True
    assert result["token_coverage"] == pytest.approx(1.0)


def test_rejects_empty_answer():
    result = evaluate_groundedness(
        answer="",
        context="Some context.",
    )

    assert result["grounded"] is False
    assert result["reason"] == "empty_answer"
    assert result["token_coverage"] == 0.0


def test_rejects_empty_context():
    result = evaluate_groundedness(
        answer="Some answer.",
        context="",
    )

    assert result["grounded"] is False
    assert result["reason"] == "empty_context"
    assert result["token_coverage"] == 0.0


def test_rejects_answer_with_insufficient_coverage():
    result = evaluate_groundedness(
        answer="Verify the hostname port credentials and restart the server.",
        context="Verify the hostname and port.",
    )

    assert result["grounded"] is False
    assert result["reason"] == "insufficient_content_token_coverage"
    assert result["token_coverage"] < 0.8


def test_supports_custom_coverage_threshold():
    result = evaluate_groundedness(
        answer="Check the hostname and restart the server.",
        context="Check the hostname.",
        min_token_coverage=0.5,
    )

    assert result["grounded"] is True
    assert result["token_coverage"] == pytest.approx(0.5)


def test_rejects_answer_containing_only_stopwords():
    result = evaluate_groundedness(
        answer="the and or",
        context="the and or",
    )

    assert result["grounded"] is False
    assert result["reason"] == "no_content_tokens"
    assert result["token_coverage"] == 0.0


def test_reports_partial_token_overlap():
    result = evaluate_groundedness(
        answer="Check hostname credentials",
        context="Check hostname",
    )

    assert result["token_coverage"] == pytest.approx(2 / 3)
    assert result["grounded"] is False
    assert result["reason"] == "insufficient_content_token_coverage"
