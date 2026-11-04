import json

import pytest

from src.evaluation import faithfulness_evaluator as evaluator


class MockJudge:
    """Mock the LLM judge without loading a real model."""

    def __init__(self, response):
        self.response = response

    def __call__(self, messages, **kwargs):
        return [{"generated_text": self.response}]


def mock_judge(monkeypatch, claims):
    """Replace the real judge with a predictable mocked response."""
    response = json.dumps({"claims": claims})

    monkeypatch.setattr(
        evaluator,
        "load_faithfulness_judge",
        lambda: MockJudge(response),
    )


def test_evaluates_supported_and_unsupported_claims(monkeypatch):
    mock_judge(
        monkeypatch,
        [
            {
                "claim": "Verify the database hostname.",
                "verdict": "supported",
                "reason": "The context explicitly mentions the hostname.",
            },
            {
                "claim": "The database uses port 5432.",
                "verdict": "unsupported",
                "reason": "The context does not specify this port.",
            },
        ],
    )

    result = evaluator.evaluate_faithfulness(
        answer="Verify the database hostname. The database uses port 5432.",
        context_chunks=[
            {
                "source": "troubleshooting.md",
                "text": "Verify the database hostname and credentials.",
            }
        ],
    )

    assert result["faithfulness_score"] == pytest.approx(0.5)
    assert result["total_claims"] == 2
    assert result["supported_claims"] == 1
    assert result["unsupported_claims"] == 1
    assert result["status"] == "evaluated"


def test_returns_perfect_score_for_supported_claims(monkeypatch):
    mock_judge(
        monkeypatch,
        [
            {
                "claim": "Verify the hostname.",
                "verdict": "supported",
                "reason": "The context mentions the hostname.",
            },
        ],
    )

    result = evaluator.evaluate_faithfulness(
        answer="Verify the hostname.",
        context_chunks=[{"text": "Verify the hostname before connecting."}],
    )

    assert result["faithfulness_score"] == pytest.approx(1.0)
    assert result["supported_claims"] == 1
    assert result["unsupported_claims"] == 0


def test_returns_zero_score_for_unsupported_claims(monkeypatch):
    mock_judge(
        monkeypatch,
        [
            {
                "claim": "Call support at 1234567890.",
                "verdict": "unsupported",
                "reason": "No support phone number is provided.",
            },
        ],
    )

    result = evaluator.evaluate_faithfulness(
        answer="Call support at 1234567890.",
        context_chunks=[{"text": "The documentation covers database troubleshooting."}],
    )

    assert result["faithfulness_score"] == pytest.approx(0.0)
    assert result["unsupported_claims"] == 1


def test_counts_insufficient_evidence_claims(monkeypatch):
    mock_judge(
        monkeypatch,
        [
            {
                "claim": "The database uses PostgreSQL.",
                "verdict": "insufficient_evidence",
                "reason": "The context does not identify the database engine.",
            },
        ],
    )

    result = evaluator.evaluate_faithfulness(
        answer="The database uses PostgreSQL.",
        context_chunks=[{"text": "Check the database connection."}],
    )

    assert result["insufficient_evidence_claims"] == 1
    assert result["faithfulness_score"] == pytest.approx(0.0)


def test_returns_no_context_status_without_loading_judge(monkeypatch):
    def fail_if_called():
        pytest.fail("Judge should not load when context is empty")

    monkeypatch.setattr(
        evaluator,
        "load_faithfulness_judge",
        fail_if_called,
    )

    result = evaluator.evaluate_faithfulness(
        answer="Verify the hostname.",
        context_chunks=[],
    )

    assert result["status"] == "no_context"
    assert result["faithfulness_score"] is None
    assert result["total_claims"] == 0


def test_returns_no_factual_claims_status(monkeypatch):
    mock_judge(monkeypatch, [])

    result = evaluator.evaluate_faithfulness(
        answer="Hello, how can I help?",
        context_chunks=[{"text": "Support documentation."}],
    )

    assert result["status"] == "no_factual_claims"
    assert result["faithfulness_score"] is None
    assert result["total_claims"] == 0


def test_rejects_empty_answer():
    with pytest.raises(ValueError, match="answer must not be empty"):
        evaluator.evaluate_faithfulness(
            answer="",
            context_chunks=[{"text": "Some context."}],
        )


def test_rejects_non_string_answer():
    with pytest.raises(TypeError, match="answer must be a string"):
        evaluator.evaluate_faithfulness(
            answer=None,
            context_chunks=[{"text": "Some context."}],
        )


def test_rejects_non_positive_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        evaluator.evaluate_faithfulness(
            answer="Some answer.",
            context_chunks=[{"text": "Some context."}],
            max_new_tokens=0,
        )


def test_rejects_invalid_judge_verdict(monkeypatch):
    mock_judge(
        monkeypatch,
        [
            {
                "claim": "Some claim.",
                "verdict": "maybe",
                "reason": "Uncertain.",
            },
        ],
    )

    with pytest.raises(ValueError, match="Invalid verdict"):
        evaluator.evaluate_faithfulness(
            answer="Some answer.",
            context_chunks=[{"text": "Some context."}],
        )


def test_rejects_invalid_judge_json(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "load_faithfulness_judge",
        lambda: MockJudge("This is not valid JSON."),
    )

    with pytest.raises(ValueError, match="no valid JSON object"):
        evaluator.evaluate_faithfulness(
            answer="Some answer.",
            context_chunks=[{"text": "Some context."}],
        )
