import pytest

from scripts.evaluate_rag import summarize_abstention


def test_summary_when_all_abstention_decisions_are_correct():
    results = [
        {"answerable": True, "abstained": False},
        {"answerable": True, "abstained": False},
        {"answerable": False, "abstained": True},
    ]

    summary = summarize_abstention(results)

    assert summary["queries_evaluated"] == 3
    assert summary["answerable_questions"] == 2
    assert summary["unanswerable_questions"] == 1
    assert summary["answerable_questions_answered"] == 2
    assert summary["answerable_false_abstentions"] == 0
    assert summary["unanswerable_questions_correctly_abstained"] == 1
    assert summary["unanswerable_false_acceptances"] == 0
    assert summary["correct_abstention_decisions"] == 3
    assert summary["decision_accuracy"] == pytest.approx(1.0)
    assert summary["answerable_coverage"] == pytest.approx(1.0)


def test_summary_counts_false_abstentions():
    results = [
        {"answerable": True, "abstained": True},
        {"answerable": True, "abstained": False},
        {"answerable": False, "abstained": True},
    ]

    summary = summarize_abstention(results)

    assert summary["answerable_false_abstentions"] == 1
    assert summary["answerable_questions_answered"] == 1
    assert summary["answerable_coverage"] == pytest.approx(0.5)
    assert summary["correct_abstention_decisions"] == 2
    assert summary["decision_accuracy"] == pytest.approx(2 / 3)


def test_summary_counts_false_acceptances():
    results = [
        {"answerable": True, "abstained": False},
        {"answerable": False, "abstained": False},
        {"answerable": False, "abstained": True},
    ]

    summary = summarize_abstention(results)

    assert summary["unanswerable_false_acceptances"] == 1
    assert summary["unanswerable_questions_correctly_abstained"] == 1
    assert summary["correct_abstention_decisions"] == 2
    assert summary["decision_accuracy"] == pytest.approx(2 / 3)


def test_summary_handles_no_answerable_questions():
    results = [
        {"answerable": False, "abstained": True},
        {"answerable": False, "abstained": False},
    ]

    summary = summarize_abstention(results)

    assert summary["answerable_questions"] == 0
    assert summary["answerable_coverage"] == 0.0
    assert summary["unanswerable_questions_correctly_abstained"] == 1
    assert summary["unanswerable_false_acceptances"] == 1
    assert summary["decision_accuracy"] == pytest.approx(0.5)


def test_summary_handles_empty_results():
    summary = summarize_abstention([])

    assert summary["queries_evaluated"] == 0
    assert summary["decision_accuracy"] == 0.0
    assert summary["answerable_coverage"] == 0.0
    assert summary["correct_abstention_decisions"] == 0
