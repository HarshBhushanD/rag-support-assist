import pytest

from src.evaluation.evidence_evaluator import evaluate_evidence


def test_accepts_sufficient_evidence():
    result = evaluate_evidence(
        required_evidence=[
            "Account Settings",
            "select Security",
            "click Reset Password",
        ],
        context=(
            "To reset your password, open Account Settings, "
            "select Security, and click Reset Password."
        ),
    )

    assert result["sufficient"] is True
    assert result["evidence_coverage"] == pytest.approx(1.0)
    assert result["missing_evidence"] == []


def test_rejects_missing_time_zone_evidence():
    result = evaluate_evidence(
        required_evidence=["support schedule time zone"],
        context="Support hours are Monday to Friday, 9 AM to 5 PM.",
    )

    assert result["sufficient"] is False
    assert result["reason"] == "missing_required_evidence"
    assert result["evidence_coverage"] == 0.0
    assert result["missing_evidence"] == ["support schedule time zone"]


def test_reports_partial_evidence():
    result = evaluate_evidence(
        required_evidence=[
            "database hostname",
            "database port",
            "database credentials",
            "network connectivity",
        ],
        context=("Verify the database hostname, port, and network connectivity."),
    )

    assert result["sufficient"] is False
    assert result["evidence_coverage"] == pytest.approx(0.75)
    assert result["missing_evidence"] == ["database credentials"]


def test_normalizes_case_and_punctuation():
    result = evaluate_evidence(
        required_evidence=["CLICK RESET PASSWORD"],
        context="Click Reset Password!",
    )

    assert result["sufficient"] is True
    assert result["evidence_coverage"] == 1.0


def test_rejects_empty_context():
    result = evaluate_evidence(
        required_evidence=["Account Settings"],
        context="",
    )

    assert result["sufficient"] is False
    assert result["reason"] == "empty_context"


def test_rejects_empty_requirements():
    result = evaluate_evidence(
        required_evidence=[],
        context="Some retrieved document content.",
    )

    assert result["sufficient"] is False
    assert result["reason"] == "no_required_evidence"


def test_rejects_unrelated_evidence():
    result = evaluate_evidence(
        required_evidence=["support phone number"],
        context="Support hours are Monday to Friday.",
    )

    assert result["sufficient"] is False
    assert result["evidence_coverage"] == 0.0


def test_does_not_accept_partial_phrase_match():
    result = evaluate_evidence(
        required_evidence=["network connectivity"],
        context="Check network configuration.",
    )

    assert result["sufficient"] is False
    assert result["missing_evidence"] == ["network connectivity"]


def test_rejects_database_credentials_partial_match():
    result = evaluate_evidence(
        required_evidence=["database credentials"],
        context="Verify your credentials before connecting.",
    )

    assert result["sufficient"] is False
    assert result["evidence_coverage"] == 0.0
    assert result["missing_evidence"] == ["database credentials"]


def test_accepts_explicit_database_port_alias():
    result = evaluate_evidence(
        required_evidence=["database port"],
        context="Verify the database hostname, port, and network connectivity.",
    )

    assert result["sufficient"] is True


def test_rejects_reversed_word_order():
    result = evaluate_evidence(
        required_evidence=["network connectivity"],
        context="Check connectivity across the network.",
    )

    assert result["sufficient"] is False


def test_rejects_negated_evidence():
    result = evaluate_evidence(
        required_evidence=["database connectivity"],
        context="Database connectivity is not available.",
    )

    assert result["sufficient"] is False


def test_rejects_evidence_with_no_prefix():
    result = evaluate_evidence(
        required_evidence=["database port"],
        context="Verify the application port.",
    )

    assert result["sufficient"] is False


def test_accepts_database_credentials_in_connection_list():
    result = evaluate_evidence(
        required_evidence=["database credentials"],
        context=(
            "For database connection failures, verify the database "
            "hostname, port, credentials, and network connectivity."
        ),
    )

    assert result["sufficient"] is True


def test_rejects_credentials_without_database_context():
    result = evaluate_evidence(
        required_evidence=["database credentials"],
        context=(
            "Verify your credentials before logging in. "
            "The database hostname is configured separately."
        ),
    )

    assert result["sufficient"] is False


def test_rejects_unrelated_database_and_credentials_in_same_sentence():
    result = evaluate_evidence(
        required_evidence=["database credentials"],
        context=(
            "The database hostname is configured separately, "
            "while user credentials are used for application login."
        ),
    )

    assert result["sufficient"] is False
