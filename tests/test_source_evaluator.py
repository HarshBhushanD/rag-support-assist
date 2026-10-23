from src.evaluation.source_evaluator import source_was_retrieved


def test_source_hit_when_expected_source_is_present():
    assert (
        source_was_retrieved(
            "auth-help.md",
            ["auth-help.md", "troubleshooting.md"],
        )
        is True
    )


def test_source_miss_when_expected_source_is_absent():
    assert (
        source_was_retrieved(
            "auth-help.md",
            ["troubleshooting.md"],
        )
        is False
    )


def test_source_metric_is_not_applicable_without_expected_source():
    assert (
        source_was_retrieved(
            None,
            ["troubleshooting.md"],
        )
        is None
    )


def test_source_hit_with_multiple_retrieved_sources():
    assert (
        source_was_retrieved(
            "api-help.html",
            ["auth-help.md", "troubleshooting.md", "api-help.html"],
        )
        is True
    )


def test_unrelated_source_does_not_count_as_a_hit():
    assert (
        source_was_retrieved(
            "api-help.html",
            ["auth-help.md"],
        )
        is False
    )
