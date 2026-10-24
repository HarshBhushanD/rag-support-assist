from unittest.mock import Mock

import pytest

from src.pipeline.retry import run_with_retries


def test_retries_transient_failure_then_succeeds():
    operation = Mock(
        side_effect=[
            TimeoutError("temporary failure"),
            "success",
        ]
    )

    result = run_with_retries(
        operation,
        retryable_exceptions=(TimeoutError,),
        max_attempts=2,
        delay_seconds=0,
    )

    assert result == "success"
    assert operation.call_count == 2


def test_does_not_retry_unclassified_failure():
    operation = Mock(side_effect=ValueError("invalid input"))

    with pytest.raises(ValueError, match="invalid input"):
        run_with_retries(
            operation,
            retryable_exceptions=(TimeoutError,),
            max_attempts=3,
            delay_seconds=0,
        )

    operation.assert_called_once()


def test_raises_after_retry_limit():
    operation = Mock(side_effect=TimeoutError("still failing"))

    with pytest.raises(TimeoutError, match="still failing"):
        run_with_retries(
            operation,
            retryable_exceptions=(TimeoutError,),
            max_attempts=2,
            delay_seconds=0,
        )

    assert operation.call_count == 2


def test_rejects_invalid_attempt_count():
    with pytest.raises(ValueError, match="max_attempts"):
        run_with_retries(
            lambda: "unused",
            retryable_exceptions=(TimeoutError,),
            max_attempts=0,
        )
