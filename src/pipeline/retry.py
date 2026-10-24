from collections.abc import Callable
from time import sleep
from typing import TypeVar

T = TypeVar("T")


def run_with_retries(
    operation: Callable[[], T],
    *,
    retryable_exceptions: tuple[type[Exception], ...],
    max_attempts: int = 2,
    delay_seconds: float = 0.1,
) -> T:
    """Run an operation, retrying only explicitly selected exceptions."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    if not retryable_exceptions or not all(
        isinstance(exc, type) and issubclass(exc, Exception)
        for exc in retryable_exceptions
    ):
        raise ValueError("retryable_exceptions must contain exception classes")

    for attempt in range(max_attempts):
        try:
            return operation()
        except retryable_exceptions:
            if attempt == max_attempts - 1:
                raise
            sleep(delay_seconds)

    raise RuntimeError("Unreachable retry state")
