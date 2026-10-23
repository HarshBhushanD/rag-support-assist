def source_was_retrieved(
    expected_source: str | None,
    retrieved_sources: list[str],
) -> bool | None:
    """Check whether the expected source appears in retrieved sources."""
    if expected_source is None:
        return None

    return expected_source in retrieved_sources
