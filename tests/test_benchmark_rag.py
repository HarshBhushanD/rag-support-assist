import pytest

from scripts.benchmark_rag import estimate_cost


def test_estimate_cost_for_1000_queries():
    result = estimate_cost(
        queries=1000,
        hourly_rate=2.0,
        mean_latency_ms=3600.0,
    )

    assert result["estimated_compute_hours"] == pytest.approx(1.0)
    assert result["estimated_cost"] == pytest.approx(2.0)


def test_estimate_cost_scales_with_query_count():
    result = estimate_cost(
        queries=500,
        hourly_rate=4.0,
        mean_latency_ms=720.0,
    )

    assert result["estimated_compute_hours"] == pytest.approx(0.1)
    assert result["estimated_cost"] == pytest.approx(0.4)


def test_estimate_cost_without_latency():
    result = estimate_cost(
        queries=100,
        hourly_rate=2.0,
        mean_latency_ms=None,
    )

    assert result["estimated_compute_hours"] is None
    assert result["estimated_cost"] is None


def test_estimate_cost_with_zero_hourly_rate():
    result = estimate_cost(
        queries=1000,
        hourly_rate=0.0,
        mean_latency_ms=3600.0,
    )

    assert result["estimated_compute_hours"] == pytest.approx(1.0)
    assert result["estimated_cost"] == pytest.approx(0.0)
    assert "not necessarily free" in result["note"]
