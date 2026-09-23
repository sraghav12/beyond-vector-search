import pytest

from pipelines.base import compute_cost


def test_dated_provider_model_name_preserves_known_price():
    assert compute_cost("gpt-4o-2024-08-06", 1000, 100) == pytest.approx(0.0035)
    assert compute_cost("openai/gpt-4o-mini-2024-07-18", 1000, 100) == pytest.approx(0.00021)
