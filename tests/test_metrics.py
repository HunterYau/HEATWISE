"""Scalar boundary checks for fixed, data-independent UTCI categories."""

from __future__ import annotations

from urban_heat_risk_ai.metrics import recall_at_utci_threshold, utci_heat_category


def test_fixed_heat_category_boundaries() -> None:
    assert utci_heat_category(25.999) == "no_heat_stress"
    assert utci_heat_category(26.0) == "moderate"
    assert utci_heat_category(32.0) == "strong"
    assert utci_heat_category(38.0) == "very_strong"
    assert utci_heat_category(46.0) == "extreme"


def test_threshold_recall_is_explicitly_unavailable_without_positive_case() -> None:
    result = recall_at_utci_threshold([37.0], [40.0], 38.0)
    assert result.value is None
    assert not result.available
    assert result.positive_support == 0
    assert result.reason is not None
