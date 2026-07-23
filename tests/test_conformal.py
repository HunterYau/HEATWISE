"""Pure order-statistic tests; no data tables, models, or files are created."""

from __future__ import annotations

import math

import pytest

from urban_heat_risk_ai.conformal import conformal_order_statistic


def test_ninety_percent_conformal_uses_exact_kth_sorted_residual() -> None:
    result = conformal_order_statistic(
        [0.9, 0.1, 0.4, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5],
        alpha=0.10,
    )
    assert result.n_calibration == 9
    assert result.k == 9
    assert result.residual_quantile_c == pytest.approx(0.9)
    assert result.bounded


def test_small_calibration_set_returns_unbounded_interval_warning() -> None:
    with pytest.warns(RuntimeWarning, match="unbounded"):
        result = conformal_order_statistic([0.2], alpha=0.10)
    assert result.k == 2
    assert math.isinf(result.residual_quantile_c)
    assert not result.bounded
    assert result.warning is not None
