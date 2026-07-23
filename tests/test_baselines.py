"""Scalar-only checks for the analytical Heat Index baseline."""

from __future__ import annotations

import pytest

from urban_heat_risk_ai.baselines import heat_index_c


def test_heat_index_known_rothfusz_case() -> None:
    # 90 F and 70% RH gives approximately 105.9 F (41.1 C).
    assert heat_index_c(32.2222222222, 70.0) == pytest.approx(41.1, abs=0.15)


def test_heat_index_rejects_invalid_humidity() -> None:
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        heat_index_c(35.0, 101.0)
