"""Scalar, data-independent checks for physical conversion functions."""

from __future__ import annotations

import math

import pytest

from urban_heat_risk_ai.physics import (
    calculate_utci,
    convert_wind_to_10m,
    invert_wind_log_adjustment,
    relative_humidity_to_vapor_pressure,
    vapor_pressure_to_relative_humidity,
    wind_log_adjustment,
)


def test_humidity_vapor_pressure_round_trip_scalar() -> None:
    vapor_pressure = relative_humidity_to_vapor_pressure(30.0, 50.0)
    assert vapor_pressure == pytest.approx(2.118, abs=0.002)
    assert vapor_pressure_to_relative_humidity(30.0, vapor_pressure) == pytest.approx(50.0)


def test_log_wind_adjustment_round_trip_at_calm_background() -> None:
    adjustment = wind_log_adjustment(1.5, 0.0)
    assert invert_wind_log_adjustment(adjustment, 0.0) == pytest.approx(1.5)


def test_neutral_log_wind_profile_and_sensitivity() -> None:
    result = convert_wind_to_10m(
        1.0,
        2.0,
        0.1,
        sensitivity_roughness_lengths_m=(0.03,),
    )
    expected = math.log(10.0 / 0.1) / math.log(2.0 / 0.1)
    assert result.applicable
    assert result.wind_speed_10m_m_s == pytest.approx(expected)
    assert 0.03 in result.sensitivity_by_roughness_m


def test_wind_profile_reports_non_neutral_inapplicability() -> None:
    result = convert_wind_to_10m(1.0, 2.0, 0.1, neutral_stability=False)
    assert not result.applicable
    assert math.isnan(result.wind_speed_10m_m_s)
    assert "neutral_stability_not_applicable" in result.flags


def test_utci_matches_documented_scalar_case_without_rounding() -> None:
    # pythermalcomfort documents this case as 24.6 °C when rounded to one decimal.
    value = calculate_utci(25.0, 25.0, 1.0, 50.0)
    assert value == pytest.approx(24.6, abs=0.05)


def test_utci_out_of_range_policy_is_explicit() -> None:
    with pytest.raises(ValueError, match="wind_10m_out_of_range"):
        calculate_utci(25.0, 25.0, 0.1, 50.0)
    assert math.isnan(calculate_utci(25.0, 25.0, 0.1, 50.0, out_of_range="nan"))
