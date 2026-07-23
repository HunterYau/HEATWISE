"""Pure feature-policy and feature-name leakage tests."""

from __future__ import annotations

import pytest

from urban_heat_risk_ai.errors import LeakageError
from urban_heat_risk_ai.features import LeakageGuard, find_leaky_feature_names, load_feature_policy


def test_core_and_enhanced_allow_lists_are_separate_and_ordered() -> None:
    policy = load_feature_policy()
    core = policy.allowed_predictors("core")
    enhanced = policy.allowed_predictors("satellite_enhanced")
    assert enhanced[: len(core)] == core
    assert "satellite_lst_c" not in core
    assert "satellite_lst_c" in enhanced


@pytest.mark.parametrize(
    "name",
    [
        "calculated_utci_c",
        "log_calculated_utci_c",
        "measured_air_temperature_c_squared",
        "site_id",
        "sensor_observed_wind",
        "timestamp_utc",
        "latitude",
        "outer_fold",
    ],
)
def test_leakage_guard_detects_target_sensor_id_and_transformed_names(name: str) -> None:
    policy = load_feature_policy()
    assert name in find_leaky_feature_names([name], policy)


def test_leakage_guard_accepts_only_declared_operational_name() -> None:
    policy = load_feature_policy()
    guard = LeakageGuard(policy, "core")
    assert guard.validate(["background_air_temperature_c"]) == (
        "background_air_temperature_c",
    )
    with pytest.raises(LeakageError, match="not declared"):
        guard.validate(["invented_weather_feature"])


def test_satellite_quality_indicator_is_explicitly_allowed_not_sensor_quality() -> None:
    policy = load_feature_policy()
    enhanced = LeakageGuard(policy, "satellite_enhanced")
    assert enhanced.validate(["satellite_lst_quality_flag"]) == (
        "satellite_lst_quality_flag",
    )
    with pytest.raises(LeakageError):
        enhanced.validate(["quality_flag"])

