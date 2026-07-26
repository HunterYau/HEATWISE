"""Scalar and column-name checks for the independent training stages."""

from __future__ import annotations

import math

import pytest

from urban_heat_risk_ai.physics import (
    calculate_mean_radiant_temperature_from_globe,
    derive_sensor_utci_target,
)
from urban_heat_risk_ai.schema import (
    PUBLIC_REFERENCE_TARGET,
    STAGE1_PUBLIC_PROVENANCE_COLUMNS,
    STAGE1_SCHEMA_PROFILE,
    STAGE2_SCHEMA_PROFILE,
    STAGE2_SENSOR_RAW_COLUMNS,
    STAGE_PREDICTION_METADATA,
    STAGE_PREDICTION_SCHEMA_PROFILE,
    is_stage1_prohibited_column,
    is_stage_prediction_prohibited_column,
)


def test_stage_profiles_keep_public_and_sensor_targets_distinct() -> None:
    assert STAGE1_SCHEMA_PROFILE.required_target_inputs == (PUBLIC_REFERENCE_TARGET,)
    assert set(STAGE1_PUBLIC_PROVENANCE_COLUMNS).issubset(
        STAGE1_SCHEMA_PROFILE.required_metadata
    )
    assert STAGE2_SCHEMA_PROFILE.required_target_inputs == STAGE2_SENSOR_RAW_COLUMNS
    assert "calculated_utci_c" not in STAGE2_SCHEMA_PROFILE.required_target_inputs
    assert "calculated_mrt_c" not in STAGE2_SCHEMA_PROFILE.required_target_inputs


@pytest.mark.parametrize(
    "column",
    [
        "sensor_id",
        "calibration_version",
        "measured_air_temperature_c",
        "local_vapor_pressure_kpa",
        "calculated_utci_c",
    ],
)
def test_stage1_name_guard_rejects_nonpublic_training_fields(column: str) -> None:
    assert is_stage1_prohibited_column(column)


def test_stage1_name_guard_accepts_public_target_and_operational_predictor() -> None:
    assert not is_stage1_prohibited_column(PUBLIC_REFERENCE_TARGET)
    assert not is_stage1_prohibited_column("background_air_temperature_c")


def test_prediction_profile_needs_no_sensor_height_and_rejects_targets() -> None:
    assert STAGE_PREDICTION_SCHEMA_PROFILE.required_metadata == STAGE_PREDICTION_METADATA
    assert "measurement_height_m" not in STAGE_PREDICTION_METADATA
    assert is_stage_prediction_prohibited_column(PUBLIC_REFERENCE_TARGET)
    assert is_stage_prediction_prohibited_column("measured_globe_temperature_c")
    assert not is_stage_prediction_prohibited_column("background_shortwave_radiation_w_m2")


def test_iso_globe_mrt_matches_pythermalcomfort_documented_scalar_case() -> None:
    result = calculate_mean_radiant_temperature_from_globe(
        53.2,
        30.0,
        0.3,
        globe_diameter_m=0.1,
        globe_emissivity=0.95,
    )
    assert result == pytest.approx(74.8, abs=0.1)


def test_globe_mrt_wrapper_refuses_method_drift() -> None:
    with pytest.raises(ValueError, match="ISO"):
        calculate_mean_radiant_temperature_from_globe(
            40.0,
            30.0,
            1.0,
            globe_diameter_m=0.15,
            globe_emissivity=0.95,
            standard="Mixed Convection",
        )


def test_scalar_sensor_target_derivation_is_valid_without_clipping() -> None:
    target = derive_sensor_utci_target(
        30.0,
        50.0,
        1.0,
        40.0,
        2.0,
        globe_diameter_m=0.15,
        globe_emissivity=0.95,
        wind_profile={
            "method": "neutral_logarithmic_profile",
            "target_height_m": 10.0,
            "roughness_length_m": 0.1,
            "displacement_height_m": 0.0,
            "applicability": {
                "minimum_source_height_m": 0.1,
                "maximum_source_height_m": 10.0,
                "assume_neutral_stability": True,
            },
        },
    )
    assert target.valid
    assert target.flags == ()
    assert math.isfinite(target.calculated_mrt_c)
    assert math.isfinite(target.calculated_utci_c)
    assert target.wind_speed_10m_m_s > 1.0


def test_invalid_sensor_humidity_is_flagged_and_not_clipped() -> None:
    target = derive_sensor_utci_target(
        30.0,
        105.0,
        1.0,
        40.0,
        2.0,
        globe_diameter_m=0.15,
        globe_emissivity=0.95,
        roughness_length_m=0.1,
    )
    assert not target.valid
    assert math.isnan(target.calculated_utci_c)
    assert "sensor:relative_humidity_out_of_range" in target.flags
    assert "utci:relative_humidity_out_of_range" in target.flags


def test_nonfinite_sensor_input_returns_flags_without_calling_physics_library() -> None:
    target = derive_sensor_utci_target(
        30.0,
        float("nan"),
        1.0,
        40.0,
        2.0,
        globe_diameter_m=0.15,
        globe_emissivity=0.95,
        roughness_length_m=0.1,
    )
    assert not target.valid
    assert math.isnan(target.calculated_utci_c)
    assert "sensor:non_finite_relative_humidity" in target.flags
