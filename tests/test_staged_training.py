"""Pure contract tests for the independent Stage 1 and Stage 2 workflows.

These tests parse configuration and scalar metadata only. They never construct
an observation table, fit a model, or write a staged-training artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from urban_heat_risk_ai.cli import build_parser
from urban_heat_risk_ai.config import load_project_config
from urban_heat_risk_ai.direct_xgb import validate_adaptation_overrides
from urban_heat_risk_ai.errors import ArtifactIntegrityError, LeakageError
from urban_heat_risk_ai.features import LeakageGuard, load_feature_policy
from urban_heat_risk_ai.staged_training import (
    STAGE_BUNDLE_CONTRACT_VERSION,
    StageInputFeatureSchema,
    ensure_separate_stage_output,
)


@pytest.mark.parametrize(
    ("argv", "command", "path_values"),
    [
        (
            ["stage1-validate", "--data", "public.parquet"],
            "stage1-validate",
            {"data": Path("public.parquet")},
        ),
        (
            [
                "stage1-train",
                "--data",
                "public.csv",
                "--output-dir",
                "base_bundle",
            ],
            "stage1-train",
            {
                "data": Path("public.csv"),
                "output_dir": Path("base_bundle"),
            },
        ),
        (
            [
                "stage2-validate",
                "--data",
                "local_sensor.csv",
                "--stage1-dir",
                "base_bundle",
            ],
            "stage2-validate",
            {
                "data": Path("local_sensor.csv"),
                "stage1_dir": Path("base_bundle"),
            },
        ),
        (
            [
                "stage2-adapt",
                "--data",
                "local_sensor.parquet",
                "--stage1-dir",
                "base_bundle",
                "--output-dir",
                "adapted_bundle",
            ],
            "stage2-adapt",
            {
                "data": Path("local_sensor.parquet"),
                "stage1_dir": Path("base_bundle"),
                "output_dir": Path("adapted_bundle"),
            },
        ),
        (
            [
                "stage-predict",
                "--data",
                "online_inputs.parquet",
                "--model-dir",
                "adapted_bundle",
                "--output",
                "predictions.parquet",
            ],
            "stage-predict",
            {
                "data": Path("online_inputs.parquet"),
                "model_dir": Path("adapted_bundle"),
                "output": Path("predictions.parquet"),
            },
        ),
    ],
)
def test_staged_cli_commands_parse_exact_explicit_paths(
    argv: list[str],
    command: str,
    path_values: dict[str, Path],
) -> None:
    args = build_parser().parse_args(argv)
    assert args.command == command
    for name, expected in path_values.items():
        assert getattr(args, name) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["stage1-validate"],
        ["stage1-train", "--data", "public.parquet"],
        ["stage2-validate", "--data", "local_sensor.parquet"],
        [
            "stage2-adapt",
            "--data",
            "local_sensor.parquet",
            "--stage1-dir",
            "base_bundle",
        ],
        [
            "stage-predict",
            "--data",
            "online_inputs.parquet",
            "--model-dir",
            "base_bundle",
        ],
    ],
)
def test_staged_cli_rejects_missing_required_paths(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)
    assert exc_info.value.code == 2


def test_checked_in_two_stage_configuration_is_lineage_safe() -> None:
    project = load_project_config()
    staged = project.model["training_stages"]
    stage1 = staged["stage1"]
    stage2 = staged["stage2"]

    assert stage1["target_column"] == "public_reference_utci_c"
    assert stage2["target_column"] == "calculated_utci_c"
    assert stage1["target_column"] != stage2["target_column"]
    assert stage2["target_provenance"] == (
        "derived_in_memory_from_raw_sensor_measurements"
    )
    assert stage2["reuse_stage1_preprocessor_without_refit"] is True
    assert stage2["require_exact_stage1_input_schema"] is True
    assert stage2["preserve_stage1_artifacts"] is True
    assert staged["common"]["require_exact_ordered_input_schema"] is True
    assert stage1["artifacts"]["model"] == "models/stage1_base.joblib"
    assert stage2["artifacts"]["model"] == "models/stage2_adapted.joblib"
    assert stage1["artifacts"]["model"] != stage2["artifacts"]["model"]
    assert project.features["training_stage_targets"] == {
        "stage1_public": "public_reference_utci_c",
        "stage2_sensor": "calculated_utci_c",
        "predictors_shared_exactly_between_stages": True,
    }


@pytest.mark.parametrize(
    "column",
    [
        "public_reference_utci_c",
        "log_public_reference_utci_c",
        "public_reference_utci_c_lag_1h",
        "public_reference_utci_c_squared",
    ],
)
def test_leakage_guard_rejects_public_target_and_transforms(column: str) -> None:
    guard = LeakageGuard(load_feature_policy(), "core")
    with pytest.raises(LeakageError):
        guard.validate([column])


def test_adaptation_overrides_accept_only_safe_new_tree_parameters() -> None:
    supplied = {
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.01,
        "reg_lambda": 2.0,
        "gamma": 0.0,
        "huber_slope": 1.0,
    }
    validated = validate_adaptation_overrides(supplied)
    assert validated == supplied
    assert validated is not supplied


@pytest.mark.parametrize("structural_key", ["objective", "n_estimators"])
def test_adaptation_overrides_reject_structural_changes(
    structural_key: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        validate_adaptation_overrides({structural_key: "not-allowed"})


def test_stage_input_feature_schema_mapping_and_digest_round_trip() -> None:
    schema = StageInputFeatureSchema(
        contract_version=STAGE_BUNDLE_CONTRACT_VERSION,
        predictor_set="core",
        predictor_policy_version="1.1.0",
        raw_predictors=(
            "background_air_temperature_c",
            "land_cover_class",
        ),
        numerical_predictors=("background_air_temperature_c",),
        categorical_predictors=("land_cover_class",),
        missing_indicator_predictors=("background_air_temperature_c",),
        transformed_features=(
            "background_air_temperature_c",
            "background_air_temperature_c__missing",
            "land_cover_class_urban",
        ),
        categorical_levels={"land_cover_class": ("urban", "water")},
        raw_feature_units={
            "background_air_temperature_c": "degree_Celsius",
            "land_cover_class": None,
        },
        feature_allowlist_sha256="a" * 64,
        stage1_model_config_sha256="b" * 64,
    )

    serialized = schema.to_dict()
    restored = StageInputFeatureSchema.from_mapping(serialized)

    assert restored == schema
    assert serialized["schema_sha256"] == schema.digest
    assert restored.digest == schema.digest


def test_stage_output_safety_rejects_nesting_without_writing() -> None:
    test_root = Path(__file__).resolve().parent
    stage1 = test_root / "__nonexistent_stage1_contract_bundle__"
    nested_stage2 = stage1 / "adapted"
    protected_output = test_root / "__nonexistent_protected_output__"

    assert not stage1.exists()
    assert not nested_stage2.exists()
    assert not protected_output.exists()
    with pytest.raises(ArtifactIntegrityError, match="wholly separate"):
        ensure_separate_stage_output(nested_stage2, stage1_dir=stage1)
    with pytest.raises(ArtifactIntegrityError, match="protected input"):
        ensure_separate_stage_output(
            protected_output,
            protected_files=[protected_output / "source.parquet"],
        )

