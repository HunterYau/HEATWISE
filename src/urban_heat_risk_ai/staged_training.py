"""Contracts and artifacts for public-base and local-sensor training stages.

Stage 1 and Stage 2 always write separate completed bundles. Stage 2 loads and
verifies the Stage 1 bundle, reuses its fitted preprocessor unchanged, and
records parent artifact hashes as immutable lineage.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from urban_heat_risk_ai.artifacts import (
    ArtifactStore,
    sha256_canonical,
    sha256_file,
    software_versions,
)
from urban_heat_risk_ai.config import ProjectConfig
from urban_heat_risk_ai.direct_xgb import FittedDirectModel
from urban_heat_risk_ai.errors import ArtifactIntegrityError
from urban_heat_risk_ai.features import FeaturePolicy
from urban_heat_risk_ai.physics import SensorUTCITarget, derive_sensor_utci_target
from urban_heat_risk_ai.schema import STAGE2_SENSOR_RAW_COLUMNS, UNIT_BY_COLUMN

StageBundleKind = Literal["stage1_public_base", "stage2_local_adapted"]
STAGE_BUNDLE_CONTRACT_VERSION = "1.0"
DEFAULT_COMPLETION_MANIFEST = "artifact_manifest.json"


@dataclass(frozen=True)
class StageInputFeatureSchema:
    """Frozen raw and transformed input contract shared by both stages."""

    contract_version: str
    predictor_set: str
    predictor_policy_version: str
    raw_predictors: tuple[str, ...]
    numerical_predictors: tuple[str, ...]
    categorical_predictors: tuple[str, ...]
    missing_indicator_predictors: tuple[str, ...]
    transformed_features: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    raw_feature_units: Mapping[str, str | None]
    feature_allowlist_sha256: str
    stage1_model_config_sha256: str

    @classmethod
    def from_fitted_model(
        cls,
        fitted: FittedDirectModel,
        *,
        predictor_set: str,
        policy: FeaturePolicy,
        feature_allowlist_sha256: str,
        model_config_sha256: str,
    ) -> StageInputFeatureSchema:
        """Capture the exact fitted Stage 1 preprocessing contract."""

        preprocessor = fitted.preprocessor
        numerical = tuple(str(value) for value in preprocessor.numerical_columns_)
        categorical = tuple(str(value) for value in preprocessor.categorical_columns_)
        indicators = tuple(
            str(value) for value in preprocessor.missing_indicator_columns_
        )
        transformed = tuple(
            str(value) for value in preprocessor.get_feature_names_out().tolist()
        )
        levels: dict[str, tuple[str, ...]] = {}
        if categorical:
            encoder = preprocessor.categorical_encoder_
            levels = {
                column: tuple(str(value) for value in values.tolist())
                for column, values in zip(
                    categorical, encoder.categories_, strict=True
                )
            }
        raw = tuple(fitted.predictors)
        return cls(
            contract_version=STAGE_BUNDLE_CONTRACT_VERSION,
            predictor_set=predictor_set,
            predictor_policy_version=policy.version,
            raw_predictors=raw,
            numerical_predictors=numerical,
            categorical_predictors=categorical,
            missing_indicator_predictors=indicators,
            transformed_features=transformed,
            categorical_levels=levels,
            raw_feature_units={
                name: UNIT_BY_COLUMN.get(name)
                for name in raw
            },
            feature_allowlist_sha256=feature_allowlist_sha256,
            stage1_model_config_sha256=model_config_sha256,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StageInputFeatureSchema:
        """Parse a JSON-compatible saved schema without accepting missing fields."""

        levels = value.get("categorical_levels")
        units = value.get("raw_feature_units")
        if not isinstance(levels, Mapping) or not isinstance(units, Mapping):
            raise ArtifactIntegrityError(
                "Saved input schema lacks categorical-level or unit mappings."
            )
        try:
            parsed = cls(
                contract_version=str(value["contract_version"]),
                predictor_set=str(value["predictor_set"]),
                predictor_policy_version=str(value["predictor_policy_version"]),
                raw_predictors=tuple(str(item) for item in value["raw_predictors"]),
                numerical_predictors=tuple(
                    str(item) for item in value["numerical_predictors"]
                ),
                categorical_predictors=tuple(
                    str(item) for item in value["categorical_predictors"]
                ),
                missing_indicator_predictors=tuple(
                    str(item) for item in value["missing_indicator_predictors"]
                ),
                transformed_features=tuple(
                    str(item) for item in value["transformed_features"]
                ),
                categorical_levels={
                    str(name): tuple(str(item) for item in items)
                    for name, items in levels.items()
                },
                raw_feature_units={
                    str(name): None if unit is None else str(unit)
                    for name, unit in units.items()
                },
                feature_allowlist_sha256=str(value["feature_allowlist_sha256"]),
                stage1_model_config_sha256=str(
                    value["stage1_model_config_sha256"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactIntegrityError(
                f"Saved Stage 1 input schema is incomplete: {exc}"
            ) from exc
        declared_digest = value.get("schema_sha256")
        if not isinstance(declared_digest, str) or declared_digest != parsed.digest:
            raise ArtifactIntegrityError(
                "Saved Stage 1 input schema content does not match schema_sha256."
            )
        return parsed

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible schema content."""

        return {
            "contract_version": self.contract_version,
            "predictor_set": self.predictor_set,
            "predictor_policy_version": self.predictor_policy_version,
            "raw_predictors": list(self.raw_predictors),
            "numerical_predictors": list(self.numerical_predictors),
            "categorical_predictors": list(self.categorical_predictors),
            "missing_indicator_predictors": list(
                self.missing_indicator_predictors
            ),
            "transformed_features": list(self.transformed_features),
            "categorical_levels": {
                name: list(values)
                for name, values in sorted(self.categorical_levels.items())
            },
            "raw_feature_units": dict(sorted(self.raw_feature_units.items())),
            "feature_allowlist_sha256": self.feature_allowlist_sha256,
            "stage1_model_config_sha256": self.stage1_model_config_sha256,
            "schema_sha256": self.digest,
        }

    @property
    def digest(self) -> str:
        """Hash the schema content without its self-referential digest field."""

        value = {
            "contract_version": self.contract_version,
            "predictor_set": self.predictor_set,
            "predictor_policy_version": self.predictor_policy_version,
            "raw_predictors": self.raw_predictors,
            "numerical_predictors": self.numerical_predictors,
            "categorical_predictors": self.categorical_predictors,
            "missing_indicator_predictors": self.missing_indicator_predictors,
            "transformed_features": self.transformed_features,
            "categorical_levels": self.categorical_levels,
            "raw_feature_units": self.raw_feature_units,
            "feature_allowlist_sha256": self.feature_allowlist_sha256,
            "stage1_model_config_sha256": self.stage1_model_config_sha256,
        }
        return sha256_canonical(value)

    def assert_compatible(
        self,
        fitted: FittedDirectModel,
        *,
        policy: FeaturePolicy,
        feature_allowlist_sha256: str,
    ) -> None:
        """Fail closed on any Stage 1/Stage 2 raw or transformed schema drift."""

        expected_raw = policy.allowed_predictors("core")
        preprocessor = fitted.preprocessor
        categorical = tuple(
            str(value) for value in preprocessor.categorical_columns_
        )
        observed_levels: dict[str, tuple[str, ...]] = {}
        if categorical:
            observed_levels = {
                column: tuple(str(value) for value in values.tolist())
                for column, values in zip(
                    categorical,
                    preprocessor.categorical_encoder_.categories_,
                    strict=True,
                )
            }
        checks = {
            "bundle contract version": (
                self.contract_version,
                STAGE_BUNDLE_CONTRACT_VERSION,
            ),
            "predictor set": (self.predictor_set, "core"),
            "predictor policy version": (
                self.predictor_policy_version,
                policy.version,
            ),
            "feature allow-list hash": (
                self.feature_allowlist_sha256,
                feature_allowlist_sha256,
            ),
            "ordered raw predictors": (
                self.raw_predictors,
                expected_raw,
            ),
            "model predictor order": (
                tuple(fitted.predictors),
                self.raw_predictors,
            ),
            "preprocessor predictor order": (
                tuple(preprocessor.predictor_names_in_),
                self.raw_predictors,
            ),
            "numerical predictor order": (
                tuple(preprocessor.numerical_columns_),
                self.numerical_predictors,
            ),
            "categorical predictor order": (
                categorical,
                self.categorical_predictors,
            ),
            "missing-indicator predictor order": (
                tuple(preprocessor.missing_indicator_columns_),
                self.missing_indicator_predictors,
            ),
            "categorical levels": (
                observed_levels,
                dict(self.categorical_levels),
            ),
            "raw feature units": (
                dict(self.raw_feature_units),
                {name: UNIT_BY_COLUMN.get(name) for name in expected_raw},
            ),
            "numeric NaN retention": (
                bool(preprocessor.retain_numeric_nan_),
                True,
            ),
            "strict input columns": (
                bool(preprocessor.strict_columns),
                True,
            ),
            "embedded predictor set": (
                preprocessor.predictor_set,
                "core",
            ),
            "embedded model kind": (
                preprocessor.model_kind,
                "xgboost",
            ),
            "embedded policy version": (
                preprocessor.policy.version,
                self.predictor_policy_version,
            ),
            "transformed feature order": (
                tuple(
                    str(value)
                    for value in preprocessor.get_feature_names_out().tolist()
                ),
                self.transformed_features,
            ),
        }
        if categorical:
            checks["unknown categorical handling"] = (
                preprocessor.categorical_encoder_.handle_unknown,
                "ignore",
            )
        mismatches = [
            name
            for name, (observed, expected) in checks.items()
            if observed != expected
        ]
        if mismatches:
            raise ArtifactIntegrityError(
                "Stage 1 input-feature contract is incompatible: "
                + ", ".join(mismatches)
            )

    def assert_separate_preprocessor_compatible(self, preprocessor: Any) -> None:
        """Verify that the separately saved preprocessor matches the schema."""

        categorical = tuple(
            str(value) for value in preprocessor.categorical_columns_
        )
        levels: dict[str, tuple[str, ...]] = {}
        if categorical:
            levels = {
                column: tuple(str(value) for value in values.tolist())
                for column, values in zip(
                    categorical,
                    preprocessor.categorical_encoder_.categories_,
                    strict=True,
                )
            }
        checks = {
            "raw predictor order": (
                tuple(preprocessor.predictor_names_in_),
                self.raw_predictors,
            ),
            "numerical predictor order": (
                tuple(preprocessor.numerical_columns_),
                self.numerical_predictors,
            ),
            "categorical predictor order": (
                categorical,
                self.categorical_predictors,
            ),
            "missing-indicator predictor order": (
                tuple(preprocessor.missing_indicator_columns_),
                self.missing_indicator_predictors,
            ),
            "categorical levels": (levels, dict(self.categorical_levels)),
            "numeric NaN retention": (
                bool(preprocessor.retain_numeric_nan_),
                True,
            ),
            "strict input columns": (
                bool(preprocessor.strict_columns),
                True,
            ),
            "embedded predictor set": (
                preprocessor.predictor_set,
                "core",
            ),
            "embedded model kind": (
                preprocessor.model_kind,
                "xgboost",
            ),
            "embedded policy version": (
                preprocessor.policy.version,
                self.predictor_policy_version,
            ),
            "transformed feature order": (
                tuple(
                    str(value)
                    for value in preprocessor.get_feature_names_out().tolist()
                ),
                self.transformed_features,
            ),
        }
        if categorical:
            checks["unknown categorical handling"] = (
                preprocessor.categorical_encoder_.handle_unknown,
                "ignore",
            )
        mismatches = [
            name
            for name, (observed, expected) in checks.items()
            if observed != expected
        ]
        if mismatches:
            raise ArtifactIntegrityError(
                "Separate saved preprocessor is incompatible with the frozen "
                "input schema: "
                + ", ".join(mismatches)
            )


@dataclass(frozen=True)
class LoadedStageBundle:
    """Verified completed model bundle ready for adaptation or prediction."""

    root: Path
    stage: StageBundleKind
    model: FittedDirectModel
    input_schema: StageInputFeatureSchema
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class Stage2TargetBatch:
    """In-memory sensor-derived target values and physical intermediates."""

    calculated_utci_c: tuple[float, ...]
    calculated_mrt_c: tuple[float, ...]
    wind_speed_10m_m_s: tuple[float, ...]
    flags: tuple[tuple[str, ...], ...]

    @property
    def valid(self) -> bool:
        return all(not row_flags for row_flags in self.flags) and bool(
            self.calculated_utci_c
        )

    def target_array(self) -> np.ndarray:
        values = np.asarray(self.calculated_utci_c, dtype=float)
        if not self.valid or not np.isfinite(values).all():
            raise ValueError("Stage 2 target batch contains invalid UTCI values.")
        return values


def staged_config(project: ProjectConfig) -> Mapping[str, Any]:
    """Return the checked and required two-stage configuration."""

    configured = project.model.get("training_stages")
    if not isinstance(configured, Mapping):
        raise ValueError("model.yaml is missing training_stages configuration.")
    return configured


def stage_config(project: ProjectConfig, stage: Literal["stage1", "stage2"]) -> Mapping[str, Any]:
    configured = staged_config(project).get(stage)
    if not isinstance(configured, Mapping):
        raise ValueError(f"model.yaml is missing training_stages.{stage}.")
    return configured


def artifact_paths(
    project: ProjectConfig, stage: Literal["stage1", "stage2"]
) -> dict[str, str]:
    """Normalize configured relative artifact names to common logical keys."""

    configured = stage_config(project, stage).get("artifacts")
    if not isinstance(configured, Mapping):
        raise ValueError(f"training_stages.{stage}.artifacts must be a mapping.")
    if stage == "stage1":
        aliases = {
            "model": "model",
            "preprocessor": "preprocessor",
            "input_schema": "input_schema",
            "metadata": "metadata",
            "hashes": "hashes",
            "completion_manifest": "completion_manifest",
        }
    else:
        aliases = {
            "model": "model",
            "frozen_preprocessor": "preprocessor",
            "input_schema": "input_schema",
            "metadata": "metadata",
            "parent_lineage": "parent_lineage",
            "hashes": "hashes",
            "completion_manifest": "completion_manifest",
        }
    result: dict[str, str] = {}
    for configured_name, logical_name in aliases.items():
        value = configured.get(configured_name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"training_stages.{stage}.artifacts.{configured_name} is required."
            )
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Stage artifact path must remain relative: {value}")
        result[logical_name] = value
    normalized_paths: dict[str, list[str]] = {}
    for value in result.values():
        normalized = str(Path(value)).replace("\\", "/").casefold()
        normalized_paths.setdefault(normalized, []).append(value)
    duplicates = sorted(
        value
        for values in normalized_paths.values()
        if len(values) > 1
        for value in values
    )
    if duplicates:
        raise ValueError(f"Stage artifact paths must be unique: {duplicates}")
    reserved = {"configs/model.yaml", "configs/features.yaml"}
    collisions = sorted(reserved.intersection(normalized_paths))
    if collisions:
        raise ValueError(
            f"Stage artifact paths collide with configuration snapshots: {collisions}"
        )
    return result


def ensure_separate_stage_output(
    output_dir: str | Path,
    *,
    stage1_dir: str | Path | None = None,
    protected_files: Sequence[str | Path] = (),
) -> Path:
    """Require a new output tree that cannot contain or replace Stage 1/input files."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ArtifactIntegrityError(
            f"Stage output directory already exists and will not be overwritten: {output}"
        )
    if stage1_dir is not None:
        parent = Path(stage1_dir).expanduser().resolve()
        if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ArtifactIntegrityError(
                "Stage 2 output must be wholly separate from the immutable Stage 1 "
                f"directory: {parent}"
            )
    for protected in protected_files:
        source = Path(protected).expanduser().resolve()
        if output == source or source.is_relative_to(output):
            raise ArtifactIntegrityError(
                f"Stage output would replace or contain a protected input: {source}"
            )
    return output


def select_stage_training_rows(
    frame: pd.DataFrame,
    *,
    stage: Literal["stage1", "stage2"],
    project: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select configured development/QC rows without modifying the source frame."""

    configured = stage_config(project, stage)
    shared = staged_config(project).get("common", {})
    if not isinstance(shared, Mapping):
        shared = {}
    role = str(shared.get("training_split_role", "development"))
    quality_column = "public_quality_flag" if stage == "stage1" else "quality_flag"
    allowed_quality = {
        str(value).strip().lower()
        for value in configured.get("allowed_quality_flags", ("pass",))
    }
    roles = frame["split_role"].astype("string").str.strip()
    qualities = frame[quality_column].astype("string").str.strip().str.lower()
    mask = roles.eq(role) & qualities.isin(allowed_quality)
    selected = frame.loc[mask].copy()
    minimum = int(configured.get("minimum_training_rows", 1))
    if len(selected) < minimum:
        raise ValueError(
            f"{stage} has {len(selected)} eligible rows after split/QC filtering; "
            f"at least {minimum} are configured."
        )
    return selected, {
        "input_rows": int(len(frame)),
        "eligible_training_rows": int(len(selected)),
        "excluded_rows": int((~mask).sum()),
    }


def derive_stage2_target_batch(
    frame: pd.DataFrame,
    *,
    project: ProjectConfig,
) -> Stage2TargetBatch:
    """Freshly derive every Stage 2 target; supplied labels are audit-only."""

    missing = [
        column
        for column in (*STAGE2_SENSOR_RAW_COLUMNS, "measurement_height_m")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Stage 2 target derivation is missing columns: {missing}")
    configured = stage_config(project, "stage2")
    globe = configured.get("globe_mrt", {})
    if not isinstance(globe, Mapping):
        raise ValueError("training_stages.stage2.globe_mrt must be a mapping.")
    if (
        globe.get("implementation")
        != "pythermalcomfort.utilities.mean_radiant_tmp"
    ):
        raise ValueError(
            "Stage 2 globe MRT must use pythermalcomfort.utilities.mean_radiant_tmp."
        )
    if bool(globe.get("clip_inputs", False)):
        raise ValueError("Stage 2 target derivation cannot clip globe MRT inputs.")
    utci_config = project.model.get("utci", {})
    if not isinstance(utci_config, Mapping):
        raise ValueError("utci configuration must be a mapping.")
    if utci_config.get("implementation") != "pythermalcomfort.models.utci":
        raise ValueError("Stage 2 UTCI must use pythermalcomfort.models.utci.")
    if str(utci_config.get("units", "")).upper() != "SI":
        raise ValueError("Stage 2 UTCI target derivation requires SI units.")
    if utci_config.get("round_output") is not False:
        raise ValueError("Stage 2 UTCI target derivation requires round_output: false.")
    if utci_config.get("pythermalcomfort_limit_inputs") is not False:
        raise ValueError(
            "Stage 2 requires explicit wrapper limits, not hidden pythermalcomfort limits."
        )
    if utci_config.get("out_of_range_policy") != "flag_and_return_nan":
        raise ValueError(
            "Stage 2 UTCI out-of-range policy must be flag_and_return_nan."
        )
    limits = utci_config.get("explicit_applicability_limits")
    if limits is not None and not isinstance(limits, Mapping):
        raise ValueError("utci.explicit_applicability_limits must be a mapping.")
    wind_profile = project.model.get("wind_profile", {})
    if not isinstance(wind_profile, Mapping):
        raise ValueError("wind_profile configuration must be a mapping.")

    results: list[SensorUTCITarget] = []
    for values in zip(
        frame["measured_air_temperature_c"],
        frame["measured_relative_humidity_pct"],
        frame["measured_pedestrian_wind_speed_m_s"],
        frame["measured_globe_temperature_c"],
        frame["measurement_height_m"],
        strict=True,
    ):
        results.append(
            derive_sensor_utci_target(
                *(float(value) for value in values),
                globe_diameter_m=float(globe.get("globe_diameter_m", 0.15)),
                globe_emissivity=float(globe.get("globe_emissivity", 0.95)),
                globe_standard=str(globe.get("standard", "ISO")),
                wind_profile=wind_profile,
                utci_limits=limits,
            )
        )
    batch = Stage2TargetBatch(
        calculated_utci_c=tuple(result.calculated_utci_c for result in results),
        calculated_mrt_c=tuple(result.calculated_mrt_c for result in results),
        wind_speed_10m_m_s=tuple(
            result.wind_speed_10m_m_s for result in results
        ),
        flags=tuple(result.flags for result in results),
    )
    invalid_positions = [
        position for position, result in enumerate(results) if not result.valid
    ]
    if invalid_positions:
        examples = []
        for position in invalid_positions[:5]:
            sample = (
                str(frame.iloc[position]["sample_id"])
                if "sample_id" in frame.columns
                else str(position)
            )
            reasons = batch.flags[position] or ("non_finite_derived_target",)
            examples.append(f"{sample}: {', '.join(reasons)}")
        raise ValueError(
            f"{len(invalid_positions)} eligible Stage 2 row(s) cannot produce a "
            "physical UTCI target. No row was dropped or clipped. Examples: "
            + "; ".join(examples)
        )

    _verify_optional_derived_column(
        frame,
        supplied_column="calculated_mrt_c",
        derived=batch.calculated_mrt_c,
        tolerance_c=float(
            configured.get("supplied_mrt_verification_tolerance_c", 0.25)
        ),
    )
    _verify_optional_derived_column(
        frame,
        supplied_column="calculated_utci_c",
        derived=batch.calculated_utci_c,
        tolerance_c=float(
            configured.get("supplied_target_verification_tolerance_c", 0.10)
        ),
    )
    _verify_optional_derived_category(frame, batch.calculated_utci_c)
    return batch


def _verify_optional_derived_column(
    frame: pd.DataFrame,
    *,
    supplied_column: str,
    derived: Sequence[float],
    tolerance_c: float,
) -> None:
    if supplied_column not in frame.columns:
        return
    if tolerance_c < 0.0:
        raise ValueError(f"{supplied_column} verification tolerance cannot be negative.")
    supplied = pd.to_numeric(frame[supplied_column], errors="coerce").to_numpy(
        dtype=float
    )
    expected = np.asarray(derived, dtype=float)
    comparable = np.isfinite(supplied) & np.isfinite(expected)
    mismatched = comparable & (np.abs(supplied - expected) > tolerance_c)
    if mismatched.any():
        positions = np.flatnonzero(mismatched)
        examples = []
        for position in positions[:5]:
            sample = (
                str(frame.iloc[int(position)]["sample_id"])
                if "sample_id" in frame.columns
                else str(position)
            )
            examples.append(
                f"{sample}: supplied={supplied[position]:.3f}, "
                f"derived={expected[position]:.3f}"
            )
        raise ValueError(
            f"{int(mismatched.sum())} supplied {supplied_column} value(s) differ "
            f"from fresh Stage 2 derivation by more than {tolerance_c:g} C. "
            "The supplied column is never used for training. Examples: "
            + "; ".join(examples)
        )


def _verify_optional_derived_category(
    frame: pd.DataFrame,
    derived_utci_c: Sequence[float],
) -> None:
    """Audit a supplied category against fixed thresholds without using it."""

    if "utci_category" not in frame.columns:
        return
    values = np.asarray(derived_utci_c, dtype=float)
    expected = np.select(
        [
            values < 26.0,
            values < 32.0,
            values < 38.0,
            values < 46.0,
        ],
        ["no_heat_stress", "moderate", "strong", "very_strong"],
        default="extreme",
    )
    supplied = frame["utci_category"].astype("string").str.strip()
    comparable = supplied.notna().to_numpy(dtype=bool)
    supplied_values = supplied.fillna("").to_numpy(dtype=str)
    mismatched = comparable & (supplied_values != expected)
    if mismatched.any():
        positions = np.flatnonzero(mismatched)
        examples = [
            (
                f"{frame.iloc[int(position)]['sample_id']}: "
                f"supplied={supplied.iloc[int(position)]!s}, "
                f"derived={expected[int(position)]}"
            )
            for position in positions[:5]
        ]
        raise ValueError(
            f"{int(mismatched.sum())} supplied utci_category value(s) disagree "
            "with the freshly derived continuous UTCI thresholds. The supplied "
            "category is never used for training. Examples: "
            + "; ".join(examples)
        )


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ArtifactIntegrityError(f"Unsafe artifact path in manifest: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ArtifactIntegrityError(f"Artifact path escapes bundle: {relative}")
    return resolved


def _verified_reference(root: Path, reference: Any, logical_name: str) -> Path:
    if not isinstance(reference, Mapping):
        raise ArtifactIntegrityError(
            f"Completed stage manifest lacks {logical_name!r} artifact metadata."
        )
    relative = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ArtifactIntegrityError(
            f"Stage artifact reference {logical_name!r} is incomplete."
        )
    path = _safe_artifact_path(root, relative)
    if not path.is_file():
        raise ArtifactIntegrityError(f"Stage artifact is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ArtifactIntegrityError(
            f"Stage artifact hash mismatch for {logical_name}: "
            f"expected {expected_hash}, observed {observed}."
        )
    return path


def load_completed_stage_bundle(
    root: str | Path,
    *,
    policy: FeaturePolicy,
    feature_allowlist_path: str | Path,
    allowed_stages: Sequence[StageBundleKind] = (
        "stage1_public_base",
        "stage2_local_adapted",
    ),
) -> LoadedStageBundle:
    """Verify all completion-manifest hashes before loading local joblib files."""

    bundle_root = Path(root).expanduser().resolve()
    manifest_path = bundle_root / DEFAULT_COMPLETION_MANIFEST
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(
            f"Completed staged-training manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ArtifactIntegrityError(
            "Staged-training completion manifest must be a JSON object."
        )
    if manifest.get("contract_version") != STAGE_BUNDLE_CONTRACT_VERSION:
        raise ArtifactIntegrityError(
            "Staged-training bundle contract version is unsupported."
        )
    if manifest.get("completed") is not True:
        raise ArtifactIntegrityError("Staged-training bundle is not marked complete.")
    stage = manifest.get("stage")
    if stage not in allowed_stages:
        raise ArtifactIntegrityError(
            f"Expected stage in {list(allowed_stages)}, found {stage!r}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArtifactIntegrityError("Stage completion manifest lacks artifact references.")
    required_artifacts = {
        "model",
        "preprocessor",
        "input_schema",
        "metadata",
        "hashes",
        "model_config",
        "feature_allowlist",
    }
    if stage == "stage2_local_adapted":
        required_artifacts.add("parent_lineage")
    missing_artifacts = sorted(required_artifacts.difference(artifacts))
    if missing_artifacts:
        raise ArtifactIntegrityError(
            f"Stage completion manifest is missing artifacts: {missing_artifacts}"
        )
    verified_paths = {
        logical_name: _verified_reference(
            bundle_root,
            artifacts.get(logical_name),
            logical_name,
        )
        for logical_name in sorted(artifacts)
    }
    model_path = verified_paths["model"]
    preprocessor_path = verified_paths["preprocessor"]
    schema_path = verified_paths["input_schema"]

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    if not isinstance(model, FittedDirectModel):
        raise ArtifactIntegrityError("Saved stage model has an unexpected object type.")
    schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_value, Mapping):
        raise ArtifactIntegrityError("Stage input schema must be a JSON object.")
    schema = StageInputFeatureSchema.from_mapping(schema_value)
    schema.assert_separate_preprocessor_compatible(preprocessor)
    preprocessing_attributes = (
        "add_missing_indicators",
        "missing_token",
        "strict_columns",
        "model_kind",
        "predictor_set",
    )
    preprocessing_mismatches = [
        name
        for name in preprocessing_attributes
        if getattr(preprocessor, name, None)
        != getattr(model.preprocessor, name, None)
    ]
    if preprocessing_mismatches:
        raise ArtifactIntegrityError(
            "Separate preprocessing artifact differs from the model-bundled "
            "preprocessor: "
            + ", ".join(preprocessing_mismatches)
        )
    expected_variant = {
        "stage1_public_base": "stage1_public_base",
        "stage2_local_adapted": "stage2_local_adapted",
    }[stage]
    if model.model_variant != expected_variant:
        raise ArtifactIntegrityError(
            f"Stage model variant {model.model_variant!r} does not match bundle "
            f"stage {stage!r}."
        )
    schema.assert_compatible(
        model,
        policy=policy,
        feature_allowlist_sha256=sha256_file(feature_allowlist_path),
    )
    if sha256_file(verified_paths["model_config"]) != schema.stage1_model_config_sha256:
        raise ArtifactIntegrityError(
            "Saved model configuration does not match the Stage 1 input schema."
        )
    if sha256_file(verified_paths["feature_allowlist"]) != schema.feature_allowlist_sha256:
        raise ArtifactIntegrityError(
            "Saved feature allow-list does not match the Stage 1 input schema."
        )
    saved_hashes = json.loads(
        verified_paths["hashes"].read_text(encoding="utf-8")
    )
    if not isinstance(saved_hashes, Mapping):
        raise ArtifactIntegrityError("hashes.json must be a JSON object.")
    if saved_hashes != manifest.get("input_hashes"):
        raise ArtifactIntegrityError(
            "hashes.json differs from the completion manifest input hashes."
        )
    metadata = json.loads(
        verified_paths["metadata"].read_text(encoding="utf-8")
    )
    if not isinstance(metadata, Mapping):
        raise ArtifactIntegrityError("Stage metadata must be a JSON object.")
    metadata_checks = {
        "stage": (metadata.get("stage"), stage),
        "model variant": (metadata.get("model_variant"), model.model_variant),
        "tree count": (metadata.get("tree_count"), model.tree_count),
        "input schema digest": (
            metadata.get("input_schema_sha256"),
            schema.digest,
        ),
    }
    metadata_mismatches = [
        name
        for name, (observed, expected) in metadata_checks.items()
        if observed != expected
    ]
    if metadata_mismatches:
        raise ArtifactIntegrityError(
            "Stage metadata is inconsistent with verified artifacts: "
            + ", ".join(metadata_mismatches)
        )
    if stage == "stage2_local_adapted":
        lineage = json.loads(
            verified_paths["parent_lineage"].read_text(encoding="utf-8")
        )
        if not isinstance(lineage, Mapping) or lineage.get("immutable_parent") is not True:
            raise ArtifactIntegrityError(
                "Stage 2 parent lineage is missing its immutable-parent assertion."
            )
        lineage_checks = {
            "completion manifest": (
                saved_hashes.get("stage1_completion_manifest_sha256"),
                lineage.get("stage1_completion_manifest_sha256"),
            ),
            "model": (
                saved_hashes.get("stage1_model_sha256"),
                lineage.get("stage1_model", {}).get("sha256")
                if isinstance(lineage.get("stage1_model"), Mapping)
                else None,
            ),
            "preprocessor": (
                saved_hashes.get("stage1_preprocessor_sha256"),
                lineage.get("stage1_preprocessor", {}).get("sha256")
                if isinstance(lineage.get("stage1_preprocessor"), Mapping)
                else None,
            ),
            "input schema": (
                saved_hashes.get("stage1_input_schema_sha256"),
                lineage.get("stage1_input_schema", {}).get("sha256")
                if isinstance(lineage.get("stage1_input_schema"), Mapping)
                else None,
            ),
            "model before adaptation": (
                lineage.get("stage1_model_sha256_before_adaptation"),
                lineage.get("stage1_model", {}).get("sha256")
                if isinstance(lineage.get("stage1_model"), Mapping)
                else None,
            ),
            "model after adaptation": (
                lineage.get("stage1_model_sha256_after_adaptation"),
                lineage.get("stage1_model", {}).get("sha256")
                if isinstance(lineage.get("stage1_model"), Mapping)
                else None,
            ),
        }
        lineage_mismatches = [
            name
            for name, (observed, expected) in lineage_checks.items()
            if not isinstance(observed, str) or observed != expected
        ]
        if lineage_mismatches:
            raise ArtifactIntegrityError(
                "Stage 2 parent lineage differs from its frozen input hashes: "
                + ", ".join(lineage_mismatches)
            )
        before_hashes = lineage.get(
            "stage1_artifact_hashes_before_adaptation"
        )
        after_hashes = lineage.get(
            "stage1_artifact_hashes_after_adaptation"
        )
        parent_artifacts = lineage.get("stage1_artifacts")
        if (
            not isinstance(before_hashes, Mapping)
            or not isinstance(after_hashes, Mapping)
            or before_hashes != after_hashes
            or not isinstance(parent_artifacts, Mapping)
        ):
            raise ArtifactIntegrityError(
                "Stage 2 lineage lacks matching before/after hashes for the "
                "immutable Stage 1 bundle."
            )
        referenced_hashes = {
            str(name): reference.get("sha256")
            for name, reference in parent_artifacts.items()
            if isinstance(reference, Mapping)
        }
        referenced_hashes["completion_manifest"] = lineage.get(
            "stage1_completion_manifest_sha256"
        )
        if dict(before_hashes) != referenced_hashes:
            raise ArtifactIntegrityError(
                "Stage 2 lineage artifact hashes do not match its Stage 1 references."
            )
    return LoadedStageBundle(
        root=bundle_root,
        stage=stage,
        model=model,
        input_schema=schema,
        manifest=manifest,
    )


def _artifact_reference(root: Path, relative: str) -> dict[str, str]:
    path = _safe_artifact_path(root, relative)
    return {"path": relative, "sha256": sha256_file(path)}


def write_completed_stage_bundle(
    *,
    output_dir: str | Path,
    stage: StageBundleKind,
    project: ProjectConfig,
    model: FittedDirectModel,
    input_schema: StageInputFeatureSchema,
    input_hashes: Mapping[str, str],
    run_metadata: Mapping[str, Any],
    model_config_snapshot: bytes,
    feature_config_snapshot: bytes,
    parent_lineage: Mapping[str, Any] | None = None,
    before_completion_commit: Callable[[], None] | None = None,
) -> Path:
    """Write a new stage bundle and commit its completion manifest last."""

    stage_key: Literal["stage1", "stage2"] = (
        "stage1" if stage == "stage1_public_base" else "stage2"
    )
    paths = artifact_paths(project, stage_key)
    completion_path = paths["completion_manifest"]
    if completion_path != DEFAULT_COMPLETION_MANIFEST:
        raise ValueError(
            "The current loader requires artifact_manifest.json as the completion name."
        )
    root = Path(output_dir).expanduser().resolve()
    store = ArtifactStore(root).initialize()
    store.write_joblib(paths["model"], model)
    store.write_joblib(paths["preprocessor"], model.preprocessor)
    store.write_json(paths["input_schema"], input_schema.to_dict())
    store.write_json(paths["hashes"], dict(input_hashes))
    metadata = {
        "stage": stage,
        "contract_version": STAGE_BUNDLE_CONTRACT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_variant": model.model_variant,
        "tree_count": model.tree_count,
        "objective": model.objective,
        "input_schema_sha256": input_schema.digest,
        "software_versions": software_versions(),
        **dict(run_metadata),
    }
    store.write_json(paths["metadata"], metadata)
    store.write_bytes("configs/model.yaml", model_config_snapshot)
    store.write_bytes("configs/features.yaml", feature_config_snapshot)
    if parent_lineage is not None:
        parent_path = paths.get("parent_lineage")
        if parent_path is None:
            raise ValueError("Stage 2 artifact configuration lacks parent_lineage.")
        store.write_json(parent_path, dict(parent_lineage))

    references = {
        "model": _artifact_reference(root, paths["model"]),
        "preprocessor": _artifact_reference(root, paths["preprocessor"]),
        "input_schema": _artifact_reference(root, paths["input_schema"]),
        "metadata": _artifact_reference(root, paths["metadata"]),
        "hashes": _artifact_reference(root, paths["hashes"]),
        "model_config": _artifact_reference(root, "configs/model.yaml"),
        "feature_allowlist": _artifact_reference(root, "configs/features.yaml"),
    }
    if parent_lineage is not None:
        references["parent_lineage"] = _artifact_reference(
            root, paths["parent_lineage"]
        )
    manifest = {
        "contract_version": STAGE_BUNDLE_CONTRACT_VERSION,
        "stage": stage,
        "completed": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": references,
        "input_hashes": dict(input_hashes),
    }
    if before_completion_commit is not None:
        before_completion_commit()
    return store.write_json(completion_path, manifest)


def parent_lineage_record(bundle: LoadedStageBundle) -> dict[str, Any]:
    """Build Stage 2 lineage from verified Stage 1 completion references."""

    artifacts = bundle.manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArtifactIntegrityError("Stage 1 manifest lacks artifact references.")
    return {
        "stage1_directory": str(bundle.root),
        "stage1_stage": bundle.stage,
        "stage1_completion_manifest_sha256": sha256_file(
            bundle.root / DEFAULT_COMPLETION_MANIFEST
        ),
        "stage1_model": dict(artifacts["model"]),
        "stage1_preprocessor": dict(artifacts["preprocessor"]),
        "stage1_input_schema": dict(artifacts["input_schema"]),
        "stage1_artifacts": {
            str(name): dict(reference)
            for name, reference in sorted(artifacts.items())
            if isinstance(reference, Mapping)
        },
        "stage1_input_schema_content_sha256": bundle.input_schema.digest,
        "immutable_parent": True,
    }
