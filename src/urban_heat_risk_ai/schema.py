"""Strict validation for user-supplied real observation tables.

The validator is diagnostic only: it never clips, casts, fills, sorts, drops,
deduplicates, or writes the source table.  Every error includes a corrective hint.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .errors import DataRequiredError, SchemaValidationError
from .features import FeaturePolicy, PredictorSet

Severity = Literal["error", "warning"]
ValidationStage = Literal["default", "stage1_public", "stage2_sensor", "stage_prediction"]

PUBLIC_REFERENCE_TARGET = "public_reference_utci_c"

STAGE1_PUBLIC_PROVENANCE_COLUMNS = (
    "public_source_name",
    "public_source_version",
    "public_source_license",
    "public_retrieved_at_utc",
    "public_target_method_version",
    "public_quality_flag",
)

STAGE1_OPTIONAL_PROVENANCE_COLUMNS = ("public_source_record_id",)

STAGE2_SENSOR_RAW_COLUMNS = (
    "measured_air_temperature_c",
    "measured_relative_humidity_pct",
    "measured_pedestrian_wind_speed_m_s",
    "measured_globe_temperature_c",
)

STAGE2_SENSOR_PROVENANCE_COLUMNS = (
    "sensor_id",
    "measurement_height_m",
    "calibration_version",
    "quality_flag",
)

REQUIRED_METADATA = (
    "sample_id",
    "site_id",
    "date",
    "timestamp_utc",
    "latitude",
    "longitude",
    "sensor_id",
    "measurement_height_m",
    "calibration_version",
    "quality_flag",
    "split_role",
)

STAGE1_REQUIRED_METADATA = (
    "sample_id",
    "site_id",
    "date",
    "timestamp_utc",
    "latitude",
    "longitude",
    *STAGE1_PUBLIC_PROVENANCE_COLUMNS,
    "split_role",
)

STAGE2_REQUIRED_METADATA = REQUIRED_METADATA

PREDICTION_METADATA = (
    "sample_id",
    "date",
    "timestamp_utc",
    "latitude",
    "longitude",
    "measurement_height_m",
)

STAGE_PREDICTION_METADATA = (
    "sample_id",
    "date",
    "timestamp_utc",
    "latitude",
    "longitude",
)

REQUIRED_LABELS = (
    *STAGE2_SENSOR_RAW_COLUMNS,
    "calculated_mrt_c",
    "calculated_utci_c",
    "utci_category",
    "label_uncertainty_c",
)

OPTIONAL_KNOWN_COLUMNS = frozenset(
    {
        "spatial_block_id",
        "weather_event_id",
        "timezone_name",
        "optional_wbgt_c",
        "wbgt_c",
        "calculated_mean_radiant_temperature_c",
        "measured_vapor_pressure_kpa",
        "local_minus_background_air_temperature_c",
        "local_vapor_pressure_kpa",
        "pedestrian_wind_log_adjustment",
        "mrt_minus_local_air_temperature_c",
        "sun_shade_group",
        "coast_distance_group",
        "time_of_day_group",
        PUBLIC_REFERENCE_TARGET,
        *STAGE1_PUBLIC_PROVENANCE_COLUMNS,
        *STAGE1_OPTIONAL_PROVENANCE_COLUMNS,
    }
)

STAGE1_PROHIBITED_EXACT_COLUMNS = frozenset(
    {
        *STAGE2_SENSOR_RAW_COLUMNS,
        *STAGE2_SENSOR_PROVENANCE_COLUMNS,
        "calculated_mrt_c",
        "calculated_mean_radiant_temperature_c",
        "calculated_utci_c",
        "utci_category",
        "optional_wbgt_c",
        "wbgt_c",
        "label_uncertainty_c",
        "measured_vapor_pressure_kpa",
        "local_minus_background_air_temperature_c",
        "local_vapor_pressure_kpa",
        "pedestrian_wind_log_adjustment",
        "mrt_minus_local_air_temperature_c",
    }
)

STAGE_PREDICTION_PROHIBITED_EXACT_COLUMNS = frozenset(
    {
        *STAGE1_PROHIBITED_EXACT_COLUMNS,
        *STAGE1_PUBLIC_PROVENANCE_COLUMNS,
        *STAGE1_OPTIONAL_PROVENANCE_COLUMNS,
        PUBLIC_REFERENCE_TARGET,
    }
)

_STAGE1_PROHIBITED_NAME = re.compile(
    r"(?:^|_)(?:sensor|measured|measurement|globe|calibrat(?:ed|ion)|local)(?:_|$)"
)
_STAGE1_PROHIBITED_TARGET_NAME = re.compile(
    r"(?:^|_)(?:calculated|utci|mrt|wbgt|label_uncertainty)(?:_|$)"
)

STRING_COLUMNS = frozenset(
    {
        "sample_id",
        "site_id",
        "date",
        "sensor_id",
        "spatial_block_id",
        "weather_event_id",
        "timezone_name",
        "calibration_version",
        "quality_flag",
        "split_role",
        "utci_category",
        "land_cover_class",
        "sun_shade_group",
        "coast_distance_group",
        "time_of_day_group",
        "satellite_thermal_source",
        "satellite_lst_quality_flag",
        *STAGE1_PUBLIC_PROVENANCE_COLUMNS,
        *STAGE1_OPTIONAL_PROVENANCE_COLUMNS,
    }
)

UNIT_BY_COLUMN = {
    "latitude": "degree",
    "longitude": "degree",
    "measurement_height_m": "m",
    "measured_air_temperature_c": "degC",
    "measured_relative_humidity_pct": "%",
    "measured_pedestrian_wind_speed_m_s": "m/s",
    "measured_globe_temperature_c": "degC",
    "calculated_mrt_c": "degC",
    "calculated_utci_c": "degC",
    PUBLIC_REFERENCE_TARGET: "degC",
    "optional_wbgt_c": "degC",
    "wbgt_c": "degC",
    "label_uncertainty_c": "degC",
    "background_air_temperature_c": "degC",
    "background_dew_point_c": "degC",
    "background_relative_humidity_pct": "%",
    "background_wind_speed_m_s": "m/s",
    "background_surface_pressure_pa": "Pa",
    "background_cloud_cover_pct": "%",
    "background_shortwave_radiation_w_m2": "W/m2",
    "background_precipitation_1h_mm": "mm",
    "background_precipitation_3h_mm": "mm",
    "background_weather_source_distance_m": "m",
    "background_weather_age_minutes": "min",
    "background_temperature_heating_rate_c_per_h": "degC/h",
    "background_cumulative_hot_hours_24h": "h",
    "elevation_m": "m",
    "slope_degrees": "degree",
    "distance_to_coast_m": "m",
    "distance_to_major_road_m": "m",
    "solar_elevation_degrees": "degree",
    "satellite_lst_c": "degC",
    "satellite_lst_minus_background_air_temperature_c": "degC",
    "satellite_image_age_hours": "h",
    "satellite_view_zenith_degrees": "degree",
}

_UNIT_ALIASES = {
    "degc": "degc",
    "c": "degc",
    "celsius": "degc",
    "°c": "degc",
    "%": "%",
    "percent": "%",
    "pct": "%",
    "m/s": "m/s",
    "m s-1": "m/s",
    "ms-1": "m/s",
    "pa": "pa",
    "w/m2": "w/m2",
    "w m-2": "w/m2",
    "mm": "mm",
    "m": "m",
    "meter": "m",
    "metre": "m",
    "min": "min",
    "minute": "min",
    "h": "h",
    "hour": "h",
    "degree": "degree",
    "degrees": "degree",
    "deg": "degree",
    "degc/h": "degc/h",
    "°c/h": "degc/h",
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    column: str | None = None
    row_count: int | None = None
    examples: tuple[str, ...] = ()
    hint: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Immutable, machine- and human-readable validation result."""

    row_count: int
    column_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(issue) for issue in self.issues],
        }

    def format_text(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Schema {status}: {self.row_count} rows, {self.column_count} columns, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings."
        ]
        for issue in self.issues:
            location = f" [{issue.column}]" if issue.column else ""
            count = f" ({issue.row_count} rows)" if issue.row_count is not None else ""
            lines.append(f"- {issue.severity.upper()} {issue.code}{location}{count}: {issue.message}")
            if issue.examples:
                lines.append(f"  Examples: {', '.join(issue.examples)}")
            if issue.hint:
                lines.append(f"  Fix: {issue.hint}")
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        if not self.is_valid:
            raise SchemaValidationError(self.format_text())


def load_observations(data_path: str | Path | None) -> pd.DataFrame:
    """Read an explicitly supplied CSV or Parquet observation table.

    No directory discovery, fallback filename, or output write is performed.
    """

    if data_path is None or not str(data_path).strip():
        raise DataRequiredError(
            "An explicit --data path is required. Supply a real .csv or .parquet observation "
            "table; this project does not search for or generate data."
        )
    path = Path(data_path).expanduser()
    if not path.exists():
        raise DataRequiredError(
            f"Observation file was not found: {path}. Supply an existing real CSV or Parquet "
            "file with --data."
        )
    if not path.is_file():
        raise DataRequiredError(f"The --data path must name a file, not a directory: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        raise DataRequiredError(f"Could not read observation file {path}: {exc}") from exc
    raise DataRequiredError(
        f"Unsupported observation format {suffix!r}. Use an explicit .csv or .parquet file."
    )


# Compatibility names used naturally by CLI callers.
load_data = load_observations
read_observations = load_observations


def _examples(series: pd.Series, mask: pd.Series, limit: int = 3) -> tuple[str, ...]:
    values = series.loc[mask].head(limit).tolist()
    return tuple(repr(value) for value in values)


def _range_for(column: str) -> tuple[float | None, float | None] | None:
    if not isinstance(column, str):
        return None
    exact = {
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
        "measurement_height_m": (math.nextafter(0.0, 1.0), 50.0),
        "measured_air_temperature_c": (-50.0, 65.0),
        "measured_relative_humidity_pct": (0.0, 100.0),
        "measured_pedestrian_wind_speed_m_s": (0.0, 40.0),
        "measured_globe_temperature_c": (-60.0, 120.0),
        "calculated_mrt_c": (-70.0, 120.0),
        "calculated_mean_radiant_temperature_c": (-70.0, 120.0),
        "calculated_utci_c": (-100.0, 100.0),
        PUBLIC_REFERENCE_TARGET: (-100.0, 100.0),
        "optional_wbgt_c": (-50.0, 65.0),
        "wbgt_c": (-50.0, 65.0),
        "label_uncertainty_c": (0.0, 30.0),
        "background_surface_pressure_pa": (50_000.0, 110_000.0),
        "background_shortwave_radiation_w_m2": (0.0, 1_400.0),
        "background_weather_source_distance_m": (0.0, 500_000.0),
        "background_weather_age_minutes": (0.0, 10_080.0),
        "background_temperature_heating_rate_c_per_h": (-30.0, 30.0),
        "background_cumulative_hot_hours_24h": (0.0, 24.0),
        "elevation_m": (-500.0, 9_000.0),
        "slope_degrees": (0.0, 90.0),
        "distance_to_coast_m": (0.0, 5_000_000.0),
        "distance_to_major_road_m": (0.0, 5_000_000.0),
        "solar_elevation_degrees": (-90.0, 90.0),
        "satellite_lst_c": (-80.0, 100.0),
        "satellite_lst_minus_background_air_temperature_c": (-100.0, 100.0),
        "satellite_image_age_hours": (0.0, None),
        "satellite_view_zenith_degrees": (0.0, 90.0),
        "satellite_lst_quality_score": (0.0, 1.0),
        "satellite_cloud_fraction": (0.0, 1.0),
        "satellite_valid_pixel_fraction": (0.0, 1.0),
        "satellite_lst_missing": (0.0, 1.0),
    }
    if column in exact:
        return exact[column]
    if column in {"ndvi", "ndbi", "ndwi"} or re.fullmatch(r"ndvi_mean_\d+m", column):
        return (-1.0, 1.0)
    if column.endswith(("_sin", "_cos")):
        return (-1.0, 1.0)
    if "fraction" in column or column == "sky_view_factor" or column == "albedo_proxy":
        return (0.0, 1.0)
    if re.fullmatch(r"(sky_view_factor|albedo_mean)_\d+m", column):
        return (0.0, 1.0)
    if "canopy_to_impervious_ratio" in column:
        return (0.0, None)
    if "mean_building_height_m" in column:
        return (0.0, 1_000.0)
    if column.startswith("background_air_temperature") or column == "background_dew_point_c":
        return (-80.0, 65.0)
    if column == "background_relative_humidity_pct" or column == "background_cloud_cover_pct":
        return (0.0, 100.0)
    if column == "background_wind_speed_m_s":
        return (0.0, 75.0)
    if column.startswith("background_precipitation_"):
        return (0.0, 500.0)
    # Products have wider but still physically finite screening bounds.
    if column in {
        "solar_elevation_x_shortwave",
        "shortwave_x_shade_fraction",
        "shortwave_x_canopy_fraction",
    }:
        return (-126_000.0, 126_000.0)
    if column in {
        "background_wind_x_sky_view_factor",
        "background_wind_x_building_fraction",
    }:
        return (0.0, 75.0)
    if column == "distance_to_coast_x_background_wind":
        return (0.0, 375_000_000.0)
    return None


def _expected_category(value: float) -> str:
    if value < 26.0:
        return "no_heat_stress"
    if value < 32.0:
        return "moderate"
    if value < 38.0:
        return "strong"
    if value < 46.0:
        return "very_strong"
    return "extreme"


def is_stage1_prohibited_column(column: str) -> bool:
    """Return whether a column violates the public-only Stage 1 boundary.

    The check is deliberately independent of dataframe contents so ingestion
    code can reject prohibited names before reading values.  The public target
    has its own explicit name and is never accepted through a generic
    ``calculated_utci_c`` or sensor-derived alias.
    """

    normalized = str(column).strip().lower()
    stage1_public_names = {
        PUBLIC_REFERENCE_TARGET,
        *STAGE1_PUBLIC_PROVENANCE_COLUMNS,
        *STAGE1_OPTIONAL_PROVENANCE_COLUMNS,
    }
    if normalized in stage1_public_names:
        return False
    return (
        normalized in STAGE1_PROHIBITED_EXACT_COLUMNS
        or _STAGE1_PROHIBITED_NAME.search(normalized) is not None
        or _STAGE1_PROHIBITED_TARGET_NAME.search(normalized) is not None
    )


def stage1_prohibited_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return sorted unique Stage 1 sensor/local/calibration column names."""

    return tuple(sorted({str(column) for column in columns if is_stage1_prohibited_column(column)}))


def is_stage_prediction_prohibited_column(column: str) -> bool:
    """Return whether a public, sensor, local, or target field entered prediction."""

    normalized = str(column).strip().lower()
    return (
        normalized in STAGE_PREDICTION_PROHIBITED_EXACT_COLUMNS
        or is_stage1_prohibited_column(normalized)
        or re.search(r"(?:^|_)(?:public|target|label|wbgt|utci_category)(?:_|$)", normalized)
        is not None
    )


def stage_prediction_prohibited_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return sorted prohibited names from a staged inference table."""

    return tuple(
        sorted(
            {
                str(column)
                for column in columns
                if is_stage_prediction_prohibited_column(column)
            }
        )
    )


@dataclass(frozen=True)
class StageSchemaProfile:
    """Declarative required-column contract for one training stage."""

    name: ValidationStage
    required_metadata: tuple[str, ...]
    required_target_inputs: tuple[str, ...]
    prohibited_columns: frozenset[str] = frozenset()


DEFAULT_SCHEMA_PROFILE = StageSchemaProfile(
    name="default",
    required_metadata=REQUIRED_METADATA,
    required_target_inputs=REQUIRED_LABELS,
)
STAGE1_SCHEMA_PROFILE = StageSchemaProfile(
    name="stage1_public",
    required_metadata=STAGE1_REQUIRED_METADATA,
    required_target_inputs=(PUBLIC_REFERENCE_TARGET,),
    prohibited_columns=STAGE1_PROHIBITED_EXACT_COLUMNS,
)
STAGE2_SCHEMA_PROFILE = StageSchemaProfile(
    name="stage2_sensor",
    required_metadata=STAGE2_REQUIRED_METADATA,
    required_target_inputs=STAGE2_SENSOR_RAW_COLUMNS,
)
STAGE_PREDICTION_SCHEMA_PROFILE = StageSchemaProfile(
    name="stage_prediction",
    required_metadata=STAGE_PREDICTION_METADATA,
    required_target_inputs=(),
    prohibited_columns=STAGE_PREDICTION_PROHIBITED_EXACT_COLUMNS,
)

SCHEMA_PROFILES: Mapping[ValidationStage, StageSchemaProfile] = {
    "default": DEFAULT_SCHEMA_PROFILE,
    "stage1_public": STAGE1_SCHEMA_PROFILE,
    "stage2_sensor": STAGE2_SCHEMA_PROFILE,
    "stage_prediction": STAGE_PREDICTION_SCHEMA_PROFILE,
}


def schema_profile_for_stage(stage: ValidationStage) -> StageSchemaProfile:
    """Return the immutable schema contract for ``stage``."""

    try:
        return SCHEMA_PROFILES[stage]
    except KeyError as exc:
        raise ValueError(
            "validation_stage must be one of: 'default', 'stage1_public', "
            "'stage2_sensor', 'stage_prediction'"
        ) from exc


class SchemaValidator:
    """Non-mutating validator parameterized by the predictor policy."""

    def __init__(
        self,
        feature_policy: FeaturePolicy,
        *,
        predictor_set: PredictorSet = "core",
        model_config: Mapping[str, Any] | None = None,
        require_spatial_block: bool = False,
        require_labels: bool = True,
        require_split_role: bool = True,
        strict_unknown_columns: bool = True,
        validation_stage: ValidationStage = "default",
    ) -> None:
        self.feature_policy = feature_policy
        self.predictor_set = predictor_set
        self.model_config = dict(model_config or {})
        self.require_spatial_block = require_spatial_block
        self.require_labels = require_labels
        self.require_split_role = require_split_role
        self.strict_unknown_columns = strict_unknown_columns
        self.validation_stage = validation_stage
        self.stage_profile = schema_profile_for_stage(validation_stage)
        if validation_stage == "stage_prediction" and predictor_set != "core":
            raise ValueError("stage_prediction validation uses the exact core predictor schema")

    def validate(
        self, frame: pd.DataFrame, *, units: Mapping[str, str] | None = None
    ) -> ValidationReport:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Schema validation requires a pandas DataFrame.")
        issues: list[ValidationIssue] = []

        def add(
            severity: Severity,
            code: str,
            message: str,
            *,
            column: str | None = None,
            row_count: int | None = None,
            examples: Sequence[str] = (),
            hint: str | None = None,
        ) -> None:
            issues.append(
                ValidationIssue(
                    severity,
                    code,
                    message,
                    column,
                    row_count,
                    tuple(examples),
                    hint,
                )
            )

        if frame.empty:
            add(
                "error",
                "empty_table",
                "The observation table has no rows.",
                hint="Supply the real site-time observation table; do not use a placeholder file.",
            )
        duplicate_column_names = frame.columns[frame.columns.duplicated()].tolist()
        if duplicate_column_names:
            add(
                "error",
                "duplicate_column_names",
                f"Column names are duplicated: {duplicate_column_names}",
                hint="Rename or remove duplicate source columns before validation.",
            )
        invalid_names = [
            str(name)
            for name in frame.columns
            if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
        ]
        if invalid_names:
            add(
                "error",
                "invalid_column_names",
                f"Columns must use exact lower snake_case names: {invalid_names}",
                hint="Use the canonical names in docs/DATA_SCHEMA.md and configs/features.yaml.",
            )
        if duplicate_column_names:
            return ValidationReport(len(frame), len(frame.columns), tuple(issues))

        if self.validation_stage == "default":
            metadata_contract = REQUIRED_METADATA if self.require_labels else PREDICTION_METADATA
            required_target_inputs = REQUIRED_LABELS if self.require_labels else ()
        else:
            metadata_contract = self.stage_profile.required_metadata
            required_target_inputs = (
                self.stage_profile.required_target_inputs if self.require_labels else ()
            )
        required = set(metadata_contract)
        required.update(required_target_inputs)
        if not self.require_split_role:
            required.discard("split_role")
        required.update(self.feature_policy.allowed_predictors(self.predictor_set))
        if self.require_spatial_block:
            required.add("spatial_block_id")
        for column in sorted(required.difference(frame.columns)):
            add(
                "error",
                "missing_required_column",
                "Required schema column is absent.",
                column=column,
                hint=(
                    "Run make-splits to create a separate manifest first."
                    if column == "spatial_block_id"
                    else "Add the real measured/operational field using the documented name and unit."
                ),
            )
        if self.validation_stage == "stage1_public":
            prohibited = stage1_prohibited_columns(tuple(str(column) for column in frame.columns))
            if prohibited:
                add(
                    "error",
                    "stage1_prohibited_columns",
                    "Stage 1 is public-only and cannot contain sensor, local-observation, "
                    f"calibration, or legacy calculated-target fields: {list(prohibited)}",
                    hint=(
                        "Remove these fields from the Stage 1 input table. Use "
                        f"{PUBLIC_REFERENCE_TARGET} with the required public provenance fields."
                    ),
                )
        elif self.validation_stage == "stage_prediction":
            prohibited = stage_prediction_prohibited_columns(
                tuple(str(column) for column in frame.columns)
            )
            if prohibited:
                add(
                    "error",
                    "stage_prediction_prohibited_columns",
                    "Staged prediction accepts operational predictors only; public-reference, "
                    f"sensor, local-observation, calibration, and target fields are forbidden: "
                    f"{list(prohibited)}",
                    hint="Pass only the frozen core feature schema and required inference metadata.",
                )
            exact_allowed = set(STAGE_PREDICTION_METADATA)
            exact_allowed.update(self.feature_policy.allowed_predictors("core"))
            extra = sorted(
                {str(column) for column in frame.columns if column not in exact_allowed}
            )
            if extra:
                add(
                    "error",
                    "stage_prediction_extra_columns",
                    f"Columns fall outside the exact frozen core inference schema: {extra}",
                    hint="Remove every non-core field; never let identifiers or targets reach inference.",
                )
        if (
            self.validation_stage != "stage_prediction"
            and "spatial_block_id" not in frame.columns
            and not self.require_spatial_block
        ):
            add(
                "warning",
                "spatial_block_not_yet_assigned",
                "spatial_block_id is absent; validation may precede deterministic block creation only.",
                column="spatial_block_id",
                hint="Run make-splits with the configured CRS and block size before training.",
            )

        known = set(REQUIRED_METADATA + REQUIRED_LABELS) | OPTIONAL_KNOWN_COLUMNS
        known.update(STAGE1_REQUIRED_METADATA)
        known.update(STAGE1_OPTIONAL_PROVENANCE_COLUMNS)
        known.add(PUBLIC_REFERENCE_TARGET)
        known.update(self.feature_policy.allowed_predictors("satellite_enhanced"))
        known.update(self.feature_policy.metadata_columns)
        known.update(self.feature_policy.label_and_sensor_columns)
        unknown = sorted({str(column) for column in frame.columns if column not in known})
        if unknown and self.strict_unknown_columns:
            add(
                "error",
                "unknown_columns",
                f"Columns are outside the versioned real-data schema: {unknown}",
                hint="Correct typos or explicitly version/document legitimate metadata; never auto-select it.",
            )

        validation_config = self.model_config.get("validation", {})
        if not isinstance(validation_config, Mapping):
            validation_config = {}
        allowed_roles = set(
            validation_config.get(
                "allowed_split_roles", ("unassigned", "development", "calibration", "final_test")
            )
        )
        warning_fraction = float(validation_config.get("warn_missing_fraction", 0.20))
        error_fraction = float(validation_config.get("error_missing_fraction", 0.80))
        if not 0 <= warning_fraction <= error_fraction <= 1:
            add(
                "error",
                "invalid_missingness_config",
                "Missingness thresholds must satisfy 0 <= warning <= error <= 1.",
                hint="Correct validation.warn_missing_fraction/error_missing_fraction in model.yaml.",
            )
            warning_fraction, error_fraction = 0.20, 0.80

        stage2_physics_columns = {
            *STAGE2_SENSOR_RAW_COLUMNS,
            "measurement_height_m",
        }
        stage2_eligible = pd.Series(True, index=frame.index, dtype=bool)
        if self.validation_stage == "stage2_sensor":
            staged_config = self.model_config.get("training_stages", {})
            if not isinstance(staged_config, Mapping):
                staged_config = {}
            common_config = staged_config.get("common", {})
            if not isinstance(common_config, Mapping):
                common_config = {}
            stage2_config = staged_config.get("stage2", {})
            if not isinstance(stage2_config, Mapping):
                stage2_config = {}
            role = str(common_config.get("training_split_role", "development"))
            allowed_quality = {
                str(value).strip().lower()
                for value in stage2_config.get("allowed_quality_flags", ("pass",))
            }
            if {"split_role", "quality_flag"}.issubset(frame.columns):
                stage2_eligible = (
                    frame["split_role"].astype("string").str.strip().eq(role)
                    & frame["quality_flag"]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .isin(allowed_quality)
                ).fillna(False)

        non_nullable = set(metadata_contract)
        non_nullable.update(required_target_inputs)
        if not self.require_split_role:
            non_nullable.discard("split_role")
        if self.require_spatial_block:
            non_nullable.add("spatial_block_id")
        for column in sorted(non_nullable.intersection(frame.columns)):
            missing_mask = frame[column].isna()
            if pdt.is_object_dtype(frame[column].dtype) or pdt.is_string_dtype(frame[column].dtype):
                missing_mask |= frame[column].astype("string").str.strip().eq("").fillna(False)
            if (
                self.validation_stage == "stage2_sensor"
                and column in stage2_physics_columns
            ):
                missing_mask &= stage2_eligible
            count = int(missing_mask.sum())
            if count:
                add(
                    "error",
                    "missing_nonnullable",
                    (
                        "Required Stage 2 target-production values may not be "
                        "missing or blank on eligible training rows."
                        if column in stage2_physics_columns
                        and self.validation_stage == "stage2_sensor"
                        else "Required metadata/sensor/label values may not be missing or blank."
                    ),
                    column=column,
                    row_count=count,
                    hint="Resolve the source/provenance gap; the validator will not impute labels or metadata.",
                )

        predictors = self.feature_policy.allowed_predictors(self.predictor_set)
        for column in predictors:
            if column not in frame.columns:
                continue
            fraction = float(frame[column].isna().mean()) if len(frame) else 1.0
            if fraction >= 1.0:
                add(
                    "error",
                    "all_missing_predictor",
                    "Allow-listed predictor is entirely missing.",
                    column=column,
                    row_count=len(frame),
                    hint="Supply the operational feature or revise and version the preregistered allow-list.",
                )
            elif fraction >= error_fraction:
                add(
                    "error",
                    "excessive_predictor_missingness",
                    f"Missing fraction {fraction:.1%} reaches the configured error threshold.",
                    column=column,
                    row_count=int(frame[column].isna().sum()),
                    hint="Investigate upstream coverage before fitting; do not fill from labels.",
                )
            elif fraction >= warning_fraction:
                add(
                    "warning",
                    "high_predictor_missingness",
                    f"Missing fraction is {fraction:.1%}.",
                    column=column,
                    row_count=int(frame[column].isna().sum()),
                    hint="Document coverage and retain the configured missingness indicator.",
                )

        categorical_columns = set(STRING_COLUMNS)
        categorical_columns.update(self.feature_policy.categorical_predictors(self.predictor_set))
        for column in frame.columns:
            if column in {"timestamp_utc", "public_retrieved_at_utc"}:
                continue
            if column in categorical_columns:
                if not (
                    pdt.is_object_dtype(frame[column].dtype)
                    or pdt.is_string_dtype(frame[column].dtype)
                    or isinstance(frame[column].dtype, pd.CategoricalDtype)
                ):
                    add(
                        "error",
                        "wrong_dtype",
                        f"Expected a string/categorical field, found {frame[column].dtype}.",
                        column=column,
                        hint="Preserve identifiers/classes as strings in the source table.",
                    )
                continue
            if column in known and column not in {"date"} and not pdt.is_numeric_dtype(
                frame[column].dtype
            ):
                add(
                    "error",
                    "wrong_dtype",
                    f"Expected a numeric field, found {frame[column].dtype}.",
                    column=column,
                    hint="Use real numeric values and NaN for missingness; textual sentinels are invalid.",
                )

        # Explicit textual sentinel detection catches values before numeric conversion.
        sentinel_pattern = re.compile(r"^(?:-?9999(?:\.0)?|missing|null|n/?a)$", re.IGNORECASE)
        for column in frame.columns:
            if pdt.is_object_dtype(frame[column].dtype) or pdt.is_string_dtype(frame[column].dtype):
                strings = frame[column].astype("string").str.strip()
                bad = strings.str.match(sentinel_pattern, na=False)
                if bool(bad.any()):
                    add(
                        "error",
                        "textual_missing_sentinel",
                        "Textual/sentinel missing values are prohibited.",
                        column=column,
                        row_count=int(bad.sum()),
                        examples=_examples(frame[column], bad),
                        hint="Encode genuine numeric missingness as NaN; do not use -9999 or text tokens.",
                    )

        for column in frame.columns:
            physical_range = _range_for(column)
            if physical_range is None or not pdt.is_numeric_dtype(frame[column].dtype):
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            finite_bad = values.notna() & ~np.isfinite(values)
            if (
                self.validation_stage == "stage2_sensor"
                and column in stage2_physics_columns
            ):
                finite_bad &= stage2_eligible
            if bool(finite_bad.any()):
                add(
                    "error",
                    "nonfinite_numeric",
                    "Infinity is invalid; numerical missingness must be NaN.",
                    column=column,
                    row_count=int(finite_bad.sum()),
                    examples=_examples(frame[column], finite_bad),
                    hint="Correct the upstream calculation and replace only genuinely missing values with NaN.",
                )
            low, high = physical_range
            bad = pd.Series(False, index=frame.index)
            if low is not None:
                bad |= values < low
            if high is not None:
                bad |= values > high
            if (
                self.validation_stage == "stage2_sensor"
                and column in stage2_physics_columns
            ):
                bad &= stage2_eligible
            if bool(bad.any()):
                range_text = f"[{low if low is not None else '-inf'}, {high if high is not None else 'inf'}]"
                add(
                    "error",
                    "physical_range",
                    f"Values fall outside the documented plausible range {range_text}.",
                    column=column,
                    row_count=int(bad.sum()),
                    examples=_examples(frame[column], bad),
                    hint="Verify source units/calibration; the validator will not clip observations.",
                )

        if "satellite_lst_missing" in frame.columns:
            values = pd.to_numeric(frame["satellite_lst_missing"], errors="coerce")
            bad = values.notna() & ~values.isin([0, 1])
            if bool(bad.any()):
                add(
                    "error",
                    "invalid_binary_indicator",
                    "Satellite LST missingness must be exactly boolean or 0/1.",
                    column="satellite_lst_missing",
                    row_count=int(bad.sum()),
                    examples=_examples(frame["satellite_lst_missing"], bad),
                )

        self._validate_timestamps(frame, add)
        if self.validation_stage == "stage1_public":
            self._validate_public_provenance(frame, add)
        self._validate_ids_roles_and_categories(frame, allowed_roles, add)
        self._validate_cross_field_physics(frame, add)
        self._validate_units(frame, units, add)

        return ValidationReport(len(frame), len(frame.columns), tuple(issues))

    def _validate_public_provenance(self, frame: pd.DataFrame, add: Any) -> None:
        retrieved_column = "public_retrieved_at_utc"
        if retrieved_column in frame.columns:
            invalid: list[Any] = []
            naive: list[Any] = []
            non_utc: list[Any] = []
            for value in frame[retrieved_column].tolist():
                if pd.isna(value):
                    continue
                try:
                    parsed = pd.Timestamp(value)
                except (TypeError, ValueError, OverflowError):
                    invalid.append(value)
                    continue
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    naive.append(value)
                elif parsed.utcoffset().total_seconds() != 0:
                    non_utc.append(value)
            for code, values, message, hint in (
                (
                    "invalid_public_retrieval_timestamp",
                    invalid,
                    "public_retrieved_at_utc contains unparseable values.",
                    "Record the actual public-source retrieval instant as ISO-8601 UTC.",
                ),
                (
                    "naive_public_retrieval_timestamp",
                    naive,
                    "public_retrieved_at_utc must be timezone-aware.",
                    "Attach the verified UTC offset; do not infer one during validation.",
                ),
                (
                    "non_utc_public_retrieval_timestamp",
                    non_utc,
                    "public_retrieved_at_utc must carry a zero UTC offset.",
                    "Convert the retrieval instant to Z or +00:00 upstream.",
                ),
            ):
                if values:
                    add(
                        "error",
                        code,
                        message,
                        column=retrieved_column,
                        row_count=len(values),
                        examples=tuple(repr(value) for value in values[:3]),
                        hint=hint,
                    )

        stages_config = self.model_config.get("training_stages", {})
        if not isinstance(stages_config, Mapping):
            stages_config = {}
        stage1_config = stages_config.get("stage1", {})
        if not isinstance(stage1_config, Mapping):
            stage1_config = {}
        configured_flags = stage1_config.get("allowed_quality_flags", ())
        allowed_flags = {str(value) for value in configured_flags}
        if allowed_flags and "public_quality_flag" in frame.columns:
            supplied = frame["public_quality_flag"].astype("string")
            invalid_flags = supplied.notna() & ~supplied.isin(allowed_flags)
            if bool(invalid_flags.any()):
                add(
                    "error",
                    "invalid_public_quality_flag",
                    f"public_quality_flag must be one of {sorted(allowed_flags)}.",
                    column="public_quality_flag",
                    row_count=int(invalid_flags.sum()),
                    examples=_examples(frame["public_quality_flag"], invalid_flags),
                    hint="Apply the frozen Stage 1 public-source quality policy upstream.",
                )

    @staticmethod
    def _validate_timestamps(frame: pd.DataFrame, add: Any) -> None:
        if "timestamp_utc" not in frame.columns:
            return
        invalid: list[Any] = []
        naive: list[Any] = []
        non_utc: list[Any] = []
        parsed_by_position: dict[Any, pd.Timestamp] = {}
        for position, value in enumerate(frame["timestamp_utc"].tolist()):
            if pd.isna(value):
                continue
            try:
                parsed = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError):
                invalid.append(value)
                continue
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                naive.append(value)
                continue
            if parsed.utcoffset().total_seconds() != 0:
                non_utc.append(value)
                continue
            parsed_by_position[position] = parsed
        if invalid:
            add(
                "error",
                "invalid_timestamp",
                "timestamp_utc contains unparseable values.",
                column="timestamp_utc",
                row_count=len(invalid),
                examples=tuple(repr(value) for value in invalid[:3]),
                hint="Use ISO-8601 UTC timestamps such as 2026-07-20T18:00:00Z.",
            )
        if naive:
            add(
                "error",
                "naive_timestamp",
                "timestamp_utc must be timezone-aware.",
                column="timestamp_utc",
                row_count=len(naive),
                examples=tuple(repr(value) for value in naive[:3]),
                hint="Attach the verified UTC offset; do not assume a timezone during validation.",
            )
        if non_utc:
            add(
                "error",
                "non_utc_timestamp",
                "timestamp_utc values must carry a zero UTC offset (Z or +00:00).",
                column="timestamp_utc",
                row_count=len(non_utc),
                examples=tuple(repr(value) for value in non_utc[:3]),
                hint="Convert instants to UTC upstream while preserving the separate local date.",
            )
        if "date" in frame.columns:
            invalid_dates: list[Any] = []
            for value in frame["date"].dropna().tolist():
                try:
                    parsed_date = pd.Timestamp(value)
                    if parsed_date.strftime("%Y-%m-%d") != str(value):
                        invalid_dates.append(value)
                except (TypeError, ValueError, OverflowError):
                    invalid_dates.append(value)
            if invalid_dates:
                add(
                    "error",
                    "invalid_local_date",
                    "date must use the exact YYYY-MM-DD local-civil-date representation.",
                    column="date",
                    row_count=len(invalid_dates),
                    examples=tuple(repr(value) for value in invalid_dates[:3]),
                    hint="Derive and verify local dates upstream using the documented site timezone.",
                )

    def _validate_ids_roles_and_categories(
        self, frame: pd.DataFrame, allowed_roles: set[str], add: Any
    ) -> None:
        if "sample_id" in frame.columns:
            duplicates = frame["sample_id"].notna() & frame["sample_id"].duplicated(keep=False)
            if bool(duplicates.any()):
                add(
                    "error",
                    "duplicate_sample_id",
                    "sample_id must be globally unique.",
                    column="sample_id",
                    row_count=int(duplicates.sum()),
                    examples=_examples(frame["sample_id"], duplicates),
                    hint="Resolve duplicate ingestion; do not deduplicate silently.",
                )
        if {"site_id", "timestamp_utc"}.issubset(frame.columns):
            duplicated = frame.duplicated(["site_id", "timestamp_utc"], keep=False)
            if bool(duplicated.any()):
                add(
                    "warning",
                    "duplicate_site_timestamp",
                    "Multiple rows share a site and UTC instant; verify they are distinct observations.",
                    row_count=int(duplicated.sum()),
                    hint="Confirm sensor/height provenance and unique sample IDs.",
                )
        if "split_role" in frame.columns:
            normalized = frame["split_role"].astype("string")
            invalid = normalized.notna() & ~normalized.isin(allowed_roles)
            if bool(invalid.any()):
                add(
                    "error",
                    "invalid_split_role",
                    f"split_role must be one of {sorted(allowed_roles)}.",
                    column="split_role",
                    row_count=int(invalid.sum()),
                    examples=_examples(frame["split_role"], invalid),
                    hint="Use make-splits or correct the explicit role metadata.",
                )
        if "quality_flag" in frame.columns:
            allowed_quality = {"pass", "suspect", "fail"}
            normalized = frame["quality_flag"].astype("string").str.strip().str.lower()
            invalid = normalized.notna() & ~normalized.isin(allowed_quality)
            if bool(invalid.any()):
                add(
                    "error",
                    "invalid_quality_flag",
                    f"quality_flag must be one of {sorted(allowed_quality)}.",
                    column="quality_flag",
                    row_count=int(invalid.sum()),
                    examples=_examples(frame["quality_flag"], invalid),
                )
        if "land_cover_class" in frame.columns and self.feature_policy.land_cover_categories:
            allowed_land_cover = set(self.feature_policy.land_cover_categories)
            normalized_land_cover = frame["land_cover_class"].astype("string")
            invalid_land_cover = normalized_land_cover.notna() & ~normalized_land_cover.isin(
                allowed_land_cover
            )
            if bool(invalid_land_cover.any()):
                add(
                    "error",
                    "invalid_land_cover_class",
                    f"land_cover_class must be one of {sorted(allowed_land_cover)}.",
                    column="land_cover_class",
                    row_count=int(invalid_land_cover.sum()),
                    examples=_examples(frame["land_cover_class"], invalid_land_cover),
                    hint="Map the source classification through the documented versioned crosswalk.",
                )
        categories = {"no_heat_stress", "moderate", "strong", "very_strong", "extreme"}
        if "utci_category" in frame.columns:
            supplied = frame["utci_category"].astype("string")
            invalid = supplied.notna() & ~supplied.isin(categories)
            if bool(invalid.any()):
                add(
                    "error",
                    "invalid_utci_category",
                    f"UTCI category must be one of {sorted(categories)}.",
                    column="utci_category",
                    row_count=int(invalid.sum()),
                    examples=_examples(frame["utci_category"], invalid),
                )
            if "calculated_utci_c" in frame.columns and pdt.is_numeric_dtype(
                frame["calculated_utci_c"].dtype
            ):
                continuous = pd.to_numeric(frame["calculated_utci_c"], errors="coerce")
                expected = continuous.map(lambda x: _expected_category(x) if pd.notna(x) else pd.NA)
                mismatch = supplied.notna() & expected.notna() & supplied.ne(expected)
                if bool(mismatch.any()):
                    add(
                        "error",
                        "category_threshold_mismatch",
                        "utci_category disagrees with the fixed continuous UTCI thresholds.",
                        column="utci_category",
                        row_count=int(mismatch.sum()),
                        examples=_examples(frame["utci_category"], mismatch),
                        hint="Derive categories from continuous UTCI; never train a category classifier.",
                    )

    @staticmethod
    def _validate_cross_field_physics(frame: pd.DataFrame, add: Any) -> None:
        if {"background_dew_point_c", "background_air_temperature_c"}.issubset(frame.columns):
            dew = pd.to_numeric(frame["background_dew_point_c"], errors="coerce")
            air = pd.to_numeric(frame["background_air_temperature_c"], errors="coerce")
            bad = dew.notna() & air.notna() & dew.gt(air + 1.0)
            if bool(bad.any()):
                add(
                    "error",
                    "dew_point_above_air_temperature",
                    "Background dew point exceeds air temperature by more than 1 °C.",
                    column="background_dew_point_c",
                    row_count=int(bad.sum()),
                    examples=_examples(frame["background_dew_point_c"], bad),
                    hint="Verify weather units, timestamp alignment, and source quality.",
                )
        for left, right in (
            ("aspect_sin", "aspect_cos"),
            ("solar_azimuth_sin", "solar_azimuth_cos"),
            ("hour_sin", "hour_cos"),
            ("day_of_year_sin", "day_of_year_cos"),
            ("satellite_overpass_hour_sin", "satellite_overpass_hour_cos"),
        ):
            if {left, right}.issubset(frame.columns):
                x = pd.to_numeric(frame[left], errors="coerce")
                y = pd.to_numeric(frame[right], errors="coerce")
                norm = np.sqrt(x * x + y * y)
                bad = norm.notna() & ~norm.between(0.95, 1.05)
                if bool(bad.any()):
                    add(
                        "warning",
                        "cyclic_pair_norm",
                        f"The {left}/{right} pair is not approximately unit length.",
                        column=left,
                        row_count=int(bad.sum()),
                        hint="Recompute the sine/cosine pair from the documented raw angle/time.",
                    )
        if {"satellite_lst_c", "satellite_lst_missing"}.issubset(frame.columns):
            lst_missing = frame["satellite_lst_c"].isna()
            flag = pd.to_numeric(frame["satellite_lst_missing"], errors="coerce").eq(1)
            mismatch = lst_missing.ne(flag) & frame["satellite_lst_missing"].notna()
            if bool(mismatch.any()):
                add(
                    "error",
                    "satellite_missingness_mismatch",
                    "satellite_lst_missing is inconsistent with satellite_lst_c missingness.",
                    column="satellite_lst_missing",
                    row_count=int(mismatch.sum()),
                    hint="Correct the declared indicator without filling thermal values.",
                )

    @staticmethod
    def _validate_units(frame: pd.DataFrame, units: Mapping[str, str] | None, add: Any) -> None:
        supplied_units = units
        if supplied_units is None:
            attr_units = frame.attrs.get("units")
            supplied_units = attr_units if isinstance(attr_units, Mapping) else None
        if not supplied_units:
            return
        for column, supplied in supplied_units.items():
            if column not in frame.columns or column not in UNIT_BY_COLUMN:
                continue
            expected = UNIT_BY_COLUMN[column]
            expected_normalized = _UNIT_ALIASES.get(expected.lower(), expected.lower())
            supplied_normalized = _UNIT_ALIASES.get(str(supplied).strip().lower())
            if supplied_normalized != expected_normalized:
                add(
                    "error",
                    "unit_mismatch",
                    f"Unit metadata {supplied!r} disagrees with required unit {expected!r}.",
                    column=column,
                    hint="Convert values upstream and update truthful Parquet/unit metadata.",
                )


def validate_schema(
    frame: pd.DataFrame,
    feature_policy: FeaturePolicy,
    *,
    predictor_set: PredictorSet = "core",
    model_config: Mapping[str, Any] | None = None,
    require_spatial_block: bool = False,
    require_labels: bool = True,
    require_split_role: bool = True,
    strict_unknown_columns: bool = True,
    validation_stage: ValidationStage = "default",
    units: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Functional wrapper around :class:`SchemaValidator`."""

    return SchemaValidator(
        feature_policy,
        predictor_set=predictor_set,
        model_config=model_config,
        require_spatial_block=require_spatial_block,
        require_labels=require_labels,
        require_split_role=require_split_role,
        strict_unknown_columns=strict_unknown_columns,
        validation_stage=validation_stage,
    ).validate(frame, units=units)


def validate_schema_or_raise(*args: Any, **kwargs: Any) -> ValidationReport:
    report = validate_schema(*args, **kwargs)
    report.raise_for_errors()
    return report


def validate_stage1_schema(
    frame: pd.DataFrame,
    feature_policy: FeaturePolicy,
    *,
    predictor_set: PredictorSet = "core",
    model_config: Mapping[str, Any] | None = None,
    require_spatial_block: bool = False,
    require_split_role: bool = True,
    strict_unknown_columns: bool = True,
    units: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate a public-only Stage 1 training table.

    Stage 1 requires its distinct public UTCI reference and complete public
    provenance, and rejects sensor/local/calibration fields even when unknown
    column checking is relaxed.
    """

    return validate_schema(
        frame,
        feature_policy,
        predictor_set=predictor_set,
        model_config=model_config,
        require_spatial_block=require_spatial_block,
        require_labels=True,
        require_split_role=require_split_role,
        strict_unknown_columns=strict_unknown_columns,
        validation_stage="stage1_public",
        units=units,
    )


def validate_stage2_schema(
    frame: pd.DataFrame,
    feature_policy: FeaturePolicy,
    *,
    predictor_set: PredictorSet = "core",
    model_config: Mapping[str, Any] | None = None,
    require_spatial_block: bool = False,
    require_split_role: bool = True,
    strict_unknown_columns: bool = True,
    units: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate Stage 2 operational predictors plus raw sensor target inputs.

    Precomputed MRT, UTCI, UTCI category, and uncertainty columns are not
    required: Stage 2 derives its target from the four raw measurements using
    the frozen physical configuration.
    """

    return validate_schema(
        frame,
        feature_policy,
        predictor_set=predictor_set,
        model_config=model_config,
        require_spatial_block=require_spatial_block,
        require_labels=True,
        require_split_role=require_split_role,
        strict_unknown_columns=strict_unknown_columns,
        validation_stage="stage2_sensor",
        units=units,
    )


def validate_stage_prediction_schema(
    frame: pd.DataFrame,
    feature_policy: FeaturePolicy,
    *,
    model_config: Mapping[str, Any] | None = None,
    strict_unknown_columns: bool = True,
    units: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate the exact operational-input contract used by staged prediction.

    No sensor height, sensor provenance, public-reference provenance, split
    metadata, or target is accepted or required.
    """

    return validate_schema(
        frame,
        feature_policy,
        predictor_set="core",
        model_config=model_config,
        require_spatial_block=False,
        require_labels=False,
        require_split_role=False,
        strict_unknown_columns=strict_unknown_columns,
        validation_stage="stage_prediction",
        units=units,
    )


def validate_stage1_schema_or_raise(*args: Any, **kwargs: Any) -> ValidationReport:
    """Validate Stage 1 and raise :class:`SchemaValidationError` on errors."""

    report = validate_stage1_schema(*args, **kwargs)
    report.raise_for_errors()
    return report


def validate_stage2_schema_or_raise(*args: Any, **kwargs: Any) -> ValidationReport:
    """Validate Stage 2 and raise :class:`SchemaValidationError` on errors."""

    report = validate_stage2_schema(*args, **kwargs)
    report.raise_for_errors()
    return report


def validate_stage_prediction_schema_or_raise(
    *args: Any, **kwargs: Any
) -> ValidationReport:
    """Validate staged inference inputs and raise on any contract violation."""

    report = validate_stage_prediction_schema(*args, **kwargs)
    report.raise_for_errors()
    return report
