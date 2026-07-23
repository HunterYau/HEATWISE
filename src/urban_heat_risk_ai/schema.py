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

PREDICTION_METADATA = (
    "sample_id",
    "date",
    "timestamp_utc",
    "latitude",
    "longitude",
    "measurement_height_m",
)

REQUIRED_LABELS = (
    "measured_air_temperature_c",
    "measured_relative_humidity_pct",
    "measured_pedestrian_wind_speed_m_s",
    "measured_globe_temperature_c",
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
    }
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
    ) -> None:
        self.feature_policy = feature_policy
        self.predictor_set = predictor_set
        self.model_config = dict(model_config or {})
        self.require_spatial_block = require_spatial_block
        self.require_labels = require_labels
        self.require_split_role = require_split_role
        self.strict_unknown_columns = strict_unknown_columns

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

        metadata_contract = REQUIRED_METADATA if self.require_labels else PREDICTION_METADATA
        required = set(metadata_contract)
        if self.require_labels:
            required.update(REQUIRED_LABELS)
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
        if "spatial_block_id" not in frame.columns and not self.require_spatial_block:
            add(
                "warning",
                "spatial_block_not_yet_assigned",
                "spatial_block_id is absent; validation may precede deterministic block creation only.",
                column="spatial_block_id",
                hint="Run make-splits with the configured CRS and block size before training.",
            )

        known = set(REQUIRED_METADATA + REQUIRED_LABELS) | OPTIONAL_KNOWN_COLUMNS
        known.update(self.feature_policy.allowed_predictors("satellite_enhanced"))
        known.update(self.feature_policy.metadata_columns)
        known.update(self.feature_policy.label_and_sensor_columns)
        unknown = sorted(set(frame.columns).difference(known))
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

        non_nullable = set(metadata_contract)
        if self.require_labels:
            non_nullable.update(REQUIRED_LABELS)
        if not self.require_split_role:
            non_nullable.discard("split_role")
        if self.require_spatial_block:
            non_nullable.add("spatial_block_id")
        for column in sorted(non_nullable.intersection(frame.columns)):
            missing_mask = frame[column].isna()
            if pdt.is_object_dtype(frame[column].dtype) or pdt.is_string_dtype(frame[column].dtype):
                missing_mask |= frame[column].astype("string").str.strip().eq("").fillna(False)
            count = int(missing_mask.sum())
            if count:
                add(
                    "error",
                    "missing_nonnullable",
                    "Required metadata/sensor/label values may not be missing or blank.",
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
            if column == "timestamp_utc":
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
        self._validate_ids_roles_and_categories(frame, allowed_roles, add)
        self._validate_cross_field_physics(frame, add)
        self._validate_units(frame, units, add)

        return ValidationReport(len(frame), len(frame.columns), tuple(issues))

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
    ).validate(frame, units=units)


def validate_schema_or_raise(*args: Any, **kwargs: Any) -> ValidationReport:
    report = validate_schema(*args, **kwargs)
    report.raise_for_errors()
    return report
