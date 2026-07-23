"""Predictor allow-lists, leakage prevention, and train-only preprocessing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import DEFAULT_FEATURE_CONFIG, load_yaml_config
from .errors import ConfigurationError, LeakageError

PredictorSet = Literal["core", "satellite_enhanced"]

_BUILTIN_BANNED_EXACT = frozenset(
    {
        "sample_id",
        "site_id",
        "sensor_id",
        "spatial_block_id",
        "weather_event_id",
        "date",
        "timestamp",
        "timestamp_utc",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "split",
        "split_role",
        "fold",
        "outer_fold",
        "inner_fold",
        "calibration_version",
        "quality_flag",
        "measurement_height_m",
        "measured_air_temperature_c",
        "measured_relative_humidity_pct",
        "measured_rh_pct",
        "measured_pedestrian_wind_m_s",
        "measured_pedestrian_wind_speed_m_s",
        "pedestrian_wind_m_s",
        "measured_globe_temperature_c",
        "globe_temperature_c",
        "calculated_mrt_c",
        "calculated_mean_radiant_temperature_c",
        "calculated_utci_c",
        "utci_category",
        "wbgt_c",
        "optional_wbgt_c",
        "label_uncertainty_c",
        "measured_vapor_pressure_kpa",
        "local_minus_background_air_temperature_c",
        "local_vapor_pressure_kpa",
        "pedestrian_wind_log_adjustment",
        "mrt_minus_local_air_temperature_c",
    }
)

_BUILTIN_SUSPICIOUS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"(^|_)(utci|wbgt|mrt)(_|$)",
        r"(^|_)(measured|observed|calculated|target|label|residual|prediction|predicted)(_|$)",
        r"(^|_)(sample|site|sensor)(_|$)",
        r"(^|_)(latitude|longitude|timestamp|calibration)(_|$)",
        r"(^|_)(split|fold)(_|$)",
        r"(^|_)(local).*(air|temp|humidity|rh|wind|globe|radiant|vapor)",
        r"(air|temp|humidity|rh|wind|globe|radiant|vapor).*(sensor|measured|observed)",
        r"(^|_)(sample|site|sensor|spatial_block|weather_event)_?id($|_)",
    )
)


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{context} must be a YAML list of column names.")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise ConfigurationError(f"{context} must contain only non-empty strings.")
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ConfigurationError(f"{context} contains duplicates: {duplicates}")
    return result


@dataclass(frozen=True)
class FeaturePolicy:
    """Canonical form of the version-controlled predictor policy."""

    version: str
    target: str
    core_predictors: tuple[str, ...]
    satellite_enhanced_additions: tuple[str, ...]
    categorical_core: tuple[str, ...] = ()
    categorical_enhanced: tuple[str, ...] = ()
    metadata_columns: tuple[str, ...] = ()
    label_and_sensor_columns: tuple[str, ...] = ()
    banned_exact: tuple[str, ...] = ()
    prohibited_prefixes: tuple[str, ...] = ()
    prohibited_substrings: tuple[str, ...] = ()
    suspicious_patterns: tuple[str, ...] = ()
    missingness_indicator_columns: tuple[str, ...] = ()
    missingness_suffix: str = "__missing"
    land_cover_categories: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> FeaturePolicy:
        """Normalize either supported feature-YAML layout."""

        permitted = config.get("permitted_predictor_sets", {})
        if permitted is None:
            permitted = {}
        if not isinstance(permitted, Mapping):
            raise ConfigurationError("permitted_predictor_sets must be a mapping.")

        raw_core = config.get("core_predictors", permitted.get("core"))
        core = _string_tuple(raw_core, context="core predictor allow-list")
        if not core:
            raise ConfigurationError("The core predictor allow-list cannot be empty.")

        raw_additions = config.get("satellite_enhanced_additions")
        enhanced_spec = permitted.get("satellite_enhanced")
        if raw_additions is None and isinstance(enhanced_spec, Mapping):
            raw_additions = enhanced_spec.get("additions", ())
        elif raw_additions is None and enhanced_spec is not None:
            full_enhanced = _string_tuple(
                enhanced_spec, context="satellite_enhanced predictor allow-list"
            )
            missing_core = [name for name in core if name not in full_enhanced]
            if missing_core:
                raise ConfigurationError(
                    "satellite_enhanced must include every core predictor; missing "
                    f"{missing_core}."
                )
            raw_additions = [name for name in full_enhanced if name not in core]
        additions = _string_tuple(raw_additions or (), context="satellite enhanced additions")
        overlap = sorted(set(core).intersection(additions))
        if overlap:
            raise ConfigurationError(f"Satellite additions repeat core predictors: {overlap}")

        categorical = config.get("categorical_features", {})
        if isinstance(categorical, Mapping):
            categorical_core = _string_tuple(
                categorical.get("core", ()), context="categorical_features.core"
            )
            categorical_enhanced = _string_tuple(
                categorical.get("satellite_enhanced", categorical_core),
                context="categorical_features.satellite_enhanced",
            )
        else:
            categorical_core = _string_tuple(categorical, context="categorical_features")
            categorical_enhanced = categorical_core

        for variant, categorical_names, allowed in (
            ("core", categorical_core, core),
            ("satellite_enhanced", categorical_enhanced, core + additions),
        ):
            undeclared = sorted(set(categorical_names).difference(allowed))
            if undeclared:
                raise ConfigurationError(
                    f"Categorical features for {variant} are not allow-listed: {undeclared}"
                )

        missing_config = config.get(
            "missingness_indicators_for", config.get("missingness_indicators", {})
        )
        if isinstance(missing_config, Mapping):
            explicit_missing = missing_config.get("columns", ())
            suffix = str(missing_config.get("suffix", "__missing"))
        else:
            explicit_missing = missing_config or ()
            suffix = "__missing"
        if not suffix or not re.fullmatch(r"[A-Za-z0-9_]+", suffix):
            raise ConfigurationError("The missingness indicator suffix must be alphanumeric/underscore.")

        target = config.get("target_column", config.get("target"))
        if not isinstance(target, str) or not target:
            raise ConfigurationError("Feature configuration needs target_column (or target).")
        version = config.get(
            "predictor_set_version", config.get("version", config.get("schema_version", "unknown"))
        )
        return cls(
            version=str(version),
            target=target,
            core_predictors=core,
            satellite_enhanced_additions=additions,
            categorical_core=categorical_core,
            categorical_enhanced=categorical_enhanced,
            metadata_columns=_string_tuple(
                config.get("metadata_columns", ()), context="metadata_columns"
            ),
            label_and_sensor_columns=_string_tuple(
                config.get(
                    "label_and_sensor_columns", config.get("sensor_label_columns", ())
                ),
                context="label_and_sensor_columns",
            ),
            banned_exact=_string_tuple(
                config.get("banned_exact", config.get("prohibited_exact", ())),
                context="banned_exact",
            ),
            prohibited_prefixes=_string_tuple(
                config.get("prohibited_prefixes", ()), context="prohibited_prefixes"
            ),
            prohibited_substrings=_string_tuple(
                config.get("prohibited_substrings", ()), context="prohibited_substrings"
            ),
            suspicious_patterns=_string_tuple(
                config.get("suspicious_patterns", ()), context="suspicious_patterns"
            ),
            missingness_indicator_columns=_string_tuple(
                explicit_missing, context="missingness indicator columns"
            ),
            missingness_suffix=suffix,
            land_cover_categories=_string_tuple(
                config.get("land_cover_categories", ()), context="land_cover_categories"
            ),
        )

    def allowed_predictors(self, predictor_set: PredictorSet = "core") -> tuple[str, ...]:
        if predictor_set == "core":
            return self.core_predictors
        if predictor_set == "satellite_enhanced":
            return self.core_predictors + self.satellite_enhanced_additions
        raise ConfigurationError(
            f"Unknown predictor set {predictor_set!r}; expected core or satellite_enhanced."
        )

    def categorical_predictors(self, predictor_set: PredictorSet = "core") -> tuple[str, ...]:
        if predictor_set == "core":
            return self.categorical_core
        if predictor_set == "satellite_enhanced":
            return self.categorical_enhanced
        raise ConfigurationError(f"Unknown predictor set: {predictor_set!r}")


def load_feature_policy(path: str | Path = DEFAULT_FEATURE_CONFIG) -> FeaturePolicy:
    """Load and validate the version-controlled feature policy."""

    return FeaturePolicy.from_mapping(load_yaml_config(path))


def resolve_predictors(
    policy_or_config: FeaturePolicy | Mapping[str, Any],
    predictor_set: PredictorSet = "core",
) -> tuple[str, ...]:
    """Resolve a core or enhanced allow-list in its fixed declared order."""

    policy = (
        policy_or_config
        if isinstance(policy_or_config, FeaturePolicy)
        else FeaturePolicy.from_mapping(policy_or_config)
    )
    return policy.allowed_predictors(predictor_set)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    reasons: tuple[str, ...]


class LeakageGuard:
    """Fail closed if an operational predictor is banned, suspicious, or undeclared."""

    def __init__(self, policy: FeaturePolicy, predictor_set: PredictorSet = "core") -> None:
        self.policy = policy
        self.predictor_set = predictor_set
        self.allowed = policy.allowed_predictors(predictor_set)
        try:
            self._configured_patterns = tuple(
                re.compile(pattern) for pattern in policy.suspicious_patterns
            )
        except re.error as exc:
            raise ConfigurationError(f"Invalid suspicious feature-name regex: {exc}") from exc

    def inspect(self, columns: Sequence[str]) -> tuple[LeakageFinding, ...]:
        findings: list[LeakageFinding] = []
        allowed_set = set(self.allowed)
        configured_banned = {_normalize_name(name) for name in self.policy.banned_exact}
        configured_banned.update(_normalize_name(name) for name in self.policy.label_and_sensor_columns)
        configured_banned.update(_normalize_name(name) for name in self.policy.metadata_columns)

        for original in columns:
            if not isinstance(original, str) or not original:
                findings.append(LeakageFinding(str(original), ("not a non-empty column name",)))
                continue
            name = _normalize_name(original)
            reasons: list[str] = []
            if original not in allowed_set:
                reasons.append("not declared in the selected predictor allow-list")
            if name in _BUILTIN_BANNED_EXACT or name in configured_banned:
                reasons.append("explicitly prohibited metadata, sensor, label, or target field")
            if name.endswith("_id"):
                reasons.append("identifier-like columns are not operational predictors")
            if any(name.startswith(_normalize_name(prefix)) for prefix in self.policy.prohibited_prefixes):
                reasons.append("matches a prohibited prefix")
            if any(
                _normalize_name(part) and _normalize_name(part) in name
                for part in self.policy.prohibited_substrings
            ):
                reasons.append("contains a prohibited target/sensor substring")
            if any(pattern.search(name) for pattern in _BUILTIN_SUSPICIOUS):
                reasons.append("looks like target-, sensor-, identifier-, split-, or coordinate-derived data")
            # Configured patterns are a second line of defense for undeclared names.
            # Explicit satellite QA predictors are intentionally allow-listed and can
            # contain words such as "quality" without being local-sensor quality flags.
            if original not in allowed_set and any(
                pattern.search(original) for pattern in self._configured_patterns
            ):
                reasons.append("matches a configured suspicious-name pattern")
            if reasons:
                findings.append(LeakageFinding(original, tuple(dict.fromkeys(reasons))))
        return tuple(findings)

    def validate(self, columns: Sequence[str]) -> tuple[str, ...]:
        """Return fixed-order names or raise before any model sees the matrix."""

        duplicates = sorted({name for name in columns if list(columns).count(name) > 1})
        if duplicates:
            raise LeakageError(f"Duplicate predictor columns are not allowed: {duplicates}")
        findings = self.inspect(columns)
        if findings:
            details = "; ".join(
                f"{finding.column!r}: {', '.join(finding.reasons)}" for finding in findings
            )
            raise LeakageError(
                f"Predictor leakage guard rejected the model matrix ({self.predictor_set}): {details}. "
                "Use only the version-controlled allow-list and derive labels outside preprocessing."
            )
        submitted = set(columns)
        return tuple(name for name in self.allowed if name in submitted)


def find_leaky_feature_names(
    columns: Sequence[str],
    policy: FeaturePolicy,
    predictor_set: PredictorSet = "core",
) -> tuple[str, ...]:
    """Pure-name helper used by diagnostics and data-independent tests."""

    return tuple(finding.column for finding in LeakageGuard(policy, predictor_set).inspect(columns))


def build_predictor_frame(
    frame: pd.DataFrame,
    policy: FeaturePolicy,
    predictor_set: PredictorSet = "core",
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    """Select an operational matrix in declared order without altering ``frame``."""

    allowed = policy.allowed_predictors(predictor_set)
    missing = [name for name in allowed if name not in frame.columns]
    if require_all and missing:
        raise LeakageError(
            f"Cannot build {predictor_set} predictor matrix; required allow-listed columns are "
            f"absent: {missing}"
        )
    selected = tuple(name for name in allowed if name in frame.columns)
    LeakageGuard(policy, predictor_set).validate(selected)
    return frame.loc[:, selected].copy()


def derive_prespecified_interactions(
    frame: pd.DataFrame, feature_config: Mapping[str, Any]
) -> pd.DataFrame:
    """Return a copy with only explicitly configured deterministic interactions.

    This has no fitted state.  It deliberately supports only named multiplication
    contracts, avoiding arbitrary expressions or label-derived transformations.
    """

    result = frame.copy()
    interactions = feature_config.get("interactions", {})
    specs = interactions.get("prespecified", ()) if isinstance(interactions, Mapping) else ()
    if not isinstance(specs, Sequence) or isinstance(specs, str):
        raise ConfigurationError("interactions.prespecified must be a list.")
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ConfigurationError("Each prespecified interaction must be a mapping.")
        name = spec.get("name")
        inputs = spec.get("inputs")
        operation = spec.get("operation")
        if not isinstance(name, str) or operation != "multiply":
            raise ConfigurationError("Interactions require a name and operation: multiply.")
        input_names = _string_tuple(inputs, context=f"interaction {name}.inputs")
        if len(input_names) != 2:
            raise ConfigurationError(f"Interaction {name!r} must have exactly two inputs.")
        missing = [column for column in input_names if column not in result.columns]
        if missing:
            raise ValueError(f"Cannot derive interaction {name!r}; missing inputs: {missing}")
        result[name] = pd.to_numeric(result[input_names[0]], errors="coerce") * pd.to_numeric(
            result[input_names[1]], errors="coerce"
        )
    return result


def satellite_eligibility_mask(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    feature_config: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Return the single eligibility mask shared by enhanced and core comparisons.

    ``config`` may be the whole model configuration, its ``satellite`` section,
    or the ``eligibility`` section itself.  Missing required QA columns cause an
    actionable error rather than silently widening the eligible population.
    """

    satellite = config.get("satellite", config)
    if not isinstance(satellite, Mapping):
        raise ConfigurationError("satellite configuration must be a mapping.")
    eligibility = satellite.get("eligibility", satellite)
    if not isinstance(eligibility, Mapping):
        raise ConfigurationError("satellite.eligibility must be a mapping.")

    satellite_features = (
        feature_config.get("satellite_features", {})
        if isinstance(feature_config, Mapping)
        else {}
    )
    if not isinstance(satellite_features, Mapping):
        satellite_features = {}
    lst_column = str(satellite_features.get("thermal_value", "satellite_lst_c"))
    age_column = str(satellite_features.get("age", "satellite_image_age_hours"))
    source_column = str(satellite_features.get("source", "satellite_thermal_source"))
    missing_column = str(satellite_features.get("missingness", "satellite_lst_missing"))
    quality_spec = satellite_features.get("quality", ())
    quality_names = set(quality_spec if isinstance(quality_spec, Sequence) else ())
    quality_flag_column = (
        "satellite_lst_quality_flag"
        if "satellite_lst_quality_flag" in quality_names or not quality_names
        else next((name for name in quality_names if str(name).endswith("quality_flag")), "")
    )
    cloud_column = "satellite_cloud_fraction"
    overpass_columns = satellite_features.get(
        "overpass_cyclic", ("satellite_overpass_hour_sin", "satellite_overpass_hour_cos")
    )
    overpass_columns = _string_tuple(overpass_columns, context="satellite overpass columns")

    required = {lst_column, age_column, missing_column}
    allowed_sources = eligibility.get("allowed_sources")
    if allowed_sources:
        required.add(source_column)
    allowed_quality = eligibility.get("allowed_quality_flags")
    if allowed_quality:
        required.add(quality_flag_column)
    if "maximum_cloud_fraction" in eligibility:
        required.add(cloud_column)
    if eligibility.get("require_overpass_time", False):
        required.update(overpass_columns)
    absent = sorted(column for column in required if not column or column not in frame.columns)
    if absent:
        raise ValueError(
            "Cannot determine satellite eligibility because required thermal/QA columns are "
            f"absent: {absent}"
        )

    mask = pd.Series(True, index=frame.index, dtype=bool, name="satellite_eligible")
    lst = pd.to_numeric(frame[lst_column], errors="coerce")
    if eligibility.get("require_valid_lst", True):
        mask &= lst.notna() & np.isfinite(lst)
    declared_missing = frame[missing_column]
    if pd.api.types.is_bool_dtype(declared_missing.dtype):
        mask &= ~declared_missing.fillna(True)
    else:
        normalized_missing = declared_missing.astype("string").str.strip().str.lower()
        mask &= normalized_missing.isin({"0", "0.0", "false", "no", "valid"})

    age = pd.to_numeric(frame[age_column], errors="coerce")
    maximum_hours = eligibility.get("maximum_image_age_hours")
    if maximum_hours is None:
        maximum_days = eligibility.get("maximum_image_age_days")
        maximum_hours = float(maximum_days) * 24.0 if maximum_days is not None else np.inf
    maximum_hours = float(maximum_hours)
    if maximum_hours < 0:
        raise ConfigurationError("Maximum satellite image age cannot be negative.")
    mask &= age.notna() & np.isfinite(age) & age.ge(0.0) & age.le(maximum_hours)

    if allowed_sources:
        allowed = {str(item).strip().lower() for item in allowed_sources}
        mask &= frame[source_column].astype("string").str.strip().str.lower().isin(allowed)
    if allowed_quality:
        allowed = {str(item).strip().lower() for item in allowed_quality}
        mask &= frame[quality_flag_column].astype("string").str.strip().str.lower().isin(allowed)
    if "maximum_cloud_fraction" in eligibility:
        cloud = pd.to_numeric(frame[cloud_column], errors="coerce")
        mask &= cloud.notna() & cloud.between(0.0, float(eligibility["maximum_cloud_fraction"]))
    if "minimum_valid_pixel_fraction" in eligibility:
        valid_column = "satellite_valid_pixel_fraction"
        if valid_column not in frame.columns:
            raise ValueError(f"Satellite eligibility needs absent column: {valid_column}")
        valid = pd.to_numeric(frame[valid_column], errors="coerce")
        mask &= valid.notna() & valid.ge(float(eligibility["minimum_valid_pixel_fraction"]))
    if eligibility.get("require_overpass_time", False):
        for column in overpass_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            mask &= values.notna() & np.isfinite(values) & values.between(-1.0, 1.0)
    return mask


def select_identical_satellite_eligible_rows(
    core_frame: pd.DataFrame,
    enhanced_frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    feature_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Apply one enhanced eligibility mask to two index-identical model tables."""

    if not core_frame.index.equals(enhanced_frame.index):
        raise ValueError("Core and enhanced frames must have exactly identical row indexes.")
    mask = satellite_eligibility_mask(enhanced_frame, config, feature_config=feature_config)
    return core_frame.loc[mask].copy(), enhanced_frame.loc[mask].copy(), mask


class PredictorPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe train-fitted encoder with model-specific numeric handling.

    XGBoost receives the original numeric NaNs plus explicit missing indicators.
    Linear and neural comparisons receive train-median imputation and scaling;
    Random Forest receives train-median imputation without scaling.  Categorical
    unknowns are ignored by a fixed train-fitted one-hot encoder.
    """

    def __init__(
        self,
        policy: FeaturePolicy,
        *,
        predictor_set: PredictorSet = "core",
        predictors: Sequence[str] | None = None,
        model_kind: str = "xgboost",
        add_missing_indicators: bool = True,
        missing_token: str = "__MISSING__",
        strict_columns: bool = True,
    ) -> None:
        self.policy = policy
        self.predictor_set = predictor_set
        self.predictors = predictors
        self.model_kind = model_kind
        self.add_missing_indicators = add_missing_indicators
        self.missing_token = missing_token
        self.strict_columns = strict_columns

    @staticmethod
    def _frame(value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("PredictorPreprocessor requires a pandas DataFrame with named columns.")
        return value

    def _choose_predictors(self, frame: pd.DataFrame) -> tuple[str, ...]:
        allowed = self.policy.allowed_predictors(self.predictor_set)
        if self.predictors is None:
            extras = [name for name in frame.columns if name not in allowed]
            if extras:
                raise LeakageError(
                    "Preprocessor input contains columns outside the predictor allow-list: "
                    f"{extras}. Build the operational matrix first."
                )
            selected = tuple(name for name in allowed if name in frame.columns)
        else:
            selected = tuple(self.predictors)
        if not selected:
            raise LeakageError("No declared predictors were supplied to preprocessing.")
        return LeakageGuard(self.policy, self.predictor_set).validate(selected)

    def fit(self, X: pd.DataFrame, y: Any = None) -> PredictorPreprocessor:
        frame = self._frame(X)
        selected = self._choose_predictors(frame)
        missing = [name for name in selected if name not in frame.columns]
        if missing:
            raise ValueError(f"Training partition is missing predictor columns: {missing}")
        if self.strict_columns:
            extras = [name for name in frame.columns if name not in selected]
            if extras:
                raise LeakageError(f"Unexpected columns entered preprocessing: {extras}")

        categorical_declared = set(self.policy.categorical_predictors(self.predictor_set))
        categorical = tuple(name for name in selected if name in categorical_declared)
        numerical = tuple(name for name in selected if name not in categorical_declared)
        numeric_frame = frame.loc[:, numerical].apply(pd.to_numeric, errors="coerce")
        coercion_failures = {
            name: int((frame[name].notna() & numeric_frame[name].isna()).sum())
            for name in numerical
        }
        coercion_failures = {name: count for name, count in coercion_failures.items() if count}
        if coercion_failures:
            raise ValueError(
                "Numeric predictors contain non-numeric values (column: count): "
                f"{coercion_failures}"
            )
        if numerical and np.isinf(numeric_frame.to_numpy(dtype=float)).any():
            raise ValueError("Numeric predictors contain infinity; use NaN plus a missingness indicator.")

        model_key = self.model_kind.strip().lower()
        xgb_kinds = {"xgboost", "xgb", "direct_xgb", "component_xgb"}
        standardized_kinds = {"linear", "elastic_net", "neural", "neural_network", "mlp"}
        imputed_kinds = standardized_kinds | {"random_forest", "rf"}
        if model_key not in xgb_kinds | imputed_kinds:
            raise ConfigurationError(
                f"Unknown preprocessing model_kind {self.model_kind!r}; use xgboost, linear, "
                "neural_network, or random_forest."
            )

        self.predictor_names_in_ = selected
        self.numerical_columns_ = numerical
        self.categorical_columns_ = categorical
        if self.add_missing_indicators:
            explicit = set(self.policy.missingness_indicator_columns)
            self.missing_indicator_columns_ = tuple(
                name
                for name in numerical
                if name in explicit or (not explicit and bool(numeric_frame[name].isna().any()))
            )
        else:
            self.missing_indicator_columns_ = ()
        self.retain_numeric_nan_ = model_key in xgb_kinds
        self.standardize_numeric_ = model_key in standardized_kinds

        if numerical and not self.retain_numeric_nan_:
            self.numeric_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            imputed = self.numeric_imputer_.fit_transform(numeric_frame)
            if self.standardize_numeric_:
                self.numeric_scaler_ = StandardScaler()
                self.numeric_scaler_.fit(imputed)
        if categorical:
            encoded_input = self._categorical_input(frame)
            self.categorical_encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float64
            )
            self.categorical_encoder_.fit(encoded_input)
        self.feature_names_out_ = self._make_feature_names()
        self._assert_transformed_feature_provenance()
        return self

    def _assert_transformed_feature_provenance(self) -> None:
        """Fail closed if preprocessing emits a name without an allowed raw parent."""

        numerical = set(self.numerical_columns_)
        categorical = tuple(self.categorical_columns_)
        missing_suffix = self.policy.missingness_suffix
        invalid: list[str] = []
        for output_name in self.feature_names_out_:
            if output_name in numerical:
                continue
            if output_name.endswith(missing_suffix):
                parent = output_name[: -len(missing_suffix)]
                if parent in numerical:
                    continue
            if any(output_name.startswith(f"{parent}_") for parent in categorical):
                continue
            invalid.append(output_name)
        if invalid:
            raise LeakageError(
                "Preprocessing generated features without a declared operational parent: "
                f"{invalid}"
            )

    def _categorical_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, self.categorical_columns_].astype("string")
        return result.fillna(self.missing_token)

    def _make_feature_names(self) -> tuple[str, ...]:
        names = list(self.numerical_columns_)
        names.extend(
            f"{name}{self.policy.missingness_suffix}"
            for name in self.missing_indicator_columns_
        )
        if self.categorical_columns_:
            names.extend(
                self.categorical_encoder_.get_feature_names_out(
                    list(self.categorical_columns_)
                ).tolist()
            )
        if len(names) != len(set(names)):
            raise ConfigurationError("Preprocessing generated duplicate output feature names.")
        return tuple(names)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        frame = self._frame(X)
        missing = [name for name in self.predictor_names_in_ if name not in frame.columns]
        if missing:
            raise ValueError(f"Prediction partition is missing predictor columns: {missing}")
        if self.strict_columns:
            extras = [name for name in frame.columns if name not in self.predictor_names_in_]
            if extras:
                raise LeakageError(f"Unexpected columns entered preprocessing: {extras}")

        pieces: list[np.ndarray] = []
        if self.numerical_columns_:
            numeric_frame = frame.loc[:, self.numerical_columns_].apply(
                pd.to_numeric, errors="coerce"
            )
            original_nonmissing = frame.loc[:, self.numerical_columns_].notna()
            bad = original_nonmissing & numeric_frame.isna()
            if bool(bad.to_numpy().any()):
                columns = bad.columns[bad.any()].tolist()
                raise ValueError(f"Non-numeric values found in numeric predictors: {columns}")
            numeric = numeric_frame.to_numpy(dtype=float)
            if np.isinf(numeric).any():
                raise ValueError("Numeric predictors contain infinity.")
            if not self.retain_numeric_nan_:
                numeric = self.numeric_imputer_.transform(numeric_frame)
                if self.standardize_numeric_:
                    numeric = self.numeric_scaler_.transform(numeric)
            pieces.append(np.asarray(numeric, dtype=float))
            if self.missing_indicator_columns_:
                indicators = numeric_frame.loc[:, self.missing_indicator_columns_].isna()
                pieces.append(indicators.to_numpy(dtype=float))
        if self.categorical_columns_:
            encoded = self.categorical_encoder_.transform(self._categorical_input(frame))
            pieces.append(np.asarray(encoded, dtype=float))
        if not pieces:
            return np.empty((len(frame), 0), dtype=float)
        result = np.hstack(pieces)
        if result.shape[1] != len(self.feature_names_out_):
            raise RuntimeError("Preprocessing feature-order invariant failed.")
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)

    def transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a named transformed copy for diagnostics, never source mutation."""

        values = self.transform(X)
        return pd.DataFrame(values, index=X.index, columns=self.feature_names_out_)


def make_preprocessor(
    policy: FeaturePolicy,
    *,
    predictor_set: PredictorSet = "core",
    predictors: Sequence[str] | None = None,
    model_kind: str = "xgboost",
    preprocessing_config: Mapping[str, Any] | None = None,
) -> PredictorPreprocessor:
    """Construct (but do not fit) preprocessing from configuration."""

    config = preprocessing_config or {}
    numerical = config.get("numerical", {}) if isinstance(config, Mapping) else {}
    categorical = config.get("categorical", {}) if isinstance(config, Mapping) else {}
    return PredictorPreprocessor(
        policy,
        predictor_set=predictor_set,
        predictors=predictors,
        model_kind=model_kind,
        add_missing_indicators=bool(
            numerical.get("add_missingness_indicators", True)
            if isinstance(numerical, Mapping)
            else True
        ),
        missing_token=str(
            categorical.get("missing_token", "__MISSING__")
            if isinstance(categorical, Mapping)
            else "__MISSING__"
        ),
    )
