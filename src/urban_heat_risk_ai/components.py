"""Supporting physical-component XGBoost models and reconstruction.

All four models consume the same leakage-checked operational predictor matrix and
must be fitted with the same spatiotemporal folds as the direct UTCI model.  This
module does not choose folds: its fit helpers require the caller to pass explicit
inner-training/validation matrices, and outer refits never accept an evaluation
fold for early stopping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .physics import calculate_utci, convert_wind_to_10m

COMPONENT_A_TARGET = "local_minus_background_air_temperature_c"
COMPONENT_B_TARGET = "local_vapor_pressure_kpa"
COMPONENT_C_TARGET = "pedestrian_wind_log_adjustment"
COMPONENT_D_TARGET = "mrt_minus_local_air_temperature_c"
COMPONENT_TARGET_NAMES = (
    COMPONENT_A_TARGET,
    COMPONENT_B_TARGET,
    COMPONENT_C_TARGET,
    COMPONENT_D_TARGET,
)


@dataclass(frozen=True)
class ComponentColumns:
    """Source-column names used to derive the four training labels."""

    measured_air_temperature_c: str = "measured_air_temperature_c"
    measured_relative_humidity_percent: str = "measured_relative_humidity_pct"
    measured_pedestrian_wind_m_s: str = "measured_pedestrian_wind_speed_m_s"
    calculated_mrt_c: str = "calculated_mrt_c"
    background_air_temperature_c: str = "background_air_temperature_c"
    background_wind_m_s: str = "background_wind_speed_m_s"

    def required_columns(self) -> tuple[str, ...]:
        """Return all sensor/label columns required for target construction."""

        return (
            self.measured_air_temperature_c,
            self.measured_relative_humidity_percent,
            self.measured_pedestrian_wind_m_s,
            self.calculated_mrt_c,
            self.background_air_temperature_c,
            self.background_wind_m_s,
        )


def build_component_targets(
    observations: pd.DataFrame,
    *,
    columns: ComponentColumns | None = None,
) -> pd.DataFrame:
    """Derive component labels without modifying the observation table.

    Missing or physically invalid source measurements yield a missing target and
    are expected to be reported by schema/quality checks before fitting.  These
    derived labels are never permitted in an operational predictor matrix.
    """

    columns = columns or ComponentColumns()
    missing = sorted(set(columns.required_columns()) - set(observations.columns))
    if missing:
        raise ValueError("cannot build component targets; missing columns: " + ", ".join(missing))

    def numeric(name: str) -> NDArray[np.float64]:
        return pd.to_numeric(observations[name], errors="coerce").to_numpy(dtype=float)

    local_temperature = numeric(columns.measured_air_temperature_c)
    local_rh = numeric(columns.measured_relative_humidity_percent)
    local_wind = numeric(columns.measured_pedestrian_wind_m_s)
    mrt = numeric(columns.calculated_mrt_c)
    background_temperature = numeric(columns.background_air_temperature_c)
    background_wind = numeric(columns.background_wind_m_s)

    air_delta = local_temperature - background_temperature
    mrt_delta = mrt - local_temperature

    vapor_pressure = np.full(local_temperature.shape, np.nan, dtype=float)
    vapor_valid = (
        np.isfinite(local_temperature)
        & np.isfinite(local_rh)
        & (local_temperature > -243.04)
        & (local_rh >= 0.0)
        & (local_rh <= 100.0)
    )
    vapor_pressure[vapor_valid] = (
        0.61094
        * np.exp(
            (17.625 * local_temperature[vapor_valid])
            / (local_temperature[vapor_valid] + 243.04)
        )
        * local_rh[vapor_valid]
        / 100.0
    )

    wind_adjustment = np.full(local_wind.shape, np.nan, dtype=float)
    wind_valid = (
        np.isfinite(local_wind)
        & np.isfinite(background_wind)
        & (local_wind >= 0.0)
        & (background_wind >= 0.0)
    )
    wind_adjustment[wind_valid] = np.log1p(local_wind[wind_valid]) - np.log1p(
        background_wind[wind_valid]
    )

    return pd.DataFrame(
        {
            COMPONENT_A_TARGET: air_delta,
            COMPONENT_B_TARGET: vapor_pressure,
            COMPONENT_C_TARGET: wind_adjustment,
            COMPONENT_D_TARGET: mrt_delta,
        },
        index=observations.index.copy(),
    )


DEFAULT_COMPONENT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:absoluteerror",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "mae",
    "device": "cpu",
}
_ALLOWED_OBJECTIVES = {"reg:absoluteerror", "reg:pseudohubererror"}


def make_component_xgb_regressor(
    *,
    params: Mapping[str, Any] | None = None,
    random_seed: int = 2026,
    n_jobs: int = 1,
    early_stopping_rounds: int | None = None,
) -> Any:
    """Construct a CPU-default component :class:`xgboost.XGBRegressor`."""

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - dependency import check covers it
        raise RuntimeError("XGBoost is required to fit component models") from exc

    model_params = dict(DEFAULT_COMPONENT_XGB_PARAMS)
    if params:
        model_params.update(dict(params))
    objective = str(model_params.get("objective"))
    if objective not in _ALLOWED_OBJECTIVES:
        raise ValueError(
            "component objective must be reg:absoluteerror or reg:pseudohubererror"
        )
    if str(model_params.get("tree_method")) != "hist":
        raise ValueError("component models require tree_method='hist'")
    if str(model_params.get("eval_metric")) != "mae":
        raise ValueError("component models require eval_metric='mae'")
    if str(model_params.get("device", "cpu")) != "cpu":
        raise ValueError("component models default to and currently require CPU execution")

    model_params["random_state"] = int(random_seed)
    model_params["n_jobs"] = int(n_jobs)
    if early_stopping_rounds is not None:
        if early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        # XGBoost >=2.1 takes this scikit-interface parameter in the constructor.
        model_params["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRegressor(**model_params)


@dataclass(frozen=True)
class InnerFoldComponentFit:
    """One fitted inner-fold component model and its selected tree count."""

    model: Any
    best_tree_count: int


def _finite_training_target(y: ArrayLike, *, name: str) -> NDArray[np.float64]:
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} target is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} target contains missing or non-finite values")
    return values


def fit_component_inner_fold(
    x_train: Any,
    y_train: ArrayLike,
    x_inner_validation: Any,
    y_inner_validation: ArrayLike,
    *,
    params: Mapping[str, Any] | None = None,
    random_seed: int = 2026,
    n_jobs: int = 1,
    early_stopping_rounds: int = 50,
) -> InnerFoldComponentFit:
    """Fit one inner fold, allowing only that inner validation fold to stop trees."""

    train_target = _finite_training_target(y_train, name="inner training")
    validation_target = _finite_training_target(
        y_inner_validation, name="inner validation"
    )
    if len(x_train) != train_target.size or len(x_inner_validation) != validation_target.size:
        raise ValueError("predictor and target row counts do not match")

    model = make_component_xgb_regressor(
        params=params,
        random_seed=random_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        x_train,
        train_target,
        eval_set=[(x_inner_validation, validation_target)],
        verbose=False,
    )
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_tree_count = int(model.get_params()["n_estimators"])
    else:
        best_tree_count = int(best_iteration) + 1
    return InnerFoldComponentFit(model=model, best_tree_count=best_tree_count)


def median_selected_tree_count(inner_fold_tree_counts: Sequence[int]) -> int:
    """Return the deterministic median tree count for an outer refit."""

    counts = [int(value) for value in inner_fold_tree_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("inner_fold_tree_counts must contain positive integers")
    return max(1, int(np.floor(float(median(counts)) + 0.5)))


def refit_component_outer_model(
    x_outer_training: Any,
    y_outer_training: ArrayLike,
    *,
    inner_fold_tree_counts: Sequence[int],
    params: Mapping[str, Any] | None = None,
    random_seed: int = 2026,
    n_jobs: int = 1,
) -> Any:
    """Refit on all outer-training rows using the inner-fold median tree count.

    There is intentionally no outer-evaluation argument, so an outer held-out
    fold cannot influence early stopping or the selected number of trees.
    """

    target = _finite_training_target(y_outer_training, name="outer training")
    if len(x_outer_training) != target.size:
        raise ValueError("predictor and target row counts do not match")
    refit_params = dict(params or {})
    if "early_stopping_rounds" in refit_params:
        raise ValueError("outer component refits cannot include early_stopping_rounds")
    refit_params["n_estimators"] = median_selected_tree_count(inner_fold_tree_counts)
    model = make_component_xgb_regressor(
        params=refit_params,
        random_seed=random_seed,
        n_jobs=n_jobs,
        early_stopping_rounds=None,
    )
    model.fit(x_outer_training, target, verbose=False)
    return model


class ComponentXGBSuite:
    """Container for the four separately fitted component regressors."""

    def __init__(self) -> None:
        self.models_: dict[str, Any] = {}
        self.feature_names_: tuple[str, ...] | None = None

    def fit_outer_models(
        self,
        x_outer_training: Any,
        targets: Mapping[str, ArrayLike],
        *,
        inner_fold_tree_counts: Mapping[str, Sequence[int]],
        params_by_component: Mapping[str, Mapping[str, Any]] | None = None,
        random_seed: int = 2026,
        n_jobs: int = 1,
    ) -> ComponentXGBSuite:
        """Fit all four models on one common, already leakage-checked matrix."""

        missing_targets = sorted(set(COMPONENT_TARGET_NAMES) - set(targets))
        missing_counts = sorted(set(COMPONENT_TARGET_NAMES) - set(inner_fold_tree_counts))
        if missing_targets or missing_counts:
            raise ValueError(
                f"component inputs incomplete; missing targets={missing_targets}, "
                f"missing tree-count histories={missing_counts}"
            )
        self.models_.clear()
        if hasattr(x_outer_training, "columns"):
            self.feature_names_ = tuple(str(name) for name in x_outer_training.columns)
        for offset, name in enumerate(COMPONENT_TARGET_NAMES):
            self.models_[name] = refit_component_outer_model(
                x_outer_training,
                targets[name],
                inner_fold_tree_counts=inner_fold_tree_counts[name],
                params=(params_by_component or {}).get(name),
                random_seed=random_seed + offset,
                n_jobs=n_jobs,
            )
        return self

    def predict_raw(self, predictors: Any) -> pd.DataFrame:
        """Return raw predictions for all four components in fixed order."""

        if set(self.models_) != set(COMPONENT_TARGET_NAMES):
            raise RuntimeError("all four component models must be fitted before prediction")
        if self.feature_names_ is not None and hasattr(predictors, "columns"):
            received = tuple(str(name) for name in predictors.columns)
            if received != self.feature_names_:
                raise ValueError("component predictor feature order differs from fitted order")
        predictions = {
            name: np.asarray(self.models_[name].predict(predictors), dtype=float)
            for name in COMPONENT_TARGET_NAMES
        }
        index = predictors.index if hasattr(predictors, "index") else None
        return pd.DataFrame(predictions, index=index)


@dataclass
class ComponentModelBundle:
    """Fitted preprocessing and four models with an immutable predictor order.

    The bundle is intentionally data-agnostic and can be persisted by the
    project's artifact layer after a real-data run.  It contains no split logic;
    callers must pass only the designated training partition to
    :func:`fit_component_models`.
    """

    predictors: tuple[str, ...]
    preprocessor: Any
    models: dict[str, Any]
    selected_params_by_component: dict[str, dict[str, Any]]
    component_columns: ComponentColumns

    def transform(self, observations: pd.DataFrame) -> Any:
        """Transform an observation frame using the frozen predictor order."""

        missing = [name for name in self.predictors if name not in observations.columns]
        if missing:
            raise ValueError("component prediction data are missing: " + ", ".join(missing))
        return self.preprocessor.transform(observations.loc[:, list(self.predictors)])

    def predict_raw(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Predict the four component targets in their fixed order."""

        if set(self.models) != set(COMPONENT_TARGET_NAMES):
            raise RuntimeError("component bundle does not contain all four models")
        transformed = self.transform(observations)
        return pd.DataFrame(
            {
                name: np.asarray(self.models[name].predict(transformed), dtype=float)
                for name in COMPONENT_TARGET_NAMES
            },
            index=observations.index.copy(),
        )


def _guard_component_predictor_names(
    predictors: Sequence[str],
    columns: ComponentColumns,
) -> tuple[str, ...]:
    ordered = tuple(str(name) for name in predictors)
    if not ordered:
        raise ValueError("an explicit non-empty component predictor list is required")
    if len(set(ordered)) != len(ordered):
        raise ValueError("component predictor list contains duplicate names")
    exact_banned = {
        columns.measured_air_temperature_c,
        columns.measured_relative_humidity_percent,
        columns.measured_pedestrian_wind_m_s,
        columns.calculated_mrt_c,
        *COMPONENT_TARGET_NAMES,
        "calculated_utci_c",
        "utci_category",
        "calculated_wbgt_c",
        "label_uncertainty_c",
        "sample_id",
        "site_id",
        "sensor_id",
        "calibration_version",
        "quality_flag",
        "timestamp_utc",
        "latitude",
        "longitude",
        "split_role",
    }
    banned = sorted(set(ordered) & exact_banned)
    if banned:
        raise ValueError(
            "sensor-, target-, identifier-, or split-derived component predictors are banned: "
            + ", ".join(banned)
        )
    return ordered


def fit_component_models(
    training_observations: pd.DataFrame,
    *,
    predictors: Sequence[str],
    preprocessor_factory: Callable[[], Any],
    fixed_selected_params_by_component: Mapping[str, Mapping[str, Any]],
    columns: ComponentColumns | None = None,
    predictor_guard: Callable[[Sequence[str]], Any] | None = None,
    random_seed: int = 2026,
    n_jobs: int = 1,
) -> ComponentModelBundle:
    """Fit a simple final component bundle on an explicit training partition.

    ``preprocessor_factory`` is invoked and fitted exactly once on the supplied
    training rows.  Each component must provide fixed, already-selected XGBoost
    parameters including ``n_estimators``.  There is no validation/evaluation
    argument and no early stopping, preventing an outer or test partition from
    controlling the refit.
    """

    columns = columns or ComponentColumns()
    ordered_predictors = _guard_component_predictor_names(predictors, columns)
    if predictor_guard is not None:
        predictor_guard(ordered_predictors)
    missing_predictors = [
        name for name in ordered_predictors if name not in training_observations.columns
    ]
    if missing_predictors:
        raise ValueError("component training data are missing: " + ", ".join(missing_predictors))
    missing_params = sorted(
        set(COMPONENT_TARGET_NAMES) - set(fixed_selected_params_by_component)
    )
    if missing_params:
        raise ValueError(
            "fixed selected parameters are required for every component: "
            + ", ".join(missing_params)
        )
    selected_params: dict[str, dict[str, Any]] = {}
    for name in COMPONENT_TARGET_NAMES:
        params = dict(fixed_selected_params_by_component[name])
        if "n_estimators" not in params or int(params["n_estimators"]) <= 0:
            raise ValueError(f"{name} fixed parameters require positive n_estimators")
        if "early_stopping_rounds" in params:
            raise ValueError(
                f"{name} final fixed parameters cannot include early_stopping_rounds"
            )
        selected_params[name] = params

    targets = build_component_targets(training_observations, columns=columns)
    if targets.isna().any(axis=None):
        counts = targets.isna().sum()
        detail = ", ".join(
            f"{name}={int(count)}" for name, count in counts.items() if count
        )
        raise ValueError(
            "component training targets must be complete on the common training rows; "
            + detail
        )
    preprocessor = preprocessor_factory()
    if preprocessor is None or not hasattr(preprocessor, "fit_transform"):
        raise TypeError("preprocessor_factory must return an object with fit_transform")
    predictor_frame = training_observations.loc[:, ordered_predictors]
    transformed = preprocessor.fit_transform(predictor_frame)
    if transformed.shape[0] != len(training_observations):
        raise RuntimeError("component preprocessor changed the number of training rows")

    models: dict[str, Any] = {}
    for offset, name in enumerate(COMPONENT_TARGET_NAMES):
        model = make_component_xgb_regressor(
            params=selected_params[name],
            random_seed=random_seed + offset,
            n_jobs=n_jobs,
            early_stopping_rounds=None,
        )
        model.fit(transformed, targets[name].to_numpy(dtype=float), verbose=False)
        models[name] = model
    return ComponentModelBundle(
        predictors=ordered_predictors,
        preprocessor=preprocessor,
        models=models,
        selected_params_by_component=selected_params,
        component_columns=columns,
    )


def predict_components(
    bundle: ComponentModelBundle,
    observations: pd.DataFrame,
    *,
    include_reconstruction: bool = True,
    bounds: ReconstructionBounds | None = None,
) -> pd.DataFrame:
    """Return raw component predictions and, by default, physical reconstructions."""

    raw = bundle.predict_raw(observations)
    if not include_reconstruction:
        return raw
    air_column = bundle.component_columns.background_air_temperature_c
    wind_column = bundle.component_columns.background_wind_m_s
    missing = [name for name in (air_column, wind_column) if name not in observations]
    if missing:
        raise ValueError(
            "component reconstruction requires background inputs: " + ", ".join(missing)
        )
    reconstruction = reconstruct_components(
        background_air_temperature_c=pd.to_numeric(
            observations[air_column], errors="coerce"
        ).to_numpy(dtype=float),
        background_wind_m_s=pd.to_numeric(
            observations[wind_column], errors="coerce"
        ).to_numpy(dtype=float),
        predicted_air_temperature_delta_c=raw[COMPONENT_A_TARGET],
        predicted_local_vapor_pressure_kpa=raw[COMPONENT_B_TARGET],
        predicted_wind_log_adjustment=raw[COMPONENT_C_TARGET],
        predicted_mrt_delta_c=raw[COMPONENT_D_TARGET],
        bounds=bounds or ReconstructionBounds(),
    )
    reconstructed = reconstruction.to_frame(index=observations.index)
    return pd.concat([raw, reconstructed], axis=1)


@dataclass(frozen=True)
class ReconstructionBounds:
    """Plausibility limits used only to flag raw reconstructions."""

    air_temperature_c: tuple[float, float] = (-50.0, 65.0)
    relative_humidity_percent: tuple[float, float] = (0.0, 100.0)
    pedestrian_wind_m_s: tuple[float, float] = (0.0, 40.0)
    mean_radiant_temperature_c: tuple[float, float] = (-70.0, 120.0)

    def __post_init__(self) -> None:
        for name in (
            "air_temperature_c",
            "relative_humidity_percent",
            "pedestrian_wind_m_s",
            "mean_radiant_temperature_c",
        ):
            lower, upper = getattr(self, name)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"{name} bounds must be finite and strictly increasing")


@dataclass(frozen=True)
class ComponentReconstruction:
    """Raw reconstructed physical variables and per-variable validity flags."""

    predicted_air_temperature_delta_c_raw: NDArray[np.float64]
    predicted_local_vapor_pressure_kpa_raw: NDArray[np.float64]
    predicted_wind_log_adjustment_raw: NDArray[np.float64]
    predicted_mrt_delta_c_raw: NDArray[np.float64]
    local_air_temperature_c_raw: NDArray[np.float64]
    relative_humidity_percent_raw: NDArray[np.float64]
    pedestrian_wind_m_s_raw: NDArray[np.float64]
    mean_radiant_temperature_c_raw: NDArray[np.float64]
    air_temperature_valid: NDArray[np.bool_]
    vapor_pressure_valid: NDArray[np.bool_]
    relative_humidity_valid: NDArray[np.bool_]
    pedestrian_wind_valid: NDArray[np.bool_]
    mean_radiant_temperature_valid: NDArray[np.bool_]
    all_physical_values_valid: NDArray[np.bool_]

    @property
    def local_air_temperature_c(self) -> NDArray[np.float64]:
        return np.where(self.air_temperature_valid, self.local_air_temperature_c_raw, np.nan)

    @property
    def local_vapor_pressure_kpa(self) -> NDArray[np.float64]:
        return np.where(
            self.vapor_pressure_valid,
            self.predicted_local_vapor_pressure_kpa_raw,
            np.nan,
        )

    @property
    def relative_humidity_percent(self) -> NDArray[np.float64]:
        return np.where(
            self.relative_humidity_valid, self.relative_humidity_percent_raw, np.nan
        )

    @property
    def pedestrian_wind_m_s(self) -> NDArray[np.float64]:
        return np.where(self.pedestrian_wind_valid, self.pedestrian_wind_m_s_raw, np.nan)

    @property
    def mean_radiant_temperature_c(self) -> NDArray[np.float64]:
        return np.where(
            self.mean_radiant_temperature_valid,
            self.mean_radiant_temperature_c_raw,
            np.nan,
        )

    def to_frame(self, *, index: pd.Index | None = None) -> pd.DataFrame:
        """Return a reporting table without altering or clipping any raw value."""

        return pd.DataFrame(
            {
                "predicted_air_temperature_delta_c_raw": self.predicted_air_temperature_delta_c_raw,
                "predicted_local_vapor_pressure_kpa_raw": self.predicted_local_vapor_pressure_kpa_raw,
                "predicted_wind_log_adjustment_raw": self.predicted_wind_log_adjustment_raw,
                "predicted_mrt_delta_c_raw": self.predicted_mrt_delta_c_raw,
                "local_air_temperature_c_raw": self.local_air_temperature_c_raw,
                "relative_humidity_percent_raw": self.relative_humidity_percent_raw,
                "pedestrian_wind_m_s_raw": self.pedestrian_wind_m_s_raw,
                "mean_radiant_temperature_c_raw": self.mean_radiant_temperature_c_raw,
                "local_air_temperature_c": self.local_air_temperature_c,
                "local_vapor_pressure_kpa": self.local_vapor_pressure_kpa,
                "relative_humidity_percent": self.relative_humidity_percent,
                "pedestrian_wind_m_s": self.pedestrian_wind_m_s,
                "mean_radiant_temperature_c": self.mean_radiant_temperature_c,
                "air_temperature_valid": self.air_temperature_valid,
                "vapor_pressure_valid": self.vapor_pressure_valid,
                "relative_humidity_valid": self.relative_humidity_valid,
                "pedestrian_wind_valid": self.pedestrian_wind_valid,
                "mean_radiant_temperature_valid": self.mean_radiant_temperature_valid,
                "all_physical_values_valid": self.all_physical_values_valid,
            },
            index=index,
        )


def reconstruct_components(
    *,
    background_air_temperature_c: ArrayLike,
    background_wind_m_s: ArrayLike,
    predicted_air_temperature_delta_c: ArrayLike,
    predicted_local_vapor_pressure_kpa: ArrayLike,
    predicted_wind_log_adjustment: ArrayLike,
    predicted_mrt_delta_c: ArrayLike,
    bounds: ReconstructionBounds | None = None,
) -> ComponentReconstruction:
    """Reconstruct local physical inputs while retaining invalid raw predictions."""

    bounds = bounds or ReconstructionBounds()
    (
        background_temperature,
        background_wind,
        air_delta,
        vapor_pressure,
        wind_delta,
        mrt_delta,
    ) = np.broadcast_arrays(
        np.asarray(background_air_temperature_c, dtype=float),
        np.asarray(background_wind_m_s, dtype=float),
        np.asarray(predicted_air_temperature_delta_c, dtype=float),
        np.asarray(predicted_local_vapor_pressure_kpa, dtype=float),
        np.asarray(predicted_wind_log_adjustment, dtype=float),
        np.asarray(predicted_mrt_delta_c, dtype=float),
    )
    local_temperature = background_temperature + air_delta
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        saturation = 0.61094 * np.exp(
            (17.625 * local_temperature) / (local_temperature + 243.04)
        )
        local_rh = 100.0 * vapor_pressure / saturation
        local_wind = np.expm1(wind_delta + np.log1p(background_wind))
    local_mrt = local_temperature + mrt_delta

    air_valid = (
        np.isfinite(local_temperature)
        & (local_temperature >= bounds.air_temperature_c[0])
        & (local_temperature <= bounds.air_temperature_c[1])
    )
    vapor_valid = (
        np.isfinite(vapor_pressure)
        & np.isfinite(saturation)
        & (vapor_pressure >= 0.0)
        & (vapor_pressure <= saturation)
    )
    rh_valid = (
        np.isfinite(local_rh)
        & (local_rh >= bounds.relative_humidity_percent[0])
        & (local_rh <= bounds.relative_humidity_percent[1])
    )
    wind_valid = (
        np.isfinite(local_wind)
        & (local_wind >= bounds.pedestrian_wind_m_s[0])
        & (local_wind <= bounds.pedestrian_wind_m_s[1])
    )
    mrt_valid = (
        np.isfinite(local_mrt)
        & (local_mrt >= bounds.mean_radiant_temperature_c[0])
        & (local_mrt <= bounds.mean_radiant_temperature_c[1])
    )
    all_valid = air_valid & vapor_valid & rh_valid & wind_valid & mrt_valid

    def copied(values: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(values).copy()

    return ComponentReconstruction(
        predicted_air_temperature_delta_c_raw=copied(air_delta),
        predicted_local_vapor_pressure_kpa_raw=copied(vapor_pressure),
        predicted_wind_log_adjustment_raw=copied(wind_delta),
        predicted_mrt_delta_c_raw=copied(mrt_delta),
        local_air_temperature_c_raw=copied(local_temperature),
        relative_humidity_percent_raw=copied(local_rh),
        pedestrian_wind_m_s_raw=copied(local_wind),
        mean_radiant_temperature_c_raw=copied(local_mrt),
        air_temperature_valid=copied(air_valid),
        vapor_pressure_valid=copied(vapor_valid),
        relative_humidity_valid=copied(rh_valid),
        pedestrian_wind_valid=copied(wind_valid),
        mean_radiant_temperature_valid=copied(mrt_valid),
        all_physical_values_valid=copied(all_valid),
    )


@dataclass(frozen=True)
class ComponentUTCIResult:
    """Component-based UTCI with wind-profile and applicability diagnostics."""

    component_utci_c: NDArray[np.float64]
    wind_speed_10m_m_s_raw: NDArray[np.float64]
    wind_profile_applicable: NDArray[np.bool_]
    wind_profile_flags: NDArray[np.object_]
    wind_speed_10m_sensitivity_by_roughness_m: NDArray[np.object_]
    utci_applicable: NDArray[np.bool_]

    def to_frame(self, *, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component_utci_c": self.component_utci_c,
                "wind_speed_10m_m_s_raw": self.wind_speed_10m_m_s_raw,
                "wind_profile_applicable": self.wind_profile_applicable,
                "wind_profile_flags": self.wind_profile_flags,
                "wind_speed_10m_sensitivity_by_roughness_m": (
                    self.wind_speed_10m_sensitivity_by_roughness_m
                ),
                "utci_applicable": self.utci_applicable,
            },
            index=index,
        )


def calculate_component_utci(
    reconstruction: ComponentReconstruction,
    *,
    measurement_height_m: ArrayLike,
    roughness_length_m: ArrayLike,
    displacement_height_m: ArrayLike = 0.0,
    neutral_stability: bool = True,
    minimum_measurement_height_m: float | None = None,
    maximum_measurement_height_m: float | None = None,
    sensitivity_roughness_lengths_m: Sequence[float] = (),
    utci_limits: Mapping[str, Sequence[float]] | None = None,
) -> ComponentUTCIResult:
    """Convert reconstructed wind to 10 m and calculate component UTCI.

    Invalid physical/profile rows receive ``NaN``.  Raw component values remain
    available in ``reconstruction`` and are never replaced by clipped values.
    """

    shape = reconstruction.local_air_temperature_c_raw.shape
    height, roughness, displacement = np.broadcast_arrays(
        np.asarray(measurement_height_m, dtype=float),
        np.asarray(roughness_length_m, dtype=float),
        np.asarray(displacement_height_m, dtype=float),
        np.empty(shape, dtype=float),
    )[:3]
    wind_10m = np.full(shape, np.nan, dtype=float)
    wind_applicable = np.zeros(shape, dtype=bool)
    wind_flags = np.empty(shape, dtype=object)
    wind_sensitivity = np.empty(shape, dtype=object)

    for index in np.ndindex(shape):
        if not reconstruction.pedestrian_wind_valid[index]:
            wind_flags[index] = "invalid_reconstructed_pedestrian_wind"
            wind_sensitivity[index] = {}
            continue
        result = convert_wind_to_10m(
            float(reconstruction.pedestrian_wind_m_s_raw[index]),
            float(height[index]),
            float(roughness[index]),
            displacement_height_m=float(displacement[index]),
            neutral_stability=neutral_stability,
            minimum_measurement_height_m=minimum_measurement_height_m,
            maximum_measurement_height_m=maximum_measurement_height_m,
            sensitivity_roughness_lengths_m=sensitivity_roughness_lengths_m,
        )
        wind_10m[index] = result.wind_speed_10m_m_s
        wind_applicable[index] = result.applicable
        wind_flags[index] = ";".join(result.flags)
        wind_sensitivity[index] = result.sensitivity_by_roughness_m

    component_utci = np.asarray(
        calculate_utci(
            reconstruction.local_air_temperature_c_raw,
            reconstruction.mean_radiant_temperature_c_raw,
            wind_10m,
            reconstruction.relative_humidity_percent_raw,
            out_of_range="nan",
            limits=utci_limits,
        ),
        dtype=float,
    )
    applicable = (
        reconstruction.all_physical_values_valid
        & wind_applicable
        & np.isfinite(component_utci)
    )
    component_utci = np.where(applicable, component_utci, np.nan)
    return ComponentUTCIResult(
        component_utci_c=component_utci,
        wind_speed_10m_m_s_raw=wind_10m,
        wind_profile_applicable=wind_applicable,
        wind_profile_flags=wind_flags,
        wind_speed_10m_sensitivity_by_roughness_m=wind_sensitivity,
        utci_applicable=applicable,
    )


@dataclass(frozen=True)
class DisagreementWarningRule:
    """Frozen warning threshold learned only from development OOF predictions."""

    threshold_c: float
    quantile: float
    n_development_oof: int
    source_partition: str = "development_oof"

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold_c) or self.threshold_c < 0.0:
            raise ValueError("disagreement threshold must be finite and non-negative")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("disagreement quantile must be strictly between 0 and 1")
        if self.n_development_oof <= 0:
            raise ValueError("disagreement rule requires development OOF support")
        if self.source_partition != "development_oof":
            raise ValueError("disagreement rule provenance must remain development_oof")


def learn_disagreement_warning_threshold(
    direct_development_oof_c: ArrayLike,
    component_development_oof_c: ArrayLike,
    *,
    source_partition: str,
    quantile: float = 0.95,
    minimum_samples: int = 20,
) -> DisagreementWarningRule:
    """Learn a direct/component warning threshold from development OOF only."""

    allowed_sources = {"development_oof", "development_out_of_fold_predictions"}
    if source_partition not in allowed_sources:
        raise ValueError(
            "disagreement threshold may be learned only from development out-of-fold predictions"
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if isinstance(direct_development_oof_c, pd.Series) and isinstance(
        component_development_oof_c, pd.Series
    ) and not direct_development_oof_c.index.equals(component_development_oof_c.index):
        raise ValueError("development OOF prediction indexes must be identical")
    direct = np.asarray(direct_development_oof_c, dtype=float).reshape(-1)
    component = np.asarray(component_development_oof_c, dtype=float).reshape(-1)
    if direct.shape != component.shape:
        raise ValueError("development OOF predictions must cover identical observations")
    common = np.isfinite(direct) & np.isfinite(component)
    disagreement = np.abs(direct[common] - component[common])
    if disagreement.size < minimum_samples:
        raise ValueError(
            f"at least {minimum_samples} paired development OOF predictions are required"
        )
    threshold = float(np.quantile(disagreement, quantile, method="higher"))
    return DisagreementWarningRule(
        threshold_c=threshold,
        quantile=float(quantile),
        n_development_oof=int(disagreement.size),
    )


def compare_direct_and_component(
    direct_utci_c: ArrayLike | pd.Series,
    component_utci_c: ArrayLike | pd.Series,
    *,
    warning_rule: DisagreementWarningRule,
) -> pd.DataFrame:
    """Compare paired estimates on identical observations without averaging them."""

    if isinstance(direct_utci_c, pd.Series) and isinstance(component_utci_c, pd.Series):
        if not direct_utci_c.index.equals(component_utci_c.index):
            raise ValueError("direct and component predictions must have identical indexes")
        index: pd.Index | None = direct_utci_c.index
    else:
        index = None
    direct = np.asarray(direct_utci_c, dtype=float).reshape(-1)
    component = np.asarray(component_utci_c, dtype=float).reshape(-1)
    if direct.shape != component.shape:
        raise ValueError("direct and component predictions must cover identical observations")
    comparable = np.isfinite(direct) & np.isfinite(component)
    disagreement = np.full(direct.shape, np.nan, dtype=float)
    disagreement[comparable] = np.abs(direct[comparable] - component[comparable])
    warning = comparable & (disagreement > warning_rule.threshold_c)
    return pd.DataFrame(
        {
            "direct_utci_c": direct,
            "component_utci_c": component,
            "absolute_disagreement_c": disagreement,
            "disagreement_warning": warning,
            "comparable": comparable,
        },
        index=index,
    )


__all__ = [
    "COMPONENT_A_TARGET",
    "COMPONENT_B_TARGET",
    "COMPONENT_C_TARGET",
    "COMPONENT_D_TARGET",
    "COMPONENT_TARGET_NAMES",
    "ComponentColumns",
    "ComponentModelBundle",
    "ComponentReconstruction",
    "ComponentUTCIResult",
    "ComponentXGBSuite",
    "DisagreementWarningRule",
    "InnerFoldComponentFit",
    "ReconstructionBounds",
    "build_component_targets",
    "calculate_component_utci",
    "compare_direct_and_component",
    "fit_component_inner_fold",
    "fit_component_models",
    "learn_disagreement_warning_threshold",
    "make_component_xgb_regressor",
    "median_selected_tree_count",
    "predict_components",
    "reconstruct_components",
    "refit_component_outer_model",
]
