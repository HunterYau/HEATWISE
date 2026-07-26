"""Primary direct-UTCI XGBoost model and leakage-safe nested tuning.

Early stopping is confined to inner validation folds.  Once a trial is selected,
the corresponding outer model is refit on every outer-training observation using
the median inner-fold best tree count; the outer evaluation fold is never passed
to XGBoost as an evaluation set.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

LOGGER = logging.getLogger(__name__)

ADAPTATION_OVERRIDE_KEYS = frozenset(
    {
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "huber_slope",
    }
)


class TrainOnlyPreprocessor(Protocol):
    """Minimal preprocessing contract used by the model-selection code."""

    def fit_transform(self, x: pd.DataFrame, y: Any = None) -> Any: ...

    def transform(self, x: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class IndexFold:
    """Integer positions for a leakage-checked validation fold."""

    train: np.ndarray
    validation: np.ndarray
    name: str = "fold"

    def __post_init__(self) -> None:
        train = np.asarray(self.train, dtype=int)
        validation = np.asarray(self.validation, dtype=int)
        if train.ndim != 1 or validation.ndim != 1:
            raise ValueError("Fold indices must be one-dimensional integer arrays.")
        if not len(train) or not len(validation):
            raise ValueError(f"{self.name} has an empty train or validation partition.")
        overlap = np.intersect1d(train, validation)
        if overlap.size:
            raise ValueError(f"{self.name} train/validation indices overlap.")


@dataclass(frozen=True)
class FoldScore:
    """Inner-fold diagnostics retained for a completed Optuna trial."""

    name: str
    grouped_mae: float
    observation_mae: float
    high_utci_mae: float | None
    best_tree_count: int


@dataclass
class TuningResult:
    """Selected hyperparameters and fold-level early-stopping evidence."""

    best_params: dict[str, Any]
    objective_value: float
    selected_tree_count: int
    fold_scores: list[FoldScore]
    best_trial_number: int
    study: optuna.Study = field(repr=False)


@dataclass
class FittedDirectModel:
    """An outer/final model bundled with its train-fitted preprocessing object."""

    model: XGBRegressor
    preprocessor: TrainOnlyPreprocessor
    predictors: tuple[str, ...]
    objective: str
    tree_count: int
    model_variant: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.predictors if name not in frame.columns]
        if missing:
            raise ValueError(f"Prediction table is missing predictors: {missing}")
        transformed = self.preprocessor.transform(
            frame.loc[:, list(self.predictors)]
        )
        return np.asarray(self.model.predict(transformed), dtype=float)


def _cfg(config: Mapping[str, Any], name: str, default: Any) -> Any:
    return config.get(name, default)


def suggest_xgb_parameters(
    trial: optuna.Trial, search: Mapping[str, Any]
) -> dict[str, Any]:
    """Sample the preregistered primary search space."""

    objectives = list(
        _cfg(search, "objectives", ["reg:absoluteerror", "reg:pseudohubererror"])
    )
    objective = trial.suggest_categorical("objective", objectives)
    params: dict[str, Any] = {
        "objective": objective,
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(_cfg(search, "n_estimators_min", 300)),
            int(_cfg(search, "n_estimators_max", 1500)),
            step=int(_cfg(search, "n_estimators_step", 50)),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            int(_cfg(search, "max_depth_min", 3)),
            int(_cfg(search, "max_depth_max", 8)),
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(_cfg(search, "learning_rate_min", 0.01)),
            float(_cfg(search, "learning_rate_max", 0.10)),
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            float(_cfg(search, "subsample_min", 0.6)),
            float(_cfg(search, "subsample_max", 1.0)),
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            float(_cfg(search, "colsample_min", 0.6)),
            float(_cfg(search, "colsample_max", 1.0)),
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            float(_cfg(search, "min_child_weight_min", 1.0)),
            float(_cfg(search, "min_child_weight_max", 20.0)),
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            float(_cfg(search, "reg_alpha_min", 1.0e-8)),
            float(_cfg(search, "reg_alpha_max", 10.0)),
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            float(_cfg(search, "reg_lambda_min", 1.0e-3)),
            float(_cfg(search, "reg_lambda_max", 100.0)),
            log=True,
        ),
    }
    if bool(_cfg(search, "tune_gamma", True)):
        params["gamma"] = trial.suggest_float(
            "gamma",
            float(_cfg(search, "gamma_min", 0.0)),
            float(_cfg(search, "gamma_max", 5.0)),
        )
    else:
        params["gamma"] = float(_cfg(search, "gamma", 0.0))
    if objective == "reg:pseudohubererror":
        params["huber_slope"] = trial.suggest_float(
            "huber_slope",
            float(_cfg(search, "huber_slope_min", 0.5)),
            float(_cfg(search, "huber_slope_max", 5.0)),
            log=True,
        )
    return params


def make_regressor(
    params: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int = 1,
    early_stopping_rounds: int | None = None,
) -> XGBRegressor:
    """Create the deterministic CPU primary model with fixed core settings."""

    kwargs = dict(params)
    kwargs.update(
        tree_method="hist",
        device="cpu",
        eval_metric="mae",
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbosity=0,
    )
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRegressor(**kwargs)


def grouped_mae(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[Any] | None,
) -> float:
    """Average per-block MAE so large spatial-day blocks cannot dominate tuning."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if groups is None:
        return float(mean_absolute_error(truth, prediction))
    group_values = np.asarray(groups)
    if len(group_values) != len(truth):
        raise ValueError("Validation group labels must align with validation outcomes.")
    scores = [
        float(np.mean(np.abs(truth[group_values == group] - prediction[group_values == group])))
        for group in pd.unique(group_values)
    ]
    return float(np.mean(scores))


def select_refit_tree_count(
    fold_tree_counts: Sequence[int],
    *,
    minimum_trees: int,
    maximum_trees: int,
) -> int:
    """Round the fold median and clamp it to the preregistered refit range."""

    if not fold_tree_counts:
        raise ValueError("At least one inner-fold tree count is required.")
    if minimum_trees < 1 or maximum_trees < minimum_trees:
        raise ValueError("Invalid outer-refit tree-count bounds.")
    median_count = math.floor(float(np.median(fold_tree_counts)) + 0.5)
    return min(maximum_trees, max(minimum_trees, median_count))


def high_utci_mae(
    y_true: Sequence[float], y_pred: Sequence[float], threshold_c: float = 38.0
) -> float | None:
    """Return high-heat MAE, or ``None`` when the fold has no high-heat labels."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    mask = truth >= float(threshold_c)
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(truth[mask] - prediction[mask])))


def heat_weights(
    y: Sequence[float], *, threshold_c: float, multiplier: float
) -> np.ndarray:
    """Weights for the separately labelled, opt-in heat-weighted experiment."""

    if multiplier < 1.0:
        raise ValueError("Heat-weight multiplier must be at least 1.0.")
    labels = np.asarray(y, dtype=float)
    return np.where(labels >= threshold_c, float(multiplier), 1.0)


def piecewise_heat_weights(
    y: Sequence[float], *, thresholds_c: Sequence[float], weights: Sequence[float]
) -> np.ndarray:
    """Apply one configured weight below, between, and above heat thresholds."""

    thresholds = np.asarray(thresholds_c, dtype=float)
    configured_weights = np.asarray(weights, dtype=float)
    if thresholds.ndim != 1 or configured_weights.ndim != 1:
        raise ValueError("Heat thresholds and weights must be one-dimensional.")
    if len(configured_weights) != len(thresholds) + 1:
        raise ValueError("Heat weights must contain exactly one more value than thresholds.")
    if np.any(np.diff(thresholds) <= 0.0) or np.any(configured_weights <= 0.0):
        raise ValueError("Heat thresholds must increase and every weight must be positive.")
    labels = np.asarray(y, dtype=float)
    bins = np.searchsorted(thresholds, labels, side="right")
    return configured_weights[bins]


def _weights_for_variant(
    y: np.ndarray, experiment: Mapping[str, Any] | None
) -> tuple[np.ndarray | None, str]:
    experiment = experiment or {}
    enabled = bool(experiment.get("enabled", False))
    if not enabled:
        return None, "primary_unweighted"
    if bool(experiment.get("replace_primary", False)) and not bool(
        experiment.get("explicit_primary_override", False)
    ):
        raise ValueError(
            "A heat-weighted experiment cannot replace the preregistered unweighted "
            "primary model unless explicit_primary_override is true in configuration."
        )
    if "thresholds_c" in experiment or "weights" in experiment:
        weights = piecewise_heat_weights(
            y,
            thresholds_c=experiment.get("thresholds_c", (32.0, 38.0, 46.0)),
            weights=experiment.get("weights", (1.0, 1.5, 2.0, 3.0)),
        )
    else:
        weights = heat_weights(
            y,
            threshold_c=float(experiment.get("threshold_c", 38.0)),
            multiplier=float(experiment.get("multiplier", 2.0)),
        )
    return weights, "secondary_heat_weighted"


def tune_inner_cv(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    folds: Sequence[IndexFold],
    preprocessor_factory: Callable[[], TrainOnlyPreprocessor],
    config: Mapping[str, Any],
    validation_groups: Sequence[Any] | None = None,
    heat_weighted_experiment: Mapping[str, Any] | None = None,
    predictor_guard: Callable[[Sequence[str]], Any] | None = None,
) -> TuningResult:
    """Tune on leakage-checked inner folds and retain every best iteration."""

    y = np.asarray(target, dtype=float)
    if len(frame) != len(y):
        raise ValueError("Predictor rows and target values have different lengths.")
    if predictor_guard is not None:
        predictor_guard(tuple(str(column) for column in frame.columns))
    for fold in folds:
        for label, indices in (("train", fold.train), ("validation", fold.validation)):
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"{fold.name} {label} indices contain duplicates.")
            if np.any(indices < 0) or np.any(indices >= len(frame)):
                raise ValueError(f"{fold.name} {label} indices are outside the predictor table.")
    groups = None if validation_groups is None else np.asarray(validation_groups)
    if groups is not None and len(groups) != len(y):
        raise ValueError("Group labels and target values have different lengths.")

    seed = int(_cfg(config, "seed", 42))
    n_jobs = int(_cfg(config, "n_jobs", 1))
    trials = int(_cfg(config, "n_trials", 50))
    early_stopping = int(_cfg(config, "early_stopping_rounds", 50))
    search = _cfg(config, "search", config)
    high_threshold = float(_cfg(config, "high_utci_threshold_c", 38.0))
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner_name = str(_cfg(config, "pruner", "MedianPruner"))
    if pruner_name == "MedianPruner":
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    elif pruner_name in {"NopPruner", "none", "None"}:
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unsupported configured Optuna pruner: {pruner_name}")
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    scores_by_trial: dict[int, list[FoldScore]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_parameters(trial, search)
        fold_scores: list[FoldScore] = []
        for fold_number, fold in enumerate(folds):
            processor = preprocessor_factory()
            x_train = processor.fit_transform(frame.iloc[fold.train], y[fold.train])
            x_validation = processor.transform(frame.iloc[fold.validation])
            weights, variant = _weights_for_variant(
                y[fold.train], heat_weighted_experiment
            )
            model = make_regressor(
                params,
                seed=seed + fold_number,
                n_jobs=n_jobs,
                early_stopping_rounds=early_stopping,
            )
            fit_kwargs: dict[str, Any] = {
                "eval_set": [(x_validation, y[fold.validation])],
                "verbose": False,
            }
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            model.fit(x_train, y[fold.train], **fit_kwargs)
            prediction = np.asarray(model.predict(x_validation), dtype=float)
            best_iteration = getattr(model, "best_iteration", None)
            best_tree_count = (
                int(best_iteration) + 1
                if best_iteration is not None
                else int(params["n_estimators"])
            )
            fold_groups = None if groups is None else groups[fold.validation]
            fold_scores.append(
                FoldScore(
                    name=fold.name,
                    grouped_mae=grouped_mae(
                        y[fold.validation], prediction, fold_groups
                    ),
                    observation_mae=float(
                        mean_absolute_error(y[fold.validation], prediction)
                    ),
                    high_utci_mae=high_utci_mae(
                        y[fold.validation], prediction, high_threshold
                    ),
                    best_tree_count=best_tree_count,
                )
            )
            trial.report(
                float(np.mean([score.grouped_mae for score in fold_scores])),
                step=fold_number,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()
        scores_by_trial[trial.number] = fold_scores
        trial.set_user_attr(
            "fold_best_tree_counts", [score.best_tree_count for score in fold_scores]
        )
        trial.set_user_attr(
            "fold_high_utci_mae", [score.high_utci_mae for score in fold_scores]
        )
        trial.set_user_attr("model_variant", variant)
        return float(np.mean([score.grouped_mae for score in fold_scores]))

    timeout_value = _cfg(config, "timeout_seconds", None)
    study.optimize(
        objective,
        n_trials=trials,
        timeout=None if timeout_value is None else float(timeout_value),
        show_progress_bar=False,
    )
    selected_scores = scores_by_trial[study.best_trial.number]
    selected_count = select_refit_tree_count(
        [score.best_tree_count for score in selected_scores],
        minimum_trees=int(_cfg(config, "minimum_refit_trees", 1)),
        maximum_trees=int(
            _cfg(
                config,
                "maximum_refit_trees",
                _cfg(search, "n_estimators_max", 1500),
            )
        ),
    )
    best_params = dict(study.best_trial.params)
    best_params["n_estimators"] = selected_count
    LOGGER.info(
        "Selected direct XGBoost trial",
        extra={
            "trial": study.best_trial.number,
            "grouped_mae": study.best_value,
            "tree_count": selected_count,
        },
    )
    return TuningResult(
        best_params=best_params,
        objective_value=float(study.best_value),
        selected_tree_count=selected_count,
        fold_scores=selected_scores,
        best_trial_number=int(study.best_trial.number),
        study=study,
    )


def refit_outer_model(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    predictors: Sequence[str],
    preprocessor_factory: Callable[[], TrainOnlyPreprocessor],
    tuning: TuningResult,
    seed: int,
    n_jobs: int = 1,
    heat_weighted_experiment: Mapping[str, Any] | None = None,
    predictor_guard: Callable[[Sequence[str]], Any] | None = None,
) -> FittedDirectModel:
    """Refit a selected outer model without an outer evaluation set."""

    y = np.asarray(target, dtype=float)
    if predictor_guard is not None:
        predictor_guard(tuple(predictors))
    processor = preprocessor_factory()
    x_train = processor.fit_transform(frame.loc[:, list(predictors)], y)
    weights, variant = _weights_for_variant(y, heat_weighted_experiment)
    params = dict(tuning.best_params)
    params["n_estimators"] = int(tuning.selected_tree_count)
    model = make_regressor(params, seed=seed, n_jobs=n_jobs)
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights
    # Intentionally no eval_set: outer/test data must never control tree count.
    model.fit(x_train, y, **fit_kwargs)
    return FittedDirectModel(
        model=model,
        preprocessor=processor,
        predictors=tuple(predictors),
        objective=str(params["objective"]),
        tree_count=int(tuning.selected_tree_count),
        model_variant=variant,
    )


def fit_fixed_model(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    predictors: Sequence[str],
    preprocessor_factory: Callable[[], TrainOnlyPreprocessor],
    params: Mapping[str, Any],
    seed: int,
    n_jobs: int = 1,
    model_variant: str = "primary_unweighted",
    predictor_guard: Callable[[Sequence[str]], Any] | None = None,
    heat_weighted_experiment: Mapping[str, Any] | None = None,
) -> FittedDirectModel:
    """Fit fixed, previously selected settings on all allowed training rows."""

    if "n_estimators" not in params:
        raise ValueError("Fixed XGBoost parameters must include n_estimators.")
    if predictor_guard is not None:
        predictor_guard(tuple(predictors))
    y = np.asarray(target, dtype=float)
    processor = preprocessor_factory()
    x_train = processor.fit_transform(frame.loc[:, list(predictors)], y)
    model = make_regressor(params, seed=seed, n_jobs=n_jobs)
    weights, inferred_variant = _weights_for_variant(y, heat_weighted_experiment)
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights
    model.fit(x_train, y, **fit_kwargs)
    return FittedDirectModel(
        model=model,
        preprocessor=processor,
        predictors=tuple(predictors),
        objective=str(params.get("objective", "reg:absoluteerror")),
        tree_count=int(params["n_estimators"]),
        model_variant=(inferred_variant if weights is not None else model_variant),
    )


def validate_adaptation_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return safe continued-boosting overrides or reject structural changes."""

    supplied = dict(overrides or {})
    unsupported = sorted(set(supplied).difference(ADAPTATION_OVERRIDE_KEYS))
    if unsupported:
        raise ValueError(
            "Stage 2 adaptation overrides may change only new-tree hyperparameters; "
            f"unsupported keys: {unsupported}"
        )
    for name in ("learning_rate", "huber_slope"):
        if name in supplied:
            value = float(supplied[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Stage 2 {name} must be finite and positive.")
    for name in ("min_child_weight", "reg_alpha", "reg_lambda", "gamma"):
        if name in supplied:
            value = float(supplied[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Stage 2 {name} must be finite and non-negative."
                )
    for name in ("subsample", "colsample_bytree"):
        if name in supplied:
            value = float(supplied[name])
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"Stage 2 {name} must be finite and in (0, 1].")
    if "max_depth" in supplied:
        depth = supplied["max_depth"]
        if isinstance(depth, bool) or int(depth) != depth or int(depth) <= 0:
            raise ValueError("Stage 2 max_depth must be a positive integer.")
    return supplied


def adapt_from_frozen_base(
    base: FittedDirectModel,
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    additional_trees: int,
    seed: int,
    n_jobs: int = 1,
    parameter_overrides: Mapping[str, Any] | None = None,
    predictor_guard: Callable[[Sequence[str]], Any] | None = None,
) -> FittedDirectModel:
    """Continue boosting from an immutable Stage 1 model.

    The fitted Stage 1 preprocessor is transformed only and is never refit. A
    copied booster initializes the new estimator, and the parent booster bytes
    are checked before and after fitting so adaptation cannot silently mutate
    the Stage 1 model in memory.
    """

    if additional_trees <= 0:
        raise ValueError("Stage 2 additional_trees must be positive.")
    if tuple(frame.columns) != tuple(base.predictors):
        raise ValueError(
            "Stage 2 predictors must exactly match the saved Stage 1 raw feature "
            "order before adaptation."
        )
    if predictor_guard is not None:
        predictor_guard(base.predictors)
    y = np.asarray(target, dtype=float)
    if len(frame) != len(y):
        raise ValueError("Stage 2 predictor rows and target values have different lengths.")
    if not np.isfinite(y).all():
        raise ValueError("Stage 2 sensor-derived UTCI targets must all be finite.")

    transformed = base.preprocessor.transform(
        frame.loc[:, list(base.predictors)]
    )
    parent_booster = base.model.get_booster()
    parent_before = bytes(parent_booster.save_raw(raw_format="ubj"))
    parent_rounds = int(parent_booster.num_boosted_rounds())

    params = {
        key: value
        for key, value in base.model.get_params(deep=False).items()
        if value is not None
    }
    params.update(validate_adaptation_overrides(parameter_overrides))
    params["objective"] = base.objective
    params["n_estimators"] = int(additional_trees)
    adapted = make_regressor(params, seed=seed, n_jobs=n_jobs)
    adapted.fit(
        transformed,
        y,
        xgb_model=parent_booster.copy(),
        verbose=False,
    )

    parent_after = bytes(parent_booster.save_raw(raw_format="ubj"))
    if parent_after != parent_before:
        raise RuntimeError("Stage 1 booster changed during Stage 2 adaptation.")
    total_rounds = int(adapted.get_booster().num_boosted_rounds())
    expected_rounds = parent_rounds + int(additional_trees)
    if total_rounds != expected_rounds:
        raise RuntimeError(
            "Unexpected continued-boosting tree count: "
            f"expected {expected_rounds}, observed {total_rounds}."
        )
    return FittedDirectModel(
        model=adapted,
        preprocessor=base.preprocessor,
        predictors=base.predictors,
        objective=base.objective,
        tree_count=total_rounds,
        model_variant="stage2_local_adapted",
    )
