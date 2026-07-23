"""Data-independent metrics for continuous UTCI and fixed heat categories.

All comparison functions use paired finite observations.  Cluster bootstrap
functions resample complete caller-supplied spatial-day or site-day blocks,
never individual rows.  Positive improvement means the candidate model has a
smaller error than the baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score

type HeatCategory = Literal[
    "no_heat_stress",
    "moderate",
    "strong",
    "very_strong",
    "extreme",
]

HEAT_CATEGORY_LABELS: tuple[HeatCategory, ...] = (
    "no_heat_stress",
    "moderate",
    "strong",
    "very_strong",
    "extreme",
)
HEAT_THRESHOLDS_C = (26.0, 32.0, 38.0, 46.0)
RECALL_THRESHOLDS_C = (32.0, 38.0, 46.0)


def utci_heat_category(value_c: float) -> HeatCategory:
    """Classify one continuous UTCI value using the fixed heat-only thresholds."""

    value = float(value_c)
    if not np.isfinite(value):
        raise ValueError("UTCI category requires a finite continuous value")
    if value < 26.0:
        return "no_heat_stress"
    if value < 32.0:
        return "moderate"
    if value < 38.0:
        return "strong"
    if value < 46.0:
        return "very_strong"
    return "extreme"


def derive_utci_categories(values_c: ArrayLike) -> NDArray[np.object_]:
    """Vectorise :func:`utci_heat_category`; missing values remain ``None``."""

    values = np.asarray(values_c, dtype=float)
    categories = np.full(values.shape, None, dtype=object)
    finite = np.isfinite(values)
    categories[finite & (values < 26.0)] = "no_heat_stress"
    categories[finite & (values >= 26.0) & (values < 32.0)] = "moderate"
    categories[finite & (values >= 32.0) & (values < 38.0)] = "strong"
    categories[finite & (values >= 38.0) & (values < 46.0)] = "very_strong"
    categories[finite & (values >= 46.0)] = "extreme"
    return categories


# Natural shorter alias used by CLI/report code.
utci_categories = derive_utci_categories


def _paired_finite(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    finite = np.isfinite(truth) & np.isfinite(prediction)
    return truth[finite], prediction[finite], int(np.count_nonzero(~finite))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float | None:
    """Return paired MAE, or ``None`` when no finite pair is available."""

    truth, prediction, _ = _paired_finite(y_true, y_pred)
    if not truth.size:
        return None
    return float(np.mean(np.abs(prediction - truth)))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float | None:
    """Return paired RMSE, or ``None`` when no finite pair is available."""

    truth, prediction, _ = _paired_finite(y_true, y_pred)
    if not truth.size:
        return None
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def r_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float | None:
    """Return R², explicitly unavailable for fewer than two or constant targets."""

    truth, prediction, _ = _paired_finite(y_true, y_pred)
    if truth.size < 2:
        return None
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    if denominator == 0.0:
        return None
    numerator = float(np.sum(np.square(truth - prediction)))
    return float(1.0 - numerator / denominator)


def spearman_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float | None:
    """Return Spearman's rho, or ``None`` when it is not defined."""

    truth, prediction, _ = _paired_finite(y_true, y_pred)
    if truth.size < 2 or np.unique(truth).size < 2 or np.unique(prediction).size < 2:
        return None
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else None


def mean_bias(y_true: ArrayLike, y_pred: ArrayLike) -> float | None:
    """Return mean signed error ``prediction - observation``."""

    truth, prediction, _ = _paired_finite(y_true, y_pred)
    if not truth.size:
        return None
    return float(np.mean(prediction - truth))


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float | int | None]:
    """Return the preregistered continuous metrics and paired support count."""

    truth, prediction, omitted = _paired_finite(y_true, y_pred)
    if not truth.size:
        return {
            "n": 0,
            "n_omitted_nonfinite": omitted,
            "mae": None,
            "rmse": None,
            "r2": None,
            "spearman": None,
            "mean_bias": None,
        }
    return {
        "n": int(truth.size),
        "n_omitted_nonfinite": omitted,
        "mae": mean_absolute_error(truth, prediction),
        "rmse": root_mean_squared_error(truth, prediction),
        "r2": r_squared(truth, prediction),
        "spearman": spearman_correlation(truth, prediction),
        "mean_bias": mean_bias(truth, prediction),
    }


def category_accuracy(y_true_c: ArrayLike, y_pred_c: ArrayLike) -> float | None:
    """Return accuracy after deriving both categories from continuous UTCI."""

    truth, prediction, _ = _paired_finite(y_true_c, y_pred_c)
    if not truth.size:
        return None
    return float(np.mean(derive_utci_categories(truth) == derive_utci_categories(prediction)))


def category_macro_f1(y_true_c: ArrayLike, y_pred_c: ArrayLike) -> float | None:
    """Return macro F1 over all five fixed labels, including unsupported labels."""

    truth, prediction, _ = _paired_finite(y_true_c, y_pred_c)
    if not truth.size:
        return None
    return float(
        f1_score(
            derive_utci_categories(truth),
            derive_utci_categories(prediction),
            labels=list(HEAT_CATEGORY_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def category_confusion_matrix(y_true_c: ArrayLike, y_pred_c: ArrayLike) -> list[list[int]]:
    """Return a 5-by-5 confusion matrix in :data:`HEAT_CATEGORY_LABELS` order."""

    truth, prediction, _ = _paired_finite(y_true_c, y_pred_c)
    if not truth.size:
        return [[0 for _ in HEAT_CATEGORY_LABELS] for _ in HEAT_CATEGORY_LABELS]
    matrix = sklearn_confusion_matrix(
        derive_utci_categories(truth),
        derive_utci_categories(prediction),
        labels=list(HEAT_CATEGORY_LABELS),
    )
    return matrix.astype(int).tolist()


def category_metrics(y_true_c: ArrayLike, y_pred_c: ArrayLike) -> dict[str, Any]:
    """Return fixed-label category accuracy, macro F1, support, and confusion."""

    truth, prediction, omitted = _paired_finite(y_true_c, y_pred_c)
    truth_categories = derive_utci_categories(truth)
    support = {
        label: int(np.count_nonzero(truth_categories == label))
        for label in HEAT_CATEGORY_LABELS
    }
    return {
        "n": int(truth.size),
        "n_omitted_nonfinite": omitted,
        "labels": list(HEAT_CATEGORY_LABELS),
        "accuracy": category_accuracy(truth, prediction),
        "macro_f1": category_macro_f1(truth, prediction),
        "support": support,
        "confusion_matrix": category_confusion_matrix(truth, prediction),
    }


@dataclass(frozen=True)
class ThresholdRecall:
    """Recall plus an explicit availability state for a UTCI threshold."""

    threshold_c: float
    value: float | None
    available: bool
    positive_support: int
    n: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recall_at_utci_threshold(
    y_true_c: ArrayLike,
    y_pred_c: ArrayLike,
    threshold_c: float,
) -> ThresholdRecall:
    """Return recall for ``UTCI >= threshold_c`` with explicit no-positive state."""

    truth, prediction, _ = _paired_finite(y_true_c, y_pred_c)
    positives = truth >= float(threshold_c)
    support = int(np.count_nonzero(positives))
    if support == 0:
        return ThresholdRecall(
            threshold_c=float(threshold_c),
            value=None,
            available=False,
            positive_support=0,
            n=int(truth.size),
            reason="unavailable: no positive observed cases in this subset",
        )
    true_positives = int(np.count_nonzero(prediction[positives] >= float(threshold_c)))
    return ThresholdRecall(
        threshold_c=float(threshold_c),
        value=float(true_positives / support),
        available=True,
        positive_support=support,
        n=int(truth.size),
    )


def threshold_recalls(
    y_true_c: ArrayLike,
    y_pred_c: ArrayLike,
    *,
    thresholds_c: Sequence[float] = RECALL_THRESHOLDS_C,
) -> dict[str, dict[str, Any]]:
    """Return recalls for all configured UTCI heat thresholds."""

    return {
        f"utci_ge_{float(threshold):g}_c": recall_at_utci_threshold(
            y_true_c, y_pred_c, float(threshold)
        ).to_dict()
        for threshold in thresholds_c
    }


def high_utci_mae(
    y_true_c: ArrayLike,
    y_pred_c: ArrayLike,
    *,
    threshold_c: float = 32.0,
) -> dict[str, float | int | bool | None | str]:
    """Return secondary high-UTCI MAE without changing the primary objective."""

    truth, prediction, _ = _paired_finite(y_true_c, y_pred_c)
    selected = truth >= float(threshold_c)
    support = int(np.count_nonzero(selected))
    if support == 0:
        return {
            "value": None,
            "available": False,
            "support": 0,
            "threshold_c": float(threshold_c),
            "reason": "unavailable: no observations meet the high-UTCI threshold",
        }
    return {
        "value": float(np.mean(np.abs(prediction[selected] - truth[selected]))),
        "available": True,
        "support": support,
        "threshold_c": float(threshold_c),
        "reason": None,
    }


def full_metric_report(y_true_c: ArrayLike, y_pred_c: ArrayLike) -> dict[str, Any]:
    """Combine continuous, fixed-category, and threshold-recall metrics."""

    return {
        "regression": regression_metrics(y_true_c, y_pred_c),
        "categories": category_metrics(y_true_c, y_pred_c),
        "threshold_recall": threshold_recalls(y_true_c, y_pred_c),
        "high_utci_mae": high_utci_mae(y_true_c, y_pred_c),
    }


def _normalise_sun_shade(
    sun_shade: Sequence[Any] | None,
    shade_fraction: ArrayLike | None,
    n: int,
    threshold: float,
) -> NDArray[np.object_] | None:
    if sun_shade is not None:
        values = np.asarray(list(sun_shade), dtype=object).reshape(-1)
        if values.size != n:
            raise ValueError("sun_shade length must match y_true")
        return values
    if shade_fraction is not None:
        fractions = np.asarray(shade_fraction, dtype=float).reshape(-1)
        if fractions.size != n:
            raise ValueError("shade_fraction length must match y_true")
        values = np.full(n, "missing", dtype=object)
        values[np.isfinite(fractions) & (fractions < threshold)] = "sun"
        values[np.isfinite(fractions) & (fractions >= threshold)] = "shade"
        return values
    return None


def _coast_groups(
    coast_distance_km: ArrayLike,
    cut_points_km: Sequence[float],
) -> NDArray[np.object_]:
    distances = np.asarray(coast_distance_km, dtype=float).reshape(-1)
    cuts = [float(value) for value in cut_points_km]
    if len(cuts) < 2 or cuts[0] != 0.0 or any(b <= a for a, b in pairwise(cuts)):
        raise ValueError("coast cut points must increase strictly from 0")
    labels = [f"{cuts[i]:g}_to_{cuts[i + 1]:g}_km" for i in range(len(cuts) - 1)]
    labels.append(f"ge_{cuts[-1]:g}_km")
    groups = np.full(distances.shape, "missing", dtype=object)
    nonnegative = np.isfinite(distances) & (distances >= 0.0)
    groups[nonnegative] = np.asarray(
        pd.cut(
            distances[nonnegative],
            bins=[*cuts, np.inf],
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype(str),
        dtype=object,
    )
    groups[np.isfinite(distances) & (distances < 0.0)] = "invalid_negative"
    return groups


def time_of_day_groups(hour_of_day: ArrayLike) -> NDArray[np.object_]:
    """Map documented local hours to fixed reporting periods."""

    hours = np.asarray(hour_of_day, dtype=float).reshape(-1)
    groups = np.full(hours.shape, "missing", dtype=object)
    valid = np.isfinite(hours) & (hours >= 0.0) & (hours < 24.0)
    groups[valid & (hours >= 6.0) & (hours < 12.0)] = "morning_06_12"
    groups[valid & (hours >= 12.0) & (hours < 18.0)] = "afternoon_12_18"
    groups[valid & (hours >= 18.0) & (hours < 22.0)] = "evening_18_22"
    groups[valid & ((hours >= 22.0) | (hours < 6.0))] = "night_22_06"
    groups[np.isfinite(hours) & ~valid] = "invalid_hour"
    return groups


def evaluate_prespecified_subgroups(
    y_true_c: ArrayLike,
    y_pred_c: ArrayLike,
    *,
    sun_shade: Sequence[Any] | None = None,
    shade_fraction: ArrayLike | None = None,
    shade_threshold: float = 0.5,
    land_cover: Sequence[Any] | None = None,
    coast_distance: Sequence[Any] | None = None,
    coast_distance_km: ArrayLike | None = None,
    coast_cut_points_km: Sequence[float] = (0.0, 5.0, 20.0),
    time_of_day: Sequence[Any] | None = None,
    hour_of_day: ArrayLike | None = None,
) -> pd.DataFrame:
    """Evaluate heat, sun/shade, land-cover, coast, and time-of-day groups."""

    truth = np.asarray(y_true_c, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred_c, dtype=float).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true_c and y_pred_c must have identical shapes")
    n = truth.size
    groupings: dict[str, NDArray[np.object_]] = {
        "observed_heat_category": derive_utci_categories(truth),
    }
    sun_values = _normalise_sun_shade(sun_shade, shade_fraction, n, shade_threshold)
    if sun_values is not None:
        groupings["sun_vs_shade"] = sun_values
    if land_cover is not None:
        values = np.asarray(list(land_cover), dtype=object).reshape(-1)
        if values.size != n:
            raise ValueError("land_cover length must match y_true")
        groupings["land_cover"] = values
    if coast_distance is not None:
        coast_values = np.asarray(list(coast_distance), dtype=object).reshape(-1)
        if coast_values.size != n:
            raise ValueError("coast_distance length must match y_true")
        groupings["coast_distance"] = coast_values
    elif coast_distance_km is not None:
        coast = np.asarray(coast_distance_km, dtype=float).reshape(-1)
        if coast.size != n:
            raise ValueError("coast_distance_km length must match y_true")
        groupings["coast_distance"] = _coast_groups(coast, coast_cut_points_km)
    if time_of_day is not None:
        time_values = np.asarray(list(time_of_day), dtype=object).reshape(-1)
        if time_values.size != n:
            raise ValueError("time_of_day length must match y_true")
        groupings["time_of_day"] = time_values
    elif hour_of_day is not None:
        hours = np.asarray(hour_of_day, dtype=float).reshape(-1)
        if hours.size != n:
            raise ValueError("hour_of_day length must match y_true")
        groupings["time_of_day"] = time_of_day_groups(hours)

    rows: list[dict[str, Any]] = []
    for grouping_name, values in groupings.items():
        display_values = np.asarray(
            ["missing" if value is None or pd.isna(value) else str(value) for value in values],
            dtype=object,
        )
        for group_name in pd.unique(display_values):
            selected = display_values == group_name
            report = regression_metrics(truth[selected], prediction[selected])
            row: dict[str, Any] = {
                "subgroup_type": grouping_name,
                "subgroup": str(group_name),
                **report,
            }
            for threshold in RECALL_THRESHOLDS_C:
                recall = recall_at_utci_threshold(
                    truth[selected], prediction[selected], threshold
                )
                key = f"recall_ge_{threshold:g}_c"
                row[key] = recall.value
                row[f"{key}_available"] = recall.available
                row[f"{key}_positive_support"] = recall.positive_support
            rows.append(row)
    return pd.DataFrame(rows)


# Shorter public alias.
evaluate_subgroups = evaluate_prespecified_subgroups


type BootstrapWeighting = Literal["observation_weighted", "block_balanced"]
type BootstrapMetric = Literal["mae", "rmse"]


@dataclass(frozen=True)
class PairedBlockBootstrapCI:
    """Paired model-minus-baseline improvement interval from block resampling."""

    metric: str
    weighting: str
    point_improvement: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int
    n_observations: int
    n_blocks: int
    random_seed: int
    interpretation: str = "positive values favor the candidate model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _error_loss(
    truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    metric: BootstrapMetric,
) -> float:
    errors = prediction - truth
    if metric == "mae":
        return float(np.mean(np.abs(errors)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(errors))))
    raise ValueError("metric must be 'mae' or 'rmse'")


def paired_block_bootstrap_improvement(
    y_true: ArrayLike,
    model_prediction: ArrayLike,
    baseline_prediction: ArrayLike,
    block_ids: Sequence[Any],
    *,
    metric: BootstrapMetric = "mae",
    weighting: BootstrapWeighting = "observation_weighted",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 2026,
) -> PairedBlockBootstrapCI:
    """Bootstrap paired improvement by resampling complete dependence blocks.

    ``block_ids`` must identify prespecified spatial-day or site-day blocks.
    ``observation_weighted`` computes the metric after concatenating sampled
    whole blocks; ``block_balanced`` gives each sampled block's metric equal
    weight.
    """

    truth = np.asarray(y_true, dtype=float).reshape(-1)
    model = np.asarray(model_prediction, dtype=float).reshape(-1)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)
    blocks = pd.Series(list(block_ids), dtype="object").to_numpy()
    if not (truth.shape == model.shape == baseline.shape == blocks.shape):
        raise ValueError("truth, predictions, and block_ids must have identical lengths")
    if weighting not in {"observation_weighted", "block_balanced"}:
        raise ValueError("weighting must be 'observation_weighted' or 'block_balanced'")
    if metric not in {"mae", "rmse"}:
        raise ValueError("metric must be 'mae' or 'rmse'")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    finite = (
        np.isfinite(truth)
        & np.isfinite(model)
        & np.isfinite(baseline)
        & pd.notna(blocks)
    )
    truth, model, baseline, blocks = (
        truth[finite],
        model[finite],
        baseline[finite],
        blocks[finite],
    )
    block_codes, unique_blocks = pd.factorize(blocks, sort=False)
    if len(unique_blocks) < 2:
        raise ValueError("paired block bootstrap requires at least two non-empty blocks")
    row_indices = [
        np.flatnonzero(block_codes == block_index)
        for block_index in range(len(unique_blocks))
    ]

    def improvement_for_selection(selection: NDArray[np.int64]) -> float:
        if weighting == "observation_weighted":
            rows = np.concatenate([row_indices[int(index)] for index in selection])
            return _error_loss(truth[rows], baseline[rows], metric) - _error_loss(
                truth[rows], model[rows], metric
            )
        block_improvements = [
            _error_loss(truth[row_indices[int(index)]], baseline[row_indices[int(index)]], metric)
            - _error_loss(truth[row_indices[int(index)]], model[row_indices[int(index)]], metric)
            for index in selection
        ]
        return float(np.mean(block_improvements))

    all_blocks = np.arange(len(unique_blocks), dtype=np.int64)
    point = improvement_for_selection(all_blocks)
    rng = np.random.default_rng(random_seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        selection = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        samples[bootstrap_index] = improvement_for_selection(selection)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return PairedBlockBootstrapCI(
        metric=metric,
        weighting=weighting,
        point_improvement=float(point),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        n_bootstrap=int(n_bootstrap),
        n_observations=int(truth.size),
        n_blocks=len(unique_blocks),
        random_seed=int(random_seed),
    )


def compose_spatial_day_blocks(
    spatial_or_site_id: Sequence[Any],
    dates: Sequence[Any],
) -> NDArray[np.object_]:
    """Create unambiguous tuple IDs for spatial-day/site-day resampling."""

    ids = list(spatial_or_site_id)
    day_values = list(dates)
    if len(ids) != len(day_values):
        raise ValueError("spatial/site IDs and dates must have identical lengths")
    result = np.empty(len(ids), dtype=object)
    for index, (location, day) in enumerate(zip(ids, day_values, strict=True)):
        if pd.isna(location) or pd.isna(day):
            raise ValueError("resampling block identifiers cannot be missing")
        result[index] = (str(location), str(day))
    return result


__all__ = [
    "HEAT_CATEGORY_LABELS",
    "HEAT_THRESHOLDS_C",
    "RECALL_THRESHOLDS_C",
    "PairedBlockBootstrapCI",
    "ThresholdRecall",
    "category_accuracy",
    "category_confusion_matrix",
    "category_macro_f1",
    "category_metrics",
    "compose_spatial_day_blocks",
    "derive_utci_categories",
    "evaluate_prespecified_subgroups",
    "evaluate_subgroups",
    "full_metric_report",
    "high_utci_mae",
    "mean_absolute_error",
    "mean_bias",
    "paired_block_bootstrap_improvement",
    "r_squared",
    "recall_at_utci_threshold",
    "regression_metrics",
    "root_mean_squared_error",
    "spearman_correlation",
    "threshold_recalls",
    "time_of_day_groups",
    "utci_categories",
    "utci_heat_category",
]
