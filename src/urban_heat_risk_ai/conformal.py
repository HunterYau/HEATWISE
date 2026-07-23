"""Split-conformal intervals calibrated on the untouched calibration partition.

The finite-sample order statistic is implemented literally as
``k = ceil((n + 1) * (1 - alpha))`` with one-based indexing.  When ``k > n``
the only honest finite-sample interval is unbounded; this module returns that
interval and emits a clear warning instead of substituting an interpolated
quantile.

Coverage guarantees require exchangeability of calibration and future errors.
Spatiotemporal distribution shift can violate that assumption, so subgroup
coverage is a required diagnostic rather than a causal or formal subgroup
guarantee.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

EXCHANGEABILITY_NOTE = (
    "The split-conformal marginal coverage guarantee depends on exchangeability "
    "between calibration and future residuals; blocked environmental data may shift."
)


@dataclass(frozen=True)
class ConformalOrderStatistic:
    """Selected absolute-residual order statistic and its availability state."""

    alpha: float
    n_calibration: int
    k: int
    residual_quantile_c: float
    bounded: bool
    minimum_calibration_rows: int | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def conformal_order_statistic(
    absolute_residuals_c: ArrayLike,
    *,
    alpha: float = 0.10,
    minimum_calibration_rows: int | None = None,
    warn: bool = True,
) -> ConformalOrderStatistic:
    """Select the exact split-conformal absolute-residual order statistic."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if minimum_calibration_rows is not None and minimum_calibration_rows <= 0:
        raise ValueError("minimum_calibration_rows must be positive when configured")
    residuals = np.asarray(absolute_residuals_c, dtype=float).reshape(-1)
    if np.any(~np.isfinite(residuals)):
        raise ValueError("calibration residuals must all be finite")
    if np.any(residuals < 0.0):
        raise ValueError("absolute calibration residuals cannot be negative")
    n = int(residuals.size)
    k = ceil((n + 1) * (1.0 - float(alpha)))
    below_configured_minimum = (
        minimum_calibration_rows is not None and n < minimum_calibration_rows
    )
    if k > n or below_configured_minimum:
        if k > n:
            reason = f"k={k} exceeds n={n}"
        else:
            reason = f"n={n} is below the configured minimum {minimum_calibration_rows}"
        message = (
            f"calibration set is too small for alpha={alpha:g}: {reason}; returning an "
            "unbounded split-conformal interval"
        )
        if warn:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return ConformalOrderStatistic(
            alpha=float(alpha),
            n_calibration=n,
            k=k,
            residual_quantile_c=float("inf"),
            bounded=False,
            minimum_calibration_rows=minimum_calibration_rows,
            warning=message,
        )
    selected = float(np.sort(residuals)[k - 1])
    return ConformalOrderStatistic(
        alpha=float(alpha),
        n_calibration=n,
        k=k,
        residual_quantile_c=selected,
        bounded=True,
        minimum_calibration_rows=minimum_calibration_rows,
    )


@dataclass(frozen=True)
class SplitConformalCalibrator:
    """A frozen symmetric split-conformal calibration result."""

    order_statistic: ConformalOrderStatistic
    partition_role: str = "calibration"

    @classmethod
    def fit(
        cls,
        y_calibration_c: ArrayLike,
        prediction_calibration_c: ArrayLike,
        *,
        partition_role: str,
        alpha: float = 0.10,
        minimum_calibration_rows: int = 20,
    ) -> SplitConformalCalibrator:
        """Fit only from paired residuals in the untouched calibration partition."""

        if partition_role != "calibration":
            raise ValueError(
                "split-conformal calibration requires partition_role='calibration'"
            )
        truth = np.asarray(y_calibration_c, dtype=float).reshape(-1)
        prediction = np.asarray(prediction_calibration_c, dtype=float).reshape(-1)
        if truth.shape != prediction.shape:
            raise ValueError("calibration observations and predictions must align exactly")
        if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(prediction)):
            raise ValueError(
                "calibration observations and predictions must all be finite; "
                "do not silently change the untouched calibration sample"
            )
        order = conformal_order_statistic(
            np.abs(truth - prediction),
            alpha=alpha,
            minimum_calibration_rows=minimum_calibration_rows,
        )
        return cls(order_statistic=order)

    @property
    def alpha(self) -> float:
        return self.order_statistic.alpha

    @property
    def residual_quantile_c(self) -> float:
        return self.order_statistic.residual_quantile_c

    def predict_interval(
        self,
        point_prediction_c: ArrayLike,
    ) -> tuple[float, float] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply the symmetric frozen interval to new point predictions."""

        prediction = np.asarray(point_prediction_c, dtype=float)
        if np.any(~np.isfinite(prediction)):
            raise ValueError("point predictions must be finite before interval construction")
        radius = self.residual_quantile_c
        lower = prediction - radius
        upper = prediction + radius
        if prediction.ndim == 0:
            return float(lower), float(upper)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_role": self.partition_role,
            "order_statistic": self.order_statistic.to_dict(),
            "exchangeability_note": EXCHANGEABILITY_NOTE,
        }


def fit_split_conformal(
    y_calibration_c: ArrayLike,
    prediction_calibration_c: ArrayLike,
    *,
    partition_role: str,
    alpha: float = 0.10,
    minimum_calibration_rows: int = 20,
) -> SplitConformalCalibrator:
    """Functional wrapper for :meth:`SplitConformalCalibrator.fit`."""

    return SplitConformalCalibrator.fit(
        y_calibration_c,
        prediction_calibration_c,
        partition_role=partition_role,
        alpha=alpha,
        minimum_calibration_rows=minimum_calibration_rows,
    )


def interval_diagnostics(
    y_true_c: ArrayLike,
    lower_c: ArrayLike,
    upper_c: ArrayLike,
) -> dict[str, float | int | None]:
    """Return empirical coverage and interval width for aligned observations."""

    truth = np.asarray(y_true_c, dtype=float).reshape(-1)
    lower = np.asarray(lower_c, dtype=float).reshape(-1)
    upper = np.asarray(upper_c, dtype=float).reshape(-1)
    if not (truth.shape == lower.shape == upper.shape):
        raise ValueError("observations and interval bounds must align exactly")
    usable = np.isfinite(truth) & ~np.isnan(lower) & ~np.isnan(upper) & (lower <= upper)
    omitted = int(np.count_nonzero(~usable))
    if not np.any(usable):
        return {
            "n": 0,
            "n_omitted": omitted,
            "empirical_coverage": None,
            "mean_interval_width_c": None,
            "median_interval_width_c": None,
        }
    covered = (truth[usable] >= lower[usable]) & (truth[usable] <= upper[usable])
    widths = upper[usable] - lower[usable]
    return {
        "n": int(np.count_nonzero(usable)),
        "n_omitted": omitted,
        "empirical_coverage": float(np.mean(covered)),
        "mean_interval_width_c": float(np.mean(widths)),
        "median_interval_width_c": float(np.median(widths)),
    }


def subgroup_interval_diagnostics(
    y_true_c: ArrayLike,
    lower_c: ArrayLike,
    upper_c: ArrayLike,
    subgroup_values: Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    """Report coverage/width within each prespecified subgroup dimension."""

    truth = np.asarray(y_true_c, dtype=float).reshape(-1)
    lower = np.asarray(lower_c, dtype=float).reshape(-1)
    upper = np.asarray(upper_c, dtype=float).reshape(-1)
    if not (truth.shape == lower.shape == upper.shape):
        raise ValueError("observations and interval bounds must align exactly")
    rows: list[dict[str, Any]] = []
    for dimension, raw_values in subgroup_values.items():
        values = np.asarray(list(raw_values), dtype=object).reshape(-1)
        if values.size != truth.size:
            raise ValueError(f"subgroup '{dimension}' does not align with intervals")
        labels = np.asarray(
            ["missing" if value is None or pd.isna(value) else str(value) for value in values],
            dtype=object,
        )
        for label in pd.unique(labels):
            selected = labels == label
            rows.append(
                {
                    "subgroup_type": str(dimension),
                    "subgroup": str(label),
                    **interval_diagnostics(
                        truth[selected], lower[selected], upper[selected]
                    ),
                }
            )
    return pd.DataFrame(rows)


def conformal_coverage_report(
    y_true_c: ArrayLike,
    point_prediction_c: ArrayLike,
    calibrator: SplitConformalCalibrator,
    *,
    subgroup_values: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Apply a calibrator and report overall and prespecified subgroup behavior."""

    interval = calibrator.predict_interval(point_prediction_c)
    lower, upper = interval
    report: dict[str, Any] = {
        "nominal_coverage": 1.0 - calibrator.alpha,
        "overall": interval_diagnostics(y_true_c, lower, upper),
        "exchangeability_note": EXCHANGEABILITY_NOTE,
        "calibration": calibrator.to_dict(),
    }
    if subgroup_values:
        report["subgroups"] = subgroup_interval_diagnostics(
            y_true_c, lower, upper, subgroup_values
        ).to_dict(orient="records")
    else:
        report["subgroups"] = []
    return report


__all__ = [
    "EXCHANGEABILITY_NOTE",
    "ConformalOrderStatistic",
    "SplitConformalCalibrator",
    "conformal_coverage_report",
    "conformal_order_statistic",
    "fit_split_conformal",
    "interval_diagnostics",
    "subgroup_interval_diagnostics",
]
