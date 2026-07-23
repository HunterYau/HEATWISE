"""Predictive-association explanations and extrapolation diagnostics.

These routines describe patterns learned by a predictive model. They do not
identify causal effects of canopy, paving, radiation, wind, coast distance, or
any other feature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

ASSOCIATION_DISCLAIMER = (
    "Model explanations are predictive associations conditional on the training "
    "data and feature set; they are not evidence that changing a feature causes a "
    "change in UTCI."
)


def _prepare_output(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _feature_names(preprocessor: Any, width: int) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(preprocessor, "feature_names_out_"):
        return [str(name) for name in preprocessor.feature_names_out_]
    return [f"feature_{index}" for index in range(width)]


def save_shap_global_plots(
    fitted: Any,
    frame: pd.DataFrame,
    *,
    bar_path: str | Path,
    beeswarm_path: str | Path,
    max_display: int = 25,
    max_rows: int = 2_000,
    seed: int = 42,
) -> dict[str, str]:
    """Save SHAP global bar and beeswarm plots for a fitted model bundle."""

    import shap

    predictors = list(fitted.predictors)
    sample = frame.loc[:, predictors]
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed)
    transformed = fitted.preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    names = _feature_names(fitted.preprocessor, transformed.shape[1])
    explainer = shap.TreeExplainer(fitted.model)
    values = explainer(transformed)
    values.feature_names = names

    bar_destination = _prepare_output(bar_path)
    shap.plots.bar(values, max_display=max_display, show=False)
    plt.title("Global predictive associations (SHAP magnitude)")
    plt.figtext(0.01, 0.01, ASSOCIATION_DISCLAIMER, wrap=True, fontsize=7)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(bar_destination, dpi=180, bbox_inches="tight")
    plt.close()

    beeswarm_destination = _prepare_output(beeswarm_path)
    shap.plots.beeswarm(values, max_display=max_display, show=False)
    plt.title("Global predictive associations (SHAP direction)")
    plt.figtext(0.01, 0.01, ASSOCIATION_DISCLAIMER, wrap=True, fontsize=7)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(beeswarm_destination, dpi=180, bbox_inches="tight")
    plt.close()
    return {
        "bar": str(bar_destination),
        "beeswarm": str(beeswarm_destination),
        "interpretation": ASSOCIATION_DISCLAIMER,
    }


def save_shap_local_waterfall(
    fitted: Any,
    row: pd.DataFrame,
    *,
    output_path: str | Path,
    max_display: int = 20,
) -> str:
    """Save a local SHAP waterfall for exactly one observation."""

    import shap

    if len(row) != 1:
        raise ValueError("A local waterfall requires exactly one observation.")
    transformed = fitted.preprocessor.transform(row.loc[:, list(fitted.predictors)])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    names = _feature_names(fitted.preprocessor, transformed.shape[1])
    values = shap.TreeExplainer(fitted.model)(transformed)
    values.feature_names = names
    destination = _prepare_output(output_path)
    shap.plots.waterfall(values[0], max_display=max_display, show=False)
    plt.title("Local predictive association explanation")
    plt.figtext(0.01, 0.01, ASSOCIATION_DISCLAIMER, wrap=True, fontsize=7)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()
    return str(destination)


def partial_dependence_curve(
    predict: Callable[[pd.DataFrame], np.ndarray],
    reference: pd.DataFrame,
    feature: str,
    *,
    recompute_derived: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    grid_size: int = 30,
) -> pd.DataFrame:
    """Calculate one-way empirical partial dependence on the reference rows."""

    if feature not in reference.columns:
        raise ValueError(f"Partial-dependence feature is missing: {feature}")
    numeric = pd.to_numeric(reference[feature], errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        raise ValueError(f"Partial-dependence feature has no finite values: {feature}")
    low, high = finite.quantile([lower_quantile, upper_quantile]).to_numpy(dtype=float)
    grid = np.linspace(low, high, int(grid_size))
    averages: list[float] = []
    for value in grid:
        changed = reference.copy()
        changed.loc[:, feature] = value
        if recompute_derived is not None:
            changed = recompute_derived(changed)
        averages.append(float(np.nanmean(predict(changed))))
    return pd.DataFrame({"feature": feature, "value": grid, "mean_prediction": averages})


def save_partial_dependence_plots(
    fitted: Any,
    reference: pd.DataFrame,
    *,
    features: Sequence[str],
    output_path: str | Path,
    recompute_derived: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    grid_size: int = 30,
    require_all: bool = True,
) -> str:
    """Save manual one-way PDPs for prespecified interpretable features."""

    missing = [feature for feature in features if feature not in reference.columns]
    if missing and require_all:
        raise ValueError(
            "Configured partial-dependence features are missing from the reference "
            f"table: {missing}"
        )
    usable = [feature for feature in features if feature in reference.columns]
    if not usable:
        raise ValueError("None of the configured partial-dependence features exist.")
    columns = min(3, len(usable))
    rows = int(np.ceil(len(usable) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.8 * rows))
    axis_list = np.asarray(axes, dtype=object).reshape(-1)
    for axis, feature in zip(axis_list, usable, strict=False):
        curve = partial_dependence_curve(
            fitted.predict,
            reference,
            feature,
            recompute_derived=recompute_derived,
            grid_size=grid_size,
        )
        axis.plot(curve["value"], curve["mean_prediction"], color="#b33b2e")
        axis.set_title(feature)
        axis.set_xlabel("Feature value")
        axis.set_ylabel("Mean predicted UTCI (°C)")
        axis.grid(alpha=0.25)
    for axis in axis_list[len(usable) :]:
        axis.set_visible(False)
    figure.suptitle("Partial dependence: predictive associations")
    figure.text(0.01, 0.01, ASSOCIATION_DISCLAIMER, wrap=True, fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    destination = _prepare_output(output_path)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(destination)


@dataclass(frozen=True)
class FeatureRangeProfile:
    """Training-only univariate ranges for visible extrapolation warnings."""

    ranges: Mapping[str, tuple[float, float]]

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, predictors: Sequence[str]
    ) -> FeatureRangeProfile:
        ranges: dict[str, tuple[float, float]] = {}
        for feature in predictors:
            numeric = pd.to_numeric(frame[feature], errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            if not finite.empty:
                ranges[feature] = (float(finite.min()), float(finite.max()))
        return cls(ranges=ranges)

    def check(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return one row per feature with out-of-training-range counts."""

        rows: list[dict[str, Any]] = []
        for feature, (minimum, maximum) in self.ranges.items():
            if feature not in frame.columns:
                rows.append(
                    {
                        "feature": feature,
                        "training_min": minimum,
                        "training_max": maximum,
                        "below_count": None,
                        "above_count": None,
                        "missing_column": True,
                    }
                )
                continue
            numeric = pd.to_numeric(frame[feature], errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "training_min": minimum,
                    "training_max": maximum,
                    "below_count": int((numeric < minimum).sum()),
                    "above_count": int((numeric > maximum).sum()),
                    "missing_column": False,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class MultivariateAnomalyDetector:
    """Optional Isolation Forest fitted only on transformed training features."""

    preprocessor: Any
    estimator: IsolationForest
    predictors: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        training_frame: pd.DataFrame,
        *,
        predictors: Sequence[str],
        preprocessor_factory: Callable[[], Any],
        contamination: float | str = "auto",
        seed: int = 42,
        n_jobs: int = 1,
    ) -> MultivariateAnomalyDetector:
        processor = preprocessor_factory()
        matrix = processor.fit_transform(training_frame.loc[:, list(predictors)])
        estimator = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=seed,
            n_jobs=n_jobs,
        )
        estimator.fit(matrix)
        return cls(processor, estimator, tuple(predictors))

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self.preprocessor.transform(frame.loc[:, self.predictors])
        return pd.DataFrame(
            {
                "anomaly_score": self.estimator.decision_function(matrix),
                "is_anomaly": self.estimator.predict(matrix) == -1,
            },
            index=frame.index,
        )


def explanation_manifest(paths: Mapping[str, str]) -> dict[str, Any]:
    """Attach the required association-not-causation language to plot metadata."""

    return {
        "artifacts": dict(paths),
        "interpretation": ASSOCIATION_DISCLAIMER,
    }
