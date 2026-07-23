"""Transparent comparison models for direct UTCI prediction."""

from __future__ import annotations

import copy
import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

LOGGER = logging.getLogger(__name__)


class ComparisonPreprocessor(Protocol):
    """A train-fitted comparison-model transformer."""

    def fit_transform(self, x: pd.DataFrame, y: Any = None) -> Any: ...

    def transform(self, x: pd.DataFrame) -> Any: ...


def heat_index_c(
    air_temperature_c: float | Sequence[float] | np.ndarray,
    relative_humidity_pct: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Calculate the US National Weather Service heat index in degrees Celsius.

    The simple Steadman approximation is used below the Rothfusz-regression
    threshold, with the documented low- and high-humidity adjustments. Inputs are
    dry-bulb temperature in Celsius and relative humidity in percent.
    """

    t_c = np.asarray(air_temperature_c, dtype=float)
    rh = np.asarray(relative_humidity_pct, dtype=float)
    t_c, rh = np.broadcast_arrays(t_c, rh)
    if np.any((rh < 0.0) | (rh > 100.0)):
        raise ValueError("Relative humidity for Heat Index must be in [0, 100] percent.")
    t_f = t_c * 9.0 / 5.0 + 32.0
    simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    simple = (simple + t_f) / 2.0
    rothfusz = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * t_f**2 * rh
        + 0.00085282 * t_f * rh**2
        - 0.00000199 * t_f**2 * rh**2
    )
    low_mask = (rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    low_adjustment = ((13.0 - rh) / 4.0) * np.sqrt(
        np.maximum(0.0, (17.0 - np.abs(t_f - 95.0)) / 17.0)
    )
    rothfusz = np.where(low_mask, rothfusz - low_adjustment, rothfusz)
    high_mask = (rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    high_adjustment = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    rothfusz = np.where(high_mask, rothfusz + high_adjustment, rothfusz)
    result_f = np.where(simple < 80.0, simple, rothfusz)
    result_c = (result_f - 32.0) * 5.0 / 9.0
    if result_c.ndim == 0:
        return float(result_c)
    return result_c


@dataclass(frozen=True)
class ColumnBaseline:
    """A no-fit baseline that returns a named operational input column."""

    column: str
    name: str

    def fit(self, _: pd.DataFrame, __: Sequence[float] | None = None) -> ColumnBaseline:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.column not in frame.columns:
            raise ValueError(f"{self.name} requires column {self.column!r}.")
        return pd.to_numeric(frame[self.column], errors="coerce").to_numpy(dtype=float)


@dataclass(frozen=True)
class HeatIndexBaseline:
    """Heat Index comparison using operational background temperature and RH."""

    temperature_column: str = "background_air_temperature_c"
    humidity_column: str = "background_relative_humidity_pct"
    name: str = "background_heat_index"

    def fit(self, _: pd.DataFrame, __: Sequence[float] | None = None) -> HeatIndexBaseline:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        required = [self.temperature_column, self.humidity_column]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Heat Index baseline is missing columns: {missing}")
        return np.asarray(
            heat_index_c(frame[self.temperature_column], frame[self.humidity_column]),
            dtype=float,
        )


@dataclass
class SklearnBaseline:
    """A fitted scikit-learn estimator and its train-only preprocessor."""

    name: str
    estimator: Any
    preprocessor: ComparisonPreprocessor
    predictors: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame.loc[:, self.predictors])
        return np.asarray(self.estimator.predict(transformed), dtype=float)


@dataclass(frozen=True)
class ElasticNetSelection:
    """Blocked-validation selection for the regularized linear comparison."""

    alpha: float
    l1_ratio: float
    grouped_mae: float


def select_elastic_net_parameters(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    predictors: Sequence[str],
    folds: Sequence[Any],
    preprocessor_factory: Callable[[], ComparisonPreprocessor],
    alphas: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
    l1_ratios: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    groups: Sequence[Any] | None = None,
    seed: int = 42,
) -> ElasticNetSelection:
    """Select Elastic Net settings using only supplied blocked validation folds."""

    y = np.asarray(target, dtype=float)
    group_values = None if groups is None else np.asarray(groups)
    if len(frame) != len(y) or (group_values is not None and len(group_values) != len(y)):
        raise ValueError("Linear-selection rows, outcomes, and groups must align.")
    candidates: list[ElasticNetSelection] = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_scores: list[float] = []
            for fold in folds:
                train = np.asarray(fold.train, dtype=int)
                validation = np.asarray(fold.validation, dtype=int)
                processor = preprocessor_factory()
                x_train = processor.fit_transform(
                    frame.iloc[train].loc[:, list(predictors)], y[train]
                )
                x_validation = processor.transform(
                    frame.iloc[validation].loc[:, list(predictors)]
                )
                estimator = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    random_state=int(seed),
                    max_iter=20_000,
                    selection="cyclic",
                ).fit(x_train, y[train])
                prediction = np.asarray(estimator.predict(x_validation), dtype=float)
                absolute_error = np.abs(y[validation] - prediction)
                if group_values is None:
                    score = float(np.mean(absolute_error))
                else:
                    fold_groups = group_values[validation]
                    score = float(
                        np.mean(
                            [
                                np.mean(absolute_error[fold_groups == group])
                                for group in pd.unique(fold_groups)
                            ]
                        )
                    )
                fold_scores.append(score)
            candidates.append(
                ElasticNetSelection(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    grouped_mae=float(np.mean(fold_scores)),
                )
            )
    return min(candidates, key=lambda item: (item.grouped_mae, item.alpha, item.l1_ratio))


def fit_regularized_linear(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    predictors: Sequence[str],
    preprocessor_factory: Callable[[], ComparisonPreprocessor],
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    seed: int = 42,
) -> SklearnBaseline:
    """Fit standardized Elastic Net on the training partition only."""

    processor = preprocessor_factory()
    transformed = processor.fit_transform(frame.loc[:, list(predictors)], target)
    estimator = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        random_state=int(seed),
        max_iter=20_000,
        selection="cyclic",
    )
    estimator.fit(transformed, np.asarray(target, dtype=float))
    return SklearnBaseline(
        name="regularized_linear",
        estimator=estimator,
        preprocessor=processor,
        predictors=tuple(predictors),
    )


def fit_random_forest(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    predictors: Sequence[str],
    preprocessor_factory: Callable[[], ComparisonPreprocessor],
    seed: int = 42,
    n_jobs: int = 1,
    min_samples_leaf: int = 2,
) -> SklearnBaseline:
    """Fit the prespecified 500-tree Random Forest comparison."""

    processor = preprocessor_factory()
    transformed = processor.fit_transform(frame.loc[:, list(predictors)], target)
    estimator = RandomForestRegressor(
        n_estimators=500,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        min_samples_leaf=int(min_samples_leaf),
        max_features=1.0,
    )
    estimator.fit(transformed, np.asarray(target, dtype=float))
    return SklearnBaseline(
        name="random_forest_500",
        estimator=estimator,
        preprocessor=processor,
        predictors=tuple(predictors),
    )


class TorchMLPRegressor:
    """Small CPU-first PyTorch neural-network baseline with early stopping."""

    def __init__(
        self,
        *,
        hidden_units: Sequence[int] = (128, 64, 32),
        activation: str = "gelu",
        dropout: float = 0.20,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        loss: str = "huber",
        batch_size: int = 128,
        max_epochs: int = 300,
        patience: int = 25,
        min_delta: float = 1.0e-4,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden_units = tuple(int(unit) for unit in hidden_units)
        self.activation = activation.lower()
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss = loss.lower()
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.seed = int(seed)
        self.device = device.lower()
        self.model_: Any | None = None
        self.best_epoch_: int | None = None
        self.validation_loss_: float | None = None

    @staticmethod
    def _dense_float32(matrix: Any) -> np.ndarray:
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Neural-network predictors must be a 2-D matrix.")
        if not np.all(np.isfinite(array)):
            raise ValueError(
                "Neural-network preprocessing must impute NaNs before model fitting."
            )
        return array

    def fit(
        self,
        x_train: Any,
        y_train: Sequence[float],
        *,
        x_validation: Any,
        y_validation: Sequence[float],
    ) -> TorchMLPRegressor:
        """Fit using only the supplied training and inner-validation partitions."""

        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "The neural baseline requires the pinned torch dependency."
            ) from exc

        if self.device != "cpu" and not torch.cuda.is_available():
            raise ValueError(
                f"Requested device {self.device!r} is unavailable; CPU is the default."
            )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            LOGGER.warning("Full deterministic PyTorch algorithms are unavailable.")

        train_x = self._dense_float32(x_train)
        valid_x = self._dense_float32(x_validation)
        train_y = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        valid_y = np.asarray(y_validation, dtype=np.float32).reshape(-1, 1)
        if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
            raise ValueError("Neural baseline predictors and outcomes do not align.")

        activation_factory: Callable[[], nn.Module]
        if self.activation == "relu":
            activation_factory = nn.ReLU
        elif self.activation == "gelu":
            activation_factory = nn.GELU
        else:
            raise ValueError("Neural activation must be 'relu' or 'gelu'.")
        layers: list[nn.Module] = []
        previous = train_x.shape[1]
        for width in self.hidden_units:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    activation_factory(),
                    nn.Dropout(self.dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        model = nn.Sequential(*layers).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.loss == "huber":
            criterion: nn.Module = nn.SmoothL1Loss()
        elif self.loss == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError("Neural loss must be 'huber' or 'mae'.")

        generator = torch.Generator().manual_seed(self.seed)
        dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        valid_x_tensor = torch.from_numpy(valid_x).to(self.device)
        valid_y_tensor = torch.from_numpy(valid_y).to(self.device)
        best_loss = math.inf
        best_state: dict[str, Any] | None = None
        stale_epochs = 0

        for epoch in range(self.max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss_value = criterion(model(batch_x), batch_y)
                loss_value.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    criterion(model(valid_x_tensor), valid_y_tensor).cpu().item()
                )
            if validation_loss < best_loss - self.min_delta:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                self.best_epoch_ = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("Neural baseline failed to produce a finite validation loss.")
        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.validation_loss_ = float(best_loss)
        return self

    def predict(self, matrix: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Neural baseline has not been fitted.")
        import torch

        values = self._dense_float32(matrix)
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(self.device)
            return self.model_(tensor).cpu().numpy().reshape(-1).astype(float)


@dataclass
class NeuralBaseline:
    """Fitted neural estimator with train-fitted standardized preprocessing."""

    estimator: TorchMLPRegressor
    preprocessor: ComparisonPreprocessor
    predictors: tuple[str, ...]
    name: str = "neural_network"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame.loc[:, self.predictors])
        return self.estimator.predict(matrix)


def fit_neural_baseline(
    train_frame: pd.DataFrame,
    train_target: Sequence[float],
    validation_frame: pd.DataFrame,
    validation_target: Sequence[float],
    *,
    predictors: Sequence[str],
    preprocessor_factory: Callable[[], ComparisonPreprocessor],
    config: Mapping[str, Any],
) -> NeuralBaseline:
    """Fit the opt-in neural comparison with inner-fold early stopping only."""

    processor = preprocessor_factory()
    x_train = processor.fit_transform(
        train_frame.loc[:, list(predictors)], train_target
    )
    x_validation = processor.transform(validation_frame.loc[:, list(predictors)])
    estimator = TorchMLPRegressor(
        hidden_units=config.get("hidden_units", (128, 64, 32)),
        activation=str(config.get("activation", "gelu")),
        dropout=float(config.get("dropout", 0.20)),
        learning_rate=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
        loss=str(config.get("loss", "huber")),
        batch_size=int(config.get("batch_size", 128)),
        max_epochs=int(config.get("max_epochs", 300)),
        patience=int(config.get("patience", config.get("early_stopping_patience", 25))),
        min_delta=float(config.get("min_delta", 1.0e-4)),
        seed=int(config.get("seed", 42)),
        device=str(config.get("device", "cpu")),
    ).fit(
        x_train,
        train_target,
        x_validation=x_validation,
        y_validation=validation_target,
    )
    return NeuralBaseline(
        estimator=estimator,
        preprocessor=processor,
        predictors=tuple(predictors),
    )


def configured_simple_baselines(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the no-fit operational and eligible-satellite baselines."""

    return {
        "background_air_temperature": ColumnBaseline(
            column=str(
                config.get(
                    "background_temperature_column", "background_air_temperature_c"
                )
            ),
            name="background_air_temperature",
        ),
        "background_heat_index": HeatIndexBaseline(
            temperature_column=str(
                config.get(
                    "background_temperature_column", "background_air_temperature_c"
                )
            ),
            humidity_column=str(
                config.get(
                    "background_humidity_column", "background_relative_humidity_pct"
                )
            ),
        ),
        "satellite_lst_alone": ColumnBaseline(
            column=str(config.get("satellite_lst_column", "satellite_lst_c")),
            name="satellite_lst_alone_eligible_rows_only",
        ),
    }
