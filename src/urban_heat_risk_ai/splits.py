"""Deterministic leakage-safe spatial and temporal blocking.

This module never performs random row splitting.  A validation fold is the
intersection of held-out spatial blocks and held-out dates/weather events; rows
sharing either a validation block or validation date/event are embargoed, not
returned to training.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from .errors import SplitInvariantError


def _stable_hash(value: Any, *, seed: int, salt: str) -> str:
    payload = f"{seed}|{salt}|{value!s}".encode()
    return hashlib.sha256(payload).hexdigest()


def balanced_group_assignment(
    groups: Sequence[Any],
    *,
    n_folds: int,
    seed: int,
    salt: str,
) -> dict[str, int]:
    """Assign whole groups to folds deterministically while balancing row counts.

    This pure metadata operation is independent of outcomes and model performance.
    Group values are normalized to strings in the returned mapping.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    normalized = [str(value) for value in groups]
    if any(not value or value.lower() in {"nan", "nat", "<na>"} for value in normalized):
        raise SplitInvariantError(f"{salt} groups contain missing or blank values.")
    counts: dict[str, int] = {}
    for value in normalized:
        counts[value] = counts.get(value, 0) + 1
    if len(counts) < n_folds:
        raise SplitInvariantError(
            f"Need at least {n_folds} distinct {salt} groups, found {len(counts)}. "
            "Reduce the configured fold count or collect broader real-data coverage."
        )
    order = sorted(counts, key=lambda value: _stable_hash(value, seed=seed, salt=salt))
    totals = [0] * n_folds
    assignment: dict[str, int] = {}
    for position, value in enumerate(order):
        if position < n_folds:
            fold = position
        else:
            fold = min(range(n_folds), key=lambda candidate: (totals[candidate], candidate))
        assignment[value] = fold
        totals[fold] += counts[value]
    return assignment


def assert_disjoint_spatiotemporal_groups(
    train_spatial_blocks: Sequence[Any],
    train_dates: Sequence[Any],
    validation_spatial_blocks: Sequence[Any],
    validation_dates: Sequence[Any],
    *,
    train_events: Sequence[Any] | None = None,
    validation_events: Sequence[Any] | None = None,
) -> None:
    """Pure invariant check used both by split generation and scalar tests."""

    shared_blocks = set(map(str, train_spatial_blocks)).intersection(
        map(str, validation_spatial_blocks)
    )
    if shared_blocks:
        raise SplitInvariantError(
            f"Training and validation share spatial block(s): {sorted(shared_blocks)}"
        )
    shared_dates = set(map(str, train_dates)).intersection(map(str, validation_dates))
    if shared_dates:
        raise SplitInvariantError(
            f"Training and validation share complete date(s): {sorted(shared_dates)}"
        )
    if train_events is not None and validation_events is not None:
        train_nonmissing = {
            str(value)
            for value in train_events
            if pd.notna(value) and str(value).strip() not in {"", "<NA>"}
        }
        validation_nonmissing = {
            str(value)
            for value in validation_events
            if pd.notna(value) and str(value).strip() not in {"", "<NA>"}
        }
        shared_events = train_nonmissing.intersection(validation_nonmissing)
        if shared_events:
            raise SplitInvariantError(
                f"Training and validation share weather event(s): {sorted(shared_events)}"
            )


def construct_spatial_block_ids(
    longitudes: Sequence[float],
    latitudes: Sequence[float],
    *,
    projected_crs: str,
    block_size_m: float,
    source_crs: str = "EPSG:4326",
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
) -> np.ndarray:
    """Project WGS84 coordinates and assign deterministic fixed-size grid blocks.

    The CRS and block size are explicit design parameters.  This utility contains
    no model scores and cannot tune block size according to predictive performance.
    """

    longitude = np.asarray(longitudes, dtype=float)
    latitude = np.asarray(latitudes, dtype=float)
    if longitude.ndim != 1 or latitude.ndim != 1 or longitude.shape != latitude.shape:
        raise ValueError("Longitude and latitude must be equal-length one-dimensional sequences.")
    if longitude.size == 0:
        return np.asarray([], dtype=object)
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise SplitInvariantError("Coordinates must be finite before spatial block construction.")
    if ((latitude < -90) | (latitude > 90)).any() or ((longitude < -180) | (longitude > 180)).any():
        raise SplitInvariantError("Coordinates fall outside valid WGS84 latitude/longitude ranges.")
    if not math.isfinite(float(block_size_m)) or float(block_size_m) <= 0:
        raise ValueError("block_size_m must be a finite positive design value.")

    target = CRS.from_user_input(projected_crs)
    if not target.is_projected:
        raise SplitInvariantError(
            f"Configured spatial blocking CRS {projected_crs!r} is not projected. "
            "Use a documented projected CRS whose axis units convert to meters."
        )
    axis_info = target.axis_info
    conversion = float(axis_info[0].unit_conversion_factor) if axis_info else 1.0
    if not math.isfinite(conversion) or conversion <= 0:
        raise SplitInvariantError(f"Cannot convert {projected_crs!r} coordinates to meters.")
    transformer = Transformer.from_crs(source_crs, target, always_xy=True)
    projected_x, projected_y = transformer.transform(longitude.tolist(), latitude.tolist())
    x_m = np.asarray(projected_x, dtype=float) * conversion
    y_m = np.asarray(projected_y, dtype=float) * conversion
    if not np.isfinite(x_m).all() or not np.isfinite(y_m).all():
        raise SplitInvariantError("Coordinate projection produced non-finite values.")
    x_index = np.floor((x_m - float(origin_x_m)) / float(block_size_m)).astype(np.int64)
    y_index = np.floor((y_m - float(origin_y_m)) / float(block_size_m)).astype(np.int64)
    authority = target.to_authority()
    crs_tag = f"{authority[0]}{authority[1]}" if authority else target.to_string()
    return np.asarray(
        [
            f"{crs_tag}_x{x_value}_y{y_value}"
            for x_value, y_value in zip(x_index, y_index, strict=True)
        ],
        dtype=object,
    )


def ensure_spatial_blocks(
    metadata: pd.DataFrame,
    split_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return a copy with deterministic blocks when the configured column is absent."""

    spatial = split_config.get("spatial", split_config)
    if not isinstance(spatial, Mapping):
        raise SplitInvariantError("splits.spatial configuration must be a mapping.")
    block_column = str(spatial.get("block_column", "spatial_block_id"))
    result = metadata.copy()
    if block_column in result.columns and result[block_column].notna().all():
        return result
    coordinate_columns = spatial.get("coordinate_columns", ("longitude", "latitude"))
    if not isinstance(coordinate_columns, Sequence) or len(coordinate_columns) != 2:
        raise SplitInvariantError("spatial.coordinate_columns must be [longitude, latitude].")
    longitude_column, latitude_column = map(str, coordinate_columns)
    missing = [name for name in (longitude_column, latitude_column) if name not in result.columns]
    if missing:
        raise SplitInvariantError(
            f"Cannot construct spatial blocks; coordinate columns are absent: {missing}"
        )
    projected_crs = spatial.get("projected_crs", spatial.get("crs"))
    if not projected_crs:
        raise SplitInvariantError("A projected CRS must be configured before deriving blocks.")
    derived = construct_spatial_block_ids(
        result[longitude_column].to_numpy(),
        result[latitude_column].to_numpy(),
        projected_crs=str(projected_crs),
        source_crs=str(spatial.get("source_crs", "EPSG:4326")),
        block_size_m=float(spatial.get("block_size_m", 0)),
        origin_x_m=float(spatial.get("block_origin_x_m", 0.0)),
        origin_y_m=float(spatial.get("block_origin_y_m", 0.0)),
    )
    if block_column in result.columns:
        missing_mask = result[block_column].isna()
        supplied = result[block_column].astype("string")
        conflict = (~missing_mask) & supplied.ne(pd.Series(derived, index=result.index, dtype="string"))
        if bool(conflict.any()):
            raise SplitInvariantError(
                f"Supplied {block_column} disagrees with configured deterministic blocks on "
                f"{int(conflict.sum())} rows; resolve the manifest instead of overwriting IDs."
            )
        result.loc[missing_mask, block_column] = derived[missing_mask.to_numpy()]
    else:
        result[block_column] = derived
    return result


# Descriptive alias for callers.
create_spatial_blocks = ensure_spatial_blocks


@dataclass(frozen=True)
class BlockedFold:
    fold: int
    train: np.ndarray
    validation: np.ndarray
    embargo: np.ndarray
    validation_spatial_blocks: tuple[str, ...]
    validation_dates: tuple[str, ...]
    validation_events: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.train) == 0 or len(self.validation) == 0:
            raise SplitInvariantError(f"Blocked fold {self.fold} has an empty train/validation set.")


@dataclass(frozen=True)
class NestedBlockedFold:
    outer: BlockedFold
    inner: tuple[BlockedFold, ...]


class SpatioTemporalBlockedSplit:
    """Scikit-learn-style dual spatial/temporal blocked cross-validator."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        spatial_block_column: str = "spatial_block_id",
        date_column: str = "date",
        event_column: str | None = "weather_event_id",
        random_state: int = 0,
        temporal_embargo_days: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if temporal_embargo_days < 0:
            raise ValueError("temporal_embargo_days cannot be negative.")
        self.n_splits = int(n_splits)
        self.spatial_block_column = spatial_block_column
        self.date_column = date_column
        self.event_column = event_column
        self.random_state = int(random_state)
        self.temporal_embargo_days = int(temporal_embargo_days)

    @classmethod
    def from_config(
        cls, split_config: Mapping[str, Any], *, inner: bool = False
    ) -> SpatioTemporalBlockedSplit:
        spatial = split_config.get("spatial", {})
        temporal = split_config.get("temporal", {})
        embargo = split_config.get("embargo", {})
        if not isinstance(spatial, Mapping) or not isinstance(temporal, Mapping):
            raise SplitInvariantError("Split spatial/temporal sections must be mappings.")
        n_splits = split_config.get("inner_folds" if inner else "outer_folds", 5)
        embargo_days = 0
        if isinstance(embargo, Mapping):
            embargo_days = int(embargo.get("days", embargo.get("temporal_days", 0)))
        return cls(
            int(n_splits),
            spatial_block_column=str(spatial.get("block_column", "spatial_block_id")),
            date_column=str(temporal.get("date_column", "date")),
            event_column=(
                str(temporal["weather_event_column"])
                if temporal.get("weather_event_column")
                else None
            ),
            random_state=int(split_config.get("random_seed", 0)) + (10_007 if inner else 0),
            temporal_embargo_days=embargo_days,
        )

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def _metadata_arrays(
        self, metadata: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("SpatioTemporalBlockedSplit requires a metadata DataFrame.")
        required = {self.spatial_block_column, self.date_column}
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise SplitInvariantError(f"Split metadata is missing column(s): {missing}")
        if metadata[list(required)].isna().any().any():
            raise SplitInvariantError("Spatial blocks and dates cannot be missing during splitting.")
        blocks = metadata[self.spatial_block_column].astype("string").to_numpy(dtype=str)
        date_values = pd.to_datetime(metadata[self.date_column], errors="coerce")
        if bool(date_values.isna().any()):
            raise SplitInvariantError("Split dates contain unparseable values.")
        dates = date_values.strftime("%Y-%m-%d").to_numpy(dtype=str)
        events: np.ndarray | None = None
        if self.event_column and self.event_column in metadata.columns:
            event_series = metadata[self.event_column].astype("string")
            # Missing event IDs fall back to their complete date, with a namespace
            # prefix to prevent accidental equality with a real event ID.
            events = np.asarray(
                [
                    f"event:{event}" if pd.notna(event) and str(event).strip() else f"date:{date}"
                    for event, date in zip(event_series.tolist(), dates, strict=True)
                ],
                dtype=str,
            )
        temporal_groups = events if events is not None else dates
        return blocks, dates, events, temporal_groups

    def _pair_temporal_folds(
        self,
        block_fold: np.ndarray,
        temporal_fold: np.ndarray,
    ) -> tuple[int, ...]:
        candidates: Iterator[tuple[int, ...]]
        if self.n_splits <= 8:
            candidates = itertools.permutations(range(self.n_splits))
        else:
            candidates = (
                tuple((fold + offset) % self.n_splits for fold in range(self.n_splits))
                for offset in range(self.n_splits)
            )
        best: tuple[tuple[int, int, tuple[int, ...]], tuple[int, ...]] | None = None
        for pairing in candidates:
            sizes = [
                int(((block_fold == fold) & (temporal_fold == pairing[fold])).sum())
                for fold in range(self.n_splits)
            ]
            score = (min(sizes), -int(np.ptp(sizes)), tuple(-size for size in sizes))
            if best is None or score > best[0]:
                best = (score, tuple(pairing))
        if best is None or best[0][0] <= 0:
            raise SplitInvariantError(
                "Could not form nonempty dual-blocked validation intersections. "
                "Inspect spatial/date coverage or reduce the fold count; do not fall back to random rows."
            )
        return best[1]

    def split_with_embargo(self, metadata: pd.DataFrame) -> Iterator[BlockedFold]:
        blocks, dates, events, temporal_groups = self._metadata_arrays(metadata)
        spatial_assignment = balanced_group_assignment(
            blocks, n_folds=self.n_splits, seed=self.random_state, salt="spatial_block"
        )
        temporal_assignment = balanced_group_assignment(
            temporal_groups,
            n_folds=self.n_splits,
            seed=self.random_state,
            salt="weather_event" if events is not None else "date",
        )
        block_fold = np.asarray([spatial_assignment[value] for value in blocks], dtype=int)
        temporal_fold = np.asarray(
            [temporal_assignment[value] for value in temporal_groups], dtype=int
        )
        pairing = self._pair_temporal_folds(block_fold, temporal_fold)
        positions = np.arange(len(metadata), dtype=int)
        parsed_dates = pd.to_datetime(dates)

        for fold in range(self.n_splits):
            validation_mask = (block_fold == fold) & (temporal_fold == pairing[fold])
            validation = positions[validation_mask]
            if validation.size == 0:
                raise SplitInvariantError(f"Fold {fold} has no validation observations.")
            validation_blocks = set(blocks[validation_mask])
            validation_dates = set(dates[validation_mask])
            validation_events = set(events[validation_mask]) if events is not None else set()
            restriction = np.isin(blocks, list(validation_blocks)) | np.isin(
                dates, list(validation_dates)
            )
            if events is not None:
                restriction |= np.isin(events, list(validation_events))
            if self.temporal_embargo_days:
                for held_date in pd.to_datetime(sorted(validation_dates)):
                    delta = np.abs((parsed_dates - held_date).days)
                    restriction |= delta <= self.temporal_embargo_days
            train_mask = ~restriction
            train = positions[train_mask]
            embargo = positions[~validation_mask & ~train_mask]
            assert_split_invariants(
                metadata,
                train,
                validation,
                embargo=embargo,
                spatial_block_column=self.spatial_block_column,
                date_column=self.date_column,
                event_column=self.event_column,
            )
            yield BlockedFold(
                fold=fold,
                train=train,
                validation=validation,
                embargo=embargo,
                validation_spatial_blocks=tuple(sorted(validation_blocks)),
                validation_dates=tuple(sorted(validation_dates)),
                validation_events=tuple(sorted(validation_events)),
            )

    def split(
        self,
        X: pd.DataFrame,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield sklearn-compatible train/validation positions."""

        for fold in self.split_with_embargo(X):
            yield fold.train, fold.validation

    def nested_split(
        self, metadata: pd.DataFrame, *, inner_n_splits: int
    ) -> Iterator[NestedBlockedFold]:
        """Yield outer folds with inner folds confined to outer training rows."""

        for outer in self.split_with_embargo(metadata):
            outer_training = metadata.iloc[outer.train]
            inner_splitter = SpatioTemporalBlockedSplit(
                inner_n_splits,
                spatial_block_column=self.spatial_block_column,
                date_column=self.date_column,
                event_column=self.event_column,
                random_state=self.random_state + 10_007 + outer.fold,
                temporal_embargo_days=self.temporal_embargo_days,
            )
            inner_global: list[BlockedFold] = []
            for inner in inner_splitter.split_with_embargo(outer_training):
                inner_global.append(
                    BlockedFold(
                        fold=inner.fold,
                        train=outer.train[inner.train],
                        validation=outer.train[inner.validation],
                        embargo=outer.train[inner.embargo],
                        validation_spatial_blocks=inner.validation_spatial_blocks,
                        validation_dates=inner.validation_dates,
                        validation_events=inner.validation_events,
                    )
                )
            yield NestedBlockedFold(outer=outer, inner=tuple(inner_global))


def assert_split_invariants(
    metadata: pd.DataFrame,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    embargo: Sequence[int] | None = None,
    spatial_block_column: str = "spatial_block_id",
    date_column: str = "date",
    event_column: str | None = "weather_event_id",
) -> None:
    """Assert positional and dual-group leakage invariants before training."""

    train = np.asarray(train_indices, dtype=int)
    validation = np.asarray(validation_indices, dtype=int)
    if train.ndim != 1 or validation.ndim != 1:
        raise SplitInvariantError("Split indices must be one-dimensional.")
    if not train.size or not validation.size:
        raise SplitInvariantError("Training and validation must both be nonempty.")
    if ((train < 0) | (train >= len(metadata))).any() or (
        (validation < 0) | (validation >= len(metadata))
    ).any():
        raise SplitInvariantError("Split indices fall outside the metadata table.")
    if np.intersect1d(train, validation).size:
        raise SplitInvariantError("Training and validation row positions overlap.")
    if embargo is not None:
        embargo_array = np.asarray(embargo, dtype=int)
        if (
            np.intersect1d(train, embargo_array).size
            or np.intersect1d(validation, embargo_array).size
        ):
            raise SplitInvariantError("Embargo rows overlap train or validation rows.")
    for column in (spatial_block_column, date_column):
        if column not in metadata.columns:
            raise SplitInvariantError(f"Cannot assert split invariants without {column!r}.")
    train_events = None
    validation_events = None
    if event_column and event_column in metadata.columns:
        train_events = metadata.iloc[train][event_column].tolist()
        validation_events = metadata.iloc[validation][event_column].tolist()
    assert_disjoint_spatiotemporal_groups(
        metadata.iloc[train][spatial_block_column].tolist(),
        metadata.iloc[train][date_column].tolist(),
        metadata.iloc[validation][spatial_block_column].tolist(),
        metadata.iloc[validation][date_column].tolist(),
        train_events=train_events,
        validation_events=validation_events,
    )
    if "sample_id" in metadata.columns:
        train_ids = set(metadata.iloc[train]["sample_id"].astype(str))
        validation_ids = set(metadata.iloc[validation]["sample_id"].astype(str))
        overlap = train_ids.intersection(validation_ids)
        if overlap:
            raise SplitInvariantError(
                f"Training/validation share sample_id values: {sorted(overlap)[:5]}"
            )


def _select_hashed_groups(
    values: Sequence[str], count: int, *, seed: int, salt: str
) -> tuple[str, ...]:
    unique = sorted(set(map(str, values)), key=lambda item: _stable_hash(item, seed=seed, salt=salt))
    return tuple(unique[:count])


def assign_partition_roles(
    metadata: pd.DataFrame,
    split_config: Mapping[str, Any],
    *,
    target_column: str = "calculated_utci_c",
) -> pd.Series:
    """Deterministically design development/calibration/final intersections.

    Crossed rows (a held-out location on a development date, for example) receive
    ``embargo``.  Selection uses only configured grouping rules and the fixed hot
    threshold; it never uses model performance.
    """

    spatial = split_config.get("spatial", {})
    temporal = split_config.get("temporal", {})
    calibration_config = split_config.get("calibration", {})
    final_config = split_config.get("final_test", {})
    if not all(isinstance(value, Mapping) for value in (spatial, temporal, calibration_config, final_config)):
        raise SplitInvariantError("Malformed spatial/temporal/calibration/final split configuration.")
    block_column = str(spatial.get("block_column", "spatial_block_id"))
    date_column = str(temporal.get("date_column", "date"))
    event_column = temporal.get("weather_event_column")
    required = {block_column, date_column, "site_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise SplitInvariantError(f"Cannot assign partition roles; missing metadata: {missing}")
    if metadata[list(required)].isna().any().any():
        raise SplitInvariantError("Partition grouping metadata cannot be missing.")

    seed = int(split_config.get("random_seed", 0))
    blocks = metadata[block_column].astype(str)
    dates = pd.to_datetime(metadata[date_column], errors="raise").strftime("%Y-%m-%d")
    if event_column and event_column in metadata.columns:
        event_values = metadata[str(event_column)].astype("string")
        temporal_groups = pd.Series(
            [
                f"event:{event}" if pd.notna(event) and str(event).strip() else f"date:{date}"
                for event, date in zip(
                    event_values.tolist(), dates.tolist(), strict=True
                )
            ],
            index=metadata.index,
        )
    else:
        temporal_groups = dates.map(lambda value: f"date:{value}")

    unique_blocks = sorted(blocks.unique())
    final_fraction = float(final_config.get("fraction_of_spatial_day_blocks", 0.15))
    calibration_fraction = float(
        calibration_config.get("fraction_of_spatial_day_blocks", 0.15)
    )
    if not (0 < final_fraction < 1 and 0 < calibration_fraction < 1):
        raise SplitInvariantError("Calibration/final fractions must each be between zero and one.")
    minimum_sites = int(final_config.get("minimum_geographically_separated_sites", 4))
    if len(unique_blocks) < minimum_sites:
        raise SplitInvariantError(
            f"Locked final design needs at least {minimum_sites} distinct spatial blocks; "
            f"only {len(unique_blocks)} exist."
        )
    final_block_count = max(minimum_sites, math.ceil(len(unique_blocks) * final_fraction))
    if final_block_count >= len(unique_blocks):
        raise SplitInvariantError("Final spatial holdout would leave no development blocks.")
    final_blocks = _select_hashed_groups(
        unique_blocks, final_block_count, seed=seed, salt="final_spatial"
    )
    remaining_blocks = [value for value in unique_blocks if value not in final_blocks]
    calibration_block_count = max(1, math.ceil(len(unique_blocks) * calibration_fraction))
    if calibration_block_count >= len(remaining_blocks):
        raise SplitInvariantError("Calibration spatial holdout would leave no development blocks.")
    calibration_blocks = _select_hashed_groups(
        remaining_blocks, calibration_block_count, seed=seed, salt="calibration_spatial"
    )
    development_blocks = set(remaining_blocks).difference(calibration_blocks)

    unique_temporal = sorted(temporal_groups.unique())
    if len(unique_temporal) < 3:
        raise SplitInvariantError(
            "Need at least three distinct complete dates/weather events for development, "
            "calibration, and final partitions."
        )
    final_time_count = max(1, math.ceil(len(unique_temporal) * final_fraction))
    calibration_time_count = max(1, math.ceil(len(unique_temporal) * calibration_fraction))
    if final_time_count + calibration_time_count >= len(unique_temporal):
        raise SplitInvariantError("Temporal holdouts would leave no development dates/events.")

    hot_groups: list[str] = []
    if target_column in metadata.columns:
        threshold = float(final_config.get("hot_date_threshold_c", 32.0))
        target = pd.to_numeric(metadata[target_column], errors="coerce")
        hot_groups = sorted(set(temporal_groups[target.ge(threshold)]))
    minimum_hot = int(final_config.get("minimum_complete_hot_dates", 1))
    preferred_hot = int(final_config.get("preferred_complete_hot_dates", 2))
    if hot_groups and len(hot_groups) < minimum_hot:
        raise SplitInvariantError(
            f"Real data contain only {len(hot_groups)} hot temporal groups; final design requires "
            f"{minimum_hot}."
        )
    selected_hot_count = min(len(hot_groups), max(minimum_hot, preferred_hot))
    selected_hot = _select_hashed_groups(
        hot_groups, selected_hot_count, seed=seed, salt="final_hot_time"
    )
    final_time_count = max(final_time_count, len(selected_hot))
    remaining_final_candidates = [value for value in unique_temporal if value not in selected_hot]
    additional_final = _select_hashed_groups(
        remaining_final_candidates,
        final_time_count - len(selected_hot),
        seed=seed,
        salt="final_time",
    )
    final_times = tuple(selected_hot) + tuple(additional_final)
    remaining_times = [value for value in unique_temporal if value not in final_times]
    calibration_times = _select_hashed_groups(
        remaining_times, calibration_time_count, seed=seed, salt="calibration_time"
    )
    development_times = set(remaining_times).difference(calibration_times)

    role = pd.Series("embargo", index=metadata.index, dtype="string", name="split_role")
    role.loc[blocks.isin(final_blocks) & temporal_groups.isin(final_times)] = "final_test"
    role.loc[blocks.isin(calibration_blocks) & temporal_groups.isin(calibration_times)] = (
        "calibration"
    )
    role.loc[blocks.isin(development_blocks) & temporal_groups.isin(development_times)] = (
        "development"
    )
    if not role.eq("final_test").any() or not role.eq("calibration").any() or not role.eq(
        "development"
    ).any():
        raise SplitInvariantError(
            "Deterministic role intersections produced an empty partition; inspect real-data "
            "spatial/temporal coverage rather than using random rows."
        )
    return role


def assert_partition_role_invariants(
    metadata: pd.DataFrame,
    roles: Sequence[Any] | pd.Series,
    *,
    spatial_block_column: str = "spatial_block_id",
    date_column: str = "date",
    event_column: str | None = "weather_event_id",
) -> None:
    """Require calibration/final to be unseen in both space and time from development."""

    role = pd.Series(roles, index=metadata.index, dtype="string")
    for held_role in ("calibration", "final_test"):
        held = np.flatnonzero(role.eq(held_role).to_numpy())
        development = np.flatnonzero(role.eq("development").to_numpy())
        if not held.size or not development.size:
            raise SplitInvariantError(f"Partition {held_role!r} or development is empty.")
        assert_split_invariants(
            metadata,
            development,
            held,
            spatial_block_column=spatial_block_column,
            date_column=date_column,
            event_column=event_column,
        )
    calibration = np.flatnonzero(role.eq("calibration").to_numpy())
    final = np.flatnonzero(role.eq("final_test").to_numpy())
    assert_split_invariants(
        metadata,
        calibration,
        final,
        spatial_block_column=spatial_block_column,
        date_column=date_column,
        event_column=event_column,
    )


def assert_final_test_design(
    metadata: pd.DataFrame,
    roles: Sequence[Any] | pd.Series,
    final_config: Mapping[str, Any],
    *,
    target_column: str = "calculated_utci_c",
    date_column: str = "date",
    site_column: str = "site_id",
    spatial_block_column: str = "spatial_block_id",
) -> None:
    """Enforce the locked final-test geographic and complete-hot-date design."""

    role = pd.Series(roles, index=metadata.index, dtype="string")
    final_mask = role.eq("final_test")
    if not bool(final_mask.any()):
        raise SplitInvariantError("Locked final-test partition is empty.")
    required_sites = int(final_config.get("minimum_geographically_separated_sites", 4))
    final_sites = metadata.loc[final_mask, site_column].nunique()
    final_blocks = metadata.loc[final_mask, spatial_block_column].nunique()
    if final_sites < required_sites or final_blocks < required_sites:
        raise SplitInvariantError(
            f"Final test needs at least {required_sites} sites in distinct spatial blocks; found "
            f"{final_sites} sites across {final_blocks} blocks."
        )
    final_site_ids = set(metadata.loc[final_mask, site_column].astype(str))
    comparison_mask = role.isin(["development", "calibration"])
    reused_sites = final_site_ids.intersection(
        metadata.loc[comparison_mask, site_column].astype(str)
    )
    if reused_sites:
        raise SplitInvariantError(
            "Final-test sites must be completely unseen outside embargo rows; reused site_id "
            f"values include {sorted(reused_sites)[:5]}."
        )
    minimum_hot_dates = int(final_config.get("minimum_complete_hot_dates", 1))
    if target_column not in metadata.columns:
        raise SplitInvariantError(
            f"Cannot verify final hot-date design without continuous target {target_column!r}."
        )
    threshold = float(final_config.get("hot_date_threshold_c", 32.0))
    hot_rows = pd.to_numeric(metadata[target_column], errors="coerce").ge(threshold)
    hot_dates = set(metadata.loc[final_mask & hot_rows, date_column].astype(str))
    if len(hot_dates) < minimum_hot_dates:
        raise SplitInvariantError(
            f"Final test has {len(hot_dates)} complete hot date(s) at >= {threshold} °C; "
            f"at least {minimum_hot_dates} are required."
        )
    # No hot date can be only partly final; non-final rows on it must be embargoed.
    for date in hot_dates:
        date_roles = set(role.loc[metadata[date_column].astype(str).eq(date)].dropna())
        if date_roles.difference({"final_test", "embargo"}):
            raise SplitInvariantError(
                f"Hot final date {date} is not held out complete; roles are {sorted(date_roles)}."
            )


def create_split_manifest(
    metadata: pd.DataFrame,
    split_config: Mapping[str, Any],
    *,
    target_column: str = "calculated_utci_c",
) -> pd.DataFrame:
    """Create a separate deterministic manifest without changing source metadata."""

    prepared = ensure_spatial_blocks(metadata, split_config)
    if "split_role" in prepared.columns:
        supplied_roles = prepared["split_role"].astype("string").str.strip()
        assigned = supplied_roles.isin(["development", "calibration", "final_test", "embargo"])
        unassigned = supplied_roles.eq("unassigned")
        invalid = supplied_roles.isna() | ~(assigned | unassigned)
        if bool(invalid.any()):
            raise SplitInvariantError(
                "Source split_role contains unsupported values: "
                f"{sorted(set(supplied_roles[invalid].dropna()))}."
            )
        if bool(assigned.any()) and bool(unassigned.any()):
            raise SplitInvariantError(
                "Source split_role mixes assigned and unassigned rows. Complete the intentional "
                "role design or set every row to unassigned before make-splits."
            )
        roles = (
            supplied_roles.rename("split_role")
            if bool(assigned.any())
            else assign_partition_roles(prepared, split_config, target_column=target_column)
        )
    else:
        roles = assign_partition_roles(prepared, split_config, target_column=target_column)
    spatial = split_config.get("spatial", {})
    temporal = split_config.get("temporal", {})
    block_column = str(spatial.get("block_column", "spatial_block_id"))
    date_column = str(temporal.get("date_column", "date"))
    event_column = temporal.get("weather_event_column")
    assert_partition_role_invariants(
        prepared,
        roles,
        spatial_block_column=block_column,
        date_column=date_column,
        event_column=str(event_column) if event_column else None,
    )
    final_config = split_config.get("final_test", {})
    assert_final_test_design(
        prepared,
        roles,
        final_config if isinstance(final_config, Mapping) else {},
        target_column=target_column,
        date_column=date_column,
        spatial_block_column=block_column,
    )

    manifest_columns = ["sample_id", "site_id", block_column, date_column]
    if event_column and event_column in prepared.columns:
        manifest_columns.append(str(event_column))
    missing = [column for column in manifest_columns if column not in prepared.columns]
    if missing:
        raise SplitInvariantError(f"Cannot create manifest; metadata columns absent: {missing}")
    manifest = prepared.loc[:, manifest_columns].copy()
    manifest["split_role"] = roles
    manifest["outer_fold"] = -1
    development_positions = np.flatnonzero(roles.eq("development").to_numpy())
    development = prepared.iloc[development_positions]
    splitter = SpatioTemporalBlockedSplit.from_config(split_config)
    for fold in splitter.split_with_embargo(development):
        global_validation = development_positions[fold.validation]
        manifest.iloc[global_validation, manifest.columns.get_loc("outer_fold")] = fold.fold
    manifest["manifest_version"] = str(split_config.get("manifest_version", "1.0"))
    return manifest


def split_manifest_hash(manifest: pd.DataFrame) -> str:
    """Hash manifest values canonically, independent of row order."""

    if "sample_id" not in manifest.columns:
        raise SplitInvariantError("Manifest hash requires sample_id.")
    records = manifest.sort_values("sample_id").replace({np.nan: None}).to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload.get("rows") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise SplitInvariantError(f"Manifest JSON must contain a rows list: {path}")
        manifest = pd.DataFrame.from_records(rows)
        if isinstance(payload, Mapping):
            recorded_hash = payload.get("manifest_hash")
            if not isinstance(recorded_hash, str) or not recorded_hash:
                raise SplitInvariantError(
                    f"Manifest JSON is missing its required integrity hash: {path}"
                )
            observed_hash = split_manifest_hash(manifest)
            if observed_hash != recorded_hash:
                raise SplitInvariantError(
                    f"Manifest JSON integrity hash mismatch: {path}. Do not use a modified or "
                    "partially written split manifest."
                )
        return manifest
    raise SplitInvariantError(f"Split manifest must be .json, .csv, or .parquet: {path}")


def _write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        manifest.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        manifest.to_parquet(path, index=False)
    elif suffix == ".json":
        payload = {
            "manifest_hash": split_manifest_hash(manifest),
            "rows": manifest.replace({np.nan: None}).to_dict(orient="records"),
        }
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
    else:
        raise SplitInvariantError(f"Split manifest must be .json, .csv, or .parquet: {path}")


def validate_manifest_matches_data(manifest: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Require a one-to-one sample match before reusing an existing manifest."""

    if "sample_id" not in manifest.columns or "sample_id" not in metadata.columns:
        raise SplitInvariantError("Both data and manifest require sample_id.")
    if manifest["sample_id"].duplicated().any():
        raise SplitInvariantError("Split manifest contains duplicate sample_id values.")
    manifest_ids = set(manifest["sample_id"].astype(str))
    data_ids = set(metadata["sample_id"].astype(str))
    missing = sorted(data_ids.difference(manifest_ids))
    extra = sorted(manifest_ids.difference(data_ids))
    if missing or extra:
        raise SplitInvariantError(
            "Split manifest does not match the supplied dataset: "
            f"{len(missing)} data IDs absent from manifest, {len(extra)} extra manifest IDs."
        )
    for column in ("spatial_block_id", "date"):
        if column in manifest.columns and column in metadata.columns:
            manifest_copy = manifest.loc[:, ["sample_id", column]].copy()
            data_copy = metadata.loc[:, ["sample_id", column]].copy()
            manifest_copy["sample_id"] = manifest_copy["sample_id"].astype(str)
            data_copy["sample_id"] = data_copy["sample_id"].astype(str)
            left = manifest_copy.set_index("sample_id")[column].astype(str).sort_index()
            right = data_copy.set_index("sample_id")[column].astype(str).sort_index()
            if not left.equals(right):
                raise SplitInvariantError(
                    f"Existing manifest {column} values differ from the supplied dataset."
                )


def _align_manifest(manifest: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return manifest rows in metadata order after one-to-one ID validation."""

    validate_manifest_matches_data(manifest, metadata)
    lookup = manifest.copy()
    lookup["sample_id"] = lookup["sample_id"].astype(str)
    lookup = lookup.set_index("sample_id", drop=False)
    order = metadata["sample_id"].astype(str)
    return lookup.loc[order].reset_index(drop=True)


def manifest_outer_folds(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    split_config: Mapping[str, Any],
) -> Iterator[BlockedFold]:
    """Reconstruct frozen outer folds and their embargo rows from a manifest."""

    aligned = _align_manifest(manifest, metadata)
    required = {"split_role", "outer_fold", "spatial_block_id", "date"}
    missing = sorted(required.difference(aligned.columns))
    if missing:
        raise SplitInvariantError(f"Split manifest is missing required columns: {missing}")
    working = metadata.reset_index(drop=True).copy()
    for column in ("spatial_block_id", "date", "weather_event_id", "site_id"):
        if column in aligned.columns:
            working[column] = aligned[column].to_numpy()
    roles = aligned["split_role"].astype("string")
    fold_numbers = pd.to_numeric(aligned["outer_fold"], errors="coerce")
    if fold_numbers.isna().any() or not bool(
        np.isclose(fold_numbers, np.round(fold_numbers)).all()
    ):
        raise SplitInvariantError("Manifest outer_fold values must be whole integers.")
    fold_numbers = fold_numbers.astype(int)
    expected_folds = int(split_config.get("outer_folds", int(fold_numbers.max()) + 1))
    development = roles.eq("development").to_numpy()
    positions = np.arange(len(working), dtype=int)
    block_values = working["spatial_block_id"].astype(str).to_numpy()
    date_values = working["date"].astype(str).to_numpy()
    event_values = (
        working["weather_event_id"].astype("string").to_numpy()
        if "weather_event_id" in working.columns
        else None
    )
    embargo_config = split_config.get("embargo", {})
    embargo_days = (
        int(embargo_config.get("days", embargo_config.get("temporal_days", 0)))
        if isinstance(embargo_config, Mapping)
        else 0
    )
    parsed_dates = pd.to_datetime(date_values, errors="raise")
    for fold in range(expected_folds):
        validation_mask = development & fold_numbers.eq(fold).to_numpy()
        if not validation_mask.any():
            raise SplitInvariantError(f"Manifest has no development validation rows for fold {fold}.")
        validation_blocks = set(block_values[validation_mask])
        validation_dates = set(date_values[validation_mask])
        validation_events: set[str] = set()
        if event_values is not None:
            validation_events = {
                str(value)
                for value in event_values[validation_mask]
                if pd.notna(value) and str(value).strip()
            }
        restricted = np.isin(block_values, list(validation_blocks)) | np.isin(
            date_values, list(validation_dates)
        )
        if validation_events and event_values is not None:
            restricted |= np.isin(event_values.astype(str), list(validation_events))
        if embargo_days:
            for held_date in pd.to_datetime(sorted(validation_dates)):
                restricted |= np.abs((parsed_dates - held_date).days) <= embargo_days
        train_mask = development & ~restricted
        embargo_mask = development & ~validation_mask & ~train_mask
        train = positions[train_mask]
        validation = positions[validation_mask]
        embargo = positions[embargo_mask]
        assert_split_invariants(
            working,
            train,
            validation,
            embargo=embargo,
            event_column="weather_event_id" if event_values is not None else None,
        )
        yield BlockedFold(
            fold=fold,
            train=train,
            validation=validation,
            embargo=embargo,
            validation_spatial_blocks=tuple(sorted(validation_blocks)),
            validation_dates=tuple(sorted(validation_dates)),
            validation_events=tuple(sorted(validation_events)),
        )


def validate_split_manifest(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    split_config: Mapping[str, Any],
    *,
    target_column: str = "calculated_utci_c",
) -> None:
    """Assert role, locked-final, and frozen outer-fold invariants before use."""

    aligned = _align_manifest(manifest, metadata)
    required = {"split_role", "spatial_block_id", "date", "site_id", "outer_fold"}
    missing = sorted(required.difference(aligned.columns))
    if missing:
        raise SplitInvariantError(f"Split manifest is missing required columns: {missing}")
    roles = aligned["split_role"].astype("string")
    allowed_roles = {"development", "calibration", "final_test", "embargo"}
    invalid_roles = sorted(set(roles.dropna()).difference(allowed_roles))
    if invalid_roles or roles.isna().any():
        raise SplitInvariantError(
            f"Manifest split_role values must be {sorted(allowed_roles)}; invalid={invalid_roles}."
        )
    working = metadata.reset_index(drop=True).copy()
    for column in ("spatial_block_id", "date", "weather_event_id", "site_id"):
        if column in aligned.columns:
            working[column] = aligned[column].to_numpy()
    assert_partition_role_invariants(working, roles)
    final_config = split_config.get("final_test", {})
    assert_final_test_design(
        working,
        roles,
        final_config if isinstance(final_config, Mapping) else {},
        target_column=target_column,
    )
    # Iteration itself verifies every outer fold's dual blocking and embargo.
    tuple(manifest_outer_folds(working, aligned, split_config))


def read_or_create_split_manifest(
    metadata: pd.DataFrame,
    manifest_path: str | Path,
    split_config: Mapping[str, Any],
    *,
    create_if_missing: bool = False,
    overwrite: bool = False,
    source_data_path: str | Path | None = None,
    target_column: str = "calculated_utci_c",
) -> pd.DataFrame:
    """Read a matching manifest or explicitly create one during ``make-splits``."""

    path = Path(manifest_path).expanduser()
    if source_data_path is not None:
        source_path = Path(source_data_path).expanduser()
        if path.resolve(strict=False) == source_path.resolve(strict=False):
            raise SplitInvariantError(
                "The split manifest path is the same as the source --data path. Choose a "
                "separate manifest destination; source observations are never overwritten."
            )
    if path.exists() and not overwrite:
        manifest = _read_manifest(path)
        validate_split_manifest(
            manifest, metadata, split_config, target_column=target_column
        )
        return manifest
    if not path.exists() and not create_if_missing and not overwrite:
        raise SplitInvariantError(
            f"Split manifest does not exist: {path}. Run make-splits intentionally; training "
            "will not create or replace a manifest automatically."
        )
    manifest = create_split_manifest(metadata, split_config, target_column=target_column)
    validate_split_manifest(
        manifest, metadata, split_config, target_column=target_column
    )
    _write_manifest(manifest, path)
    return manifest
