"""Pure metadata/scalar split tests; no mock observation table is constructed."""

from __future__ import annotations

import pytest

from urban_heat_risk_ai.errors import SplitInvariantError
from urban_heat_risk_ai.splits import (
    assert_disjoint_spatiotemporal_groups,
    balanced_group_assignment,
    construct_spatial_block_ids,
)


def test_group_assignment_is_deterministic_and_keeps_groups_whole() -> None:
    groups = ["a", "a", "b", "c", "d", "e"]
    first = balanced_group_assignment(groups, n_folds=2, seed=7, salt="space")
    second = balanced_group_assignment(groups, n_folds=2, seed=7, salt="space")
    assert first == second
    assert set(first) == set(groups)
    assert set(first.values()) == {0, 1}


def test_dual_group_invariant_rejects_shared_space_or_date() -> None:
    assert_disjoint_spatiotemporal_groups(["b1"], ["2026-07-01"], ["b2"], ["2026-07-02"])
    with pytest.raises(SplitInvariantError, match="spatial block"):
        assert_disjoint_spatiotemporal_groups(
            ["b1"], ["2026-07-01"], ["b1"], ["2026-07-02"]
        )
    with pytest.raises(SplitInvariantError, match="complete date"):
        assert_disjoint_spatiotemporal_groups(
            ["b1"], ["2026-07-01"], ["b2"], ["2026-07-01"]
        )


def test_coordinate_projection_produces_deterministic_candidate_block_id() -> None:
    first = construct_spatial_block_ids(
        [-122.4194],
        [37.7749],
        projected_crs="EPSG:6933",
        block_size_m=500.0,
    )
    second = construct_spatial_block_ids(
        [-122.4194],
        [37.7749],
        projected_crs="EPSG:6933",
        block_size_m=500.0,
    )
    assert first.tolist() == second.tolist()
    assert first[0].startswith("EPSG6933_x")


def test_geographic_crs_is_rejected_for_meter_sized_blocks() -> None:
    with pytest.raises(SplitInvariantError, match="not projected"):
        construct_spatial_block_ids(
            [-122.4194], [37.7749], projected_crs="EPSG:4326", block_size_m=500.0
        )

