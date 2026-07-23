"""Scalar-only checks for the opt-in heat-weighting policy."""

from __future__ import annotations

import numpy as np

from urban_heat_risk_ai.direct_xgb import (
    piecewise_heat_weights,
    select_refit_tree_count,
)


def test_piecewise_heat_weights_follow_fixed_threshold_tiers() -> None:
    weights = piecewise_heat_weights(
        [31.9, 32.0, 37.9, 38.0, 46.0],
        thresholds_c=[32.0, 38.0, 46.0],
        weights=[1.0, 1.5, 2.0, 3.0],
    )
    assert np.array_equal(weights, np.asarray([1.0, 1.5, 1.5, 2.0, 3.0]))


def test_refit_tree_count_uses_rounded_median_and_configured_bounds() -> None:
    assert (
        select_refit_tree_count(
            [120, 200, 250], minimum_trees=300, maximum_trees=1500
        )
        == 300
    )
    assert (
        select_refit_tree_count(
            [501, 500], minimum_trees=300, maximum_trees=1500
        )
        == 501
    )
    assert (
        select_refit_tree_count(
            [1600, 1700], minimum_trees=300, maximum_trees=1500
        )
        == 1500
    )
