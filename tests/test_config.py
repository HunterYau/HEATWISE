"""Configuration tests that do not create observation or artifact files."""

from __future__ import annotations

from urban_heat_risk_ai.config import (
    canonical_config_hash,
    deep_merge,
    dotted_get,
    load_project_config,
)


def test_checked_in_configuration_contract_loads() -> None:
    config = load_project_config()
    assert config.model["project"]["name"] == "urban_heat_risk_ai"
    assert config.features["target_column"] == "calculated_utci_c"
    assert config.seed == 20260720
    assert len(config.digest) == 64


def test_deep_merge_is_non_mutating_and_dotted_get_reads_nested_value() -> None:
    base = {"outer": {"enabled": False, "count": 2}}
    merged = deep_merge(base, {"outer": {"enabled": True}})
    assert base["outer"]["enabled"] is False
    assert dotted_get(merged, "outer.enabled") is True
    assert dotted_get(merged, "outer.count") == 2


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    first = {"a": 1, "b": {"x": 2}}
    second = {"b": {"x": 2}, "a": 1}
    assert canonical_config_hash(first) == canonical_config_hash(second)

