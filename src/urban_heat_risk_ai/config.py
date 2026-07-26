"""Strict, side-effect-free YAML configuration loading.

Configuration is deliberately represented as ordinary mappings so experiment
snapshots can be serialized without custom encoders.  This module validates the
small set of cross-file contracts needed before any observation data are read;
model-specific validation remains next to the model implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = PACKAGE_ROOT / "configs" / "model.yaml"
DEFAULT_FEATURE_CONFIG = PACKAGE_ROOT / "configs" / "features.yaml"


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be a YAML mapping, not {type(value).__name__}.")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{context} contains a non-string key: {key!r}.")
        result[key] = item
    return result


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping without mutating it or resolving arbitrary tags.

    The path is always explicit; this function never searches parent folders or
    the user's computer.  ``yaml.safe_load`` prevents executable YAML tags.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}. "
            "Pass the path to an existing .yaml or .yml file."
        )
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(f"Configuration must be YAML (.yaml or .yml): {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration {config_path}: {exc}") from exc
    if loaded is None:
        raise ConfigurationError(f"Configuration file is empty: {config_path}")
    return _require_mapping(loaded, context=str(config_path))


# A concise alias is convenient in CLI and tests.
load_config = load_yaml_config


def deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any], *, reject_unknown: bool = True
) -> dict[str, Any]:
    """Recursively apply explicit overrides to a copy of ``base``.

    Rejecting unknown keys by default prevents misspelled experimental settings
    from being silently ignored.  The input mappings are never modified.
    """

    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if reject_unknown and key not in merged:
            raise ConfigurationError(f"Unknown configuration override key: {key}")
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(
                _require_mapping(merged[key], context=f"base.{key}"),
                _require_mapping(value, context=f"override.{key}"),
                reject_unknown=reject_unknown,
            )
        else:
            merged[key] = deepcopy(value)
    return merged


def dotted_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested value such as ``"splits.spatial.block_size_m"``."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def require_keys(config: Mapping[str, Any], keys: tuple[str, ...], *, context: str) -> None:
    """Raise an actionable error when required mapping keys are absent."""

    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigurationError(f"{context} is missing required key(s): {', '.join(missing)}")


def canonical_config_hash(*configs: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for already-loaded configuration mappings."""

    payload = json.dumps(configs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProjectConfig:
    """Validated model and predictor configuration bundle."""

    model: dict[str, Any]
    features: dict[str, Any]
    model_path: Path
    features_path: Path

    @property
    def seed(self) -> int:
        project = self.model.get("project", {})
        if isinstance(project, Mapping) and "seed" in project:
            return int(project["seed"])
        return int(self.model.get("seed", 0))

    @property
    def digest(self) -> str:
        return canonical_config_hash(self.model, self.features)


def _validate_project_contract(model: Mapping[str, Any], features: Mapping[str, Any]) -> None:
    require_keys(
        model,
        ("project", "data", "splits", "preprocessing", "direct_xgb", "satellite"),
        context="model configuration",
    )
    target = model.get("data", {}).get("target_column", model.get("data", {}).get("target"))
    feature_target = features.get("target_column", features.get("target"))
    if not isinstance(target, str) or not target:
        raise ConfigurationError("model.data.target_column must be a non-empty string.")
    if not isinstance(feature_target, str) or not feature_target:
        raise ConfigurationError("features.target_column must be a non-empty string.")
    if target != feature_target:
        raise ConfigurationError(
            f"Target mismatch: model declares {target!r}, features declare {feature_target!r}."
        )

    staged = model.get("training_stages")
    if staged is not None:
        staged_config = _require_mapping(staged, context="model.training_stages")
        stage1 = _require_mapping(
            staged_config.get("stage1"), context="model.training_stages.stage1"
        )
        stage2 = _require_mapping(
            staged_config.get("stage2"), context="model.training_stages.stage2"
        )
        stage1_target = stage1.get("target_column")
        stage2_target = stage2.get("target_column")
        if not isinstance(stage1_target, str) or not stage1_target:
            raise ConfigurationError(
                "training_stages.stage1.target_column must be a non-empty string."
            )
        if not isinstance(stage2_target, str) or not stage2_target:
            raise ConfigurationError(
                "training_stages.stage2.target_column must be a non-empty string."
            )
        if stage1_target == stage2_target:
            raise ConfigurationError(
                "Stage 1 public-reference and Stage 2 sensor-derived targets must be distinct."
            )
        if stage1_target != "public_reference_utci_c":
            raise ConfigurationError(
                "training_stages.stage1.target_column must be public_reference_utci_c."
            )
        expected_public_provenance = (
            "public_source_name",
            "public_source_version",
            "public_source_license",
            "public_retrieved_at_utc",
            "public_target_method_version",
            "public_quality_flag",
        )
        if tuple(stage1.get("required_public_provenance", ())) != expected_public_provenance:
            raise ConfigurationError(
                "training_stages.stage1.required_public_provenance must match "
                "the frozen public-source contract."
            )
        if stage1.get("target_provenance") != "public_online_only":
            raise ConfigurationError(
                "Stage 1 target provenance must be public_online_only."
            )
        if stage2_target != target:
            raise ConfigurationError(
                "training_stages.stage2.target_column must match data.target_column."
            )
        if stage2_target != "calculated_utci_c":
            raise ConfigurationError(
                "training_stages.stage2.target_column must be calculated_utci_c."
            )
        if (
            stage2.get("target_provenance")
            != "derived_in_memory_from_raw_sensor_measurements"
        ):
            raise ConfigurationError(
                "Stage 2 target must be derived in memory from raw sensor measurements."
            )
        if stage2.get("requires_stage1_bundle") is not True:
            raise ConfigurationError("Stage 2 must require a completed Stage 1 bundle.")
        feature_stage_targets = features.get("training_stage_targets", {})
        if not isinstance(feature_stage_targets, Mapping):
            raise ConfigurationError("features.training_stage_targets must be a mapping.")
        if feature_stage_targets.get("stage1_public") != stage1_target:
            raise ConfigurationError(
                "Stage 1 target differs between model.yaml and features.yaml."
            )
        if feature_stage_targets.get("stage2_sensor") != stage2_target:
            raise ConfigurationError(
                "Stage 2 target differs between model.yaml and features.yaml."
            )
        if feature_stage_targets.get("predictors_shared_exactly_between_stages") is not True:
            raise ConfigurationError(
                "The feature contract must require predictors_shared_exactly_between_stages."
            )
        if staged_config.get("predictor_set", "core") != "core":
            raise ConfigurationError(
                "The two-stage lineage currently requires the public, non-thermal core "
                "predictor set."
            )
        if not bool(stage2.get("reuse_stage1_preprocessor_without_refit", False)):
            raise ConfigurationError(
                "Stage 2 must reuse the frozen Stage 1 preprocessor without refitting."
            )
        if not bool(stage2.get("require_exact_stage1_input_schema", False)):
            raise ConfigurationError("Stage 2 must require the exact Stage 1 input schema.")
        if not bool(stage2.get("preserve_stage1_artifacts", False)):
            raise ConfigurationError("Stage 2 must preserve the Stage 1 artifacts.")
        common = _require_mapping(
            staged_config.get("common"), context="model.training_stages.common"
        )
        if not bool(common.get("prohibit_sensor_inputs", False)):
            raise ConfigurationError(
                "The staged workflow must prohibit sensor inputs."
            )
        if not bool(common.get("require_exact_ordered_input_schema", False)):
            raise ConfigurationError(
                "The staged workflow must require one exact ordered input schema."
            )
        if common.get("tree_method") != "hist" or common.get("device") != "cpu":
            raise ConfigurationError(
                "The staged XGBoost workflow must default to CPU histogram trees."
            )

    split_config = _require_mapping(model["splits"], context="model.splits")
    spatial = _require_mapping(split_config.get("spatial"), context="model.splits.spatial")
    projected_crs = spatial.get("projected_crs", spatial.get("crs"))
    block_size = spatial.get("block_size_m")
    if not isinstance(projected_crs, str) or not projected_crs:
        raise ConfigurationError("splits.spatial.projected_crs must be configured explicitly.")
    if not isinstance(block_size, (int, float)) or float(block_size) <= 0:
        raise ConfigurationError("splits.spatial.block_size_m must be a positive number.")

    project = _require_mapping(model["project"], context="model.project")
    seed = project.get("seed", model.get("seed"))
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigurationError("project.seed must be a non-negative integer.")


def load_project_config(
    model_path: str | Path = DEFAULT_MODEL_CONFIG,
    features_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """Load and cross-check the project's model and feature YAML files.

    If ``features_path`` is omitted, ``paths.features_config`` is resolved against
    the model configuration's project root.  No output directory is created.
    """

    resolved_model_path = Path(model_path).expanduser().resolve()
    model = load_yaml_config(resolved_model_path)
    if overrides:
        model = deep_merge(model, overrides)

    if features_path is None:
        configured = dotted_get(model, "paths.features_config", "configs/features.yaml")
        configured_path = Path(str(configured))
        if configured_path.is_absolute():
            resolved_features_path = configured_path
        else:
            # The default file lives in <root>/configs/model.yaml.  A custom model
            # file resolves project-relative paths from its parent unless that
            # parent itself is named "configs".
            base = (
                resolved_model_path.parent.parent
                if resolved_model_path.parent.name == "configs"
                else resolved_model_path.parent
            )
            resolved_features_path = base / configured_path
    else:
        resolved_features_path = Path(features_path).expanduser()
    resolved_features_path = resolved_features_path.resolve()
    features = load_yaml_config(resolved_features_path)
    _validate_project_contract(model, features)
    return ProjectConfig(
        model=model,
        features=features,
        model_path=resolved_model_path,
        features_path=resolved_features_path,
    )


def mutable_copy(config: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Return a deep mutable copy for callers that need runtime-only annotations."""

    return deepcopy(dict(config))
