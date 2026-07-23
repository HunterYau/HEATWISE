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

