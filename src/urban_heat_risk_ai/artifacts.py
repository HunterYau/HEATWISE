"""Runtime-only artifact persistence and reproducibility hashes.

Nothing in this module creates a directory or file until an explicit write method
is called by a data-dependent CLI command.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from urban_heat_risk_ai.errors import ArtifactIntegrityError

HASH_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise ArtifactIntegrityError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical(value: Any) -> str:
    """Hash a JSON-compatible value using deterministic serialization."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenHashes:
    """Inputs that must be unchanged before the locked final test is opened."""

    dataset_sha256: str
    model_config_sha256: str
    feature_allowlist_sha256: str
    split_manifest_sha256: str

    @classmethod
    def from_files(
        cls,
        *,
        dataset: str | Path,
        model_config: str | Path,
        feature_allowlist: str | Path,
        split_manifest: str | Path,
    ) -> FrozenHashes:
        return cls(
            dataset_sha256=sha256_file(dataset),
            model_config_sha256=sha256_file(model_config),
            feature_allowlist_sha256=sha256_file(feature_allowlist),
            split_manifest_sha256=sha256_file(split_manifest),
        )

    def compare(self, expected: FrozenHashes) -> dict[str, tuple[str, str]]:
        """Return mismatches as ``field -> (expected, observed)``."""

        observed_map = asdict(self)
        expected_map = asdict(expected)
        return {
            key: (expected_map[key], observed_map[key])
            for key in expected_map
            if expected_map[key] != observed_map[key]
        }


@dataclass(frozen=True)
class FinalTestLock:
    """Serializable final-test lock created when the training run is frozen."""

    hashes: FrozenHashes
    created_at_utc: str
    run_id: str

    @classmethod
    def create(cls, hashes: FrozenHashes, run_id: str) -> FinalTestLock:
        return cls(
            hashes=hashes,
            created_at_utc=datetime.now(UTC).isoformat(),
            run_id=run_id,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> FinalTestLock:
        source = Path(path)
        if not source.is_file():
            raise ArtifactIntegrityError(
                f"Final-test lock is missing: {source}. Train and freeze a run first."
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            hashes=FrozenHashes(**payload["hashes"]),
            created_at_utc=str(payload["created_at_utc"]),
            run_id=str(payload["run_id"]),
        )


def authorize_final_test(
    *,
    unlock_requested: bool,
    current_hashes: FrozenHashes,
    lock: FinalTestLock,
) -> None:
    """Require the explicit one-way intent flag and exact frozen input hashes."""

    if not unlock_requested:
        raise ArtifactIntegrityError(
            "The final test is locked. Re-run with --unlock-final-test only after "
            "all model and analysis decisions are frozen."
        )
    mismatches = current_hashes.compare(lock.hashes)
    if mismatches:
        details = "; ".join(
            f"{field}: expected {expected}, observed {observed}"
            for field, (expected, observed) in mismatches.items()
        )
        raise ArtifactIntegrityError(
            "Frozen inputs changed; final testing is refused. " + details
        )


def software_versions(packages: Sequence[str] | None = None) -> dict[str, str]:
    """Capture Python/platform and installed distribution versions."""

    names = packages or (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "optuna",
        "pythermalcomfort",
        "shap",
        "matplotlib",
        "seaborn",
        "PyYAML",
        "joblib",
        "pyarrow",
        "pyproj",
        "torch",
    )
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


class ArtifactStore:
    """Explicit writer for one runtime experiment directory.

    Constructing the class is side-effect free. Call :meth:`initialize` only from
    a command that the user intentionally ran with a real dataset.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._initialized = False

    def initialize(self) -> ArtifactStore:
        self.root.mkdir(parents=True, exist_ok=False)
        self._initialized = True
        return self

    def open_existing(self) -> ArtifactStore:
        """Attach to an existing run directory for an explicit later-stage command."""

        if not self.root.is_dir():
            raise ArtifactIntegrityError(f"Run directory does not exist: {self.root}")
        self._initialized = True
        return self

    def _path(self, relative: str | Path) -> Path:
        if not self._initialized:
            raise ArtifactIntegrityError(
                "ArtifactStore has not been initialized by a runtime command."
            )
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ArtifactIntegrityError("Artifact paths must be relative to the run directory.")
        path = self.root / relative_path
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ArtifactIntegrityError(
                f"Artifact path escapes the run directory: {relative_path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self._path(relative)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def write_bytes(self, relative: str | Path, value: bytes) -> Path:
        """Write an already frozen binary snapshot into the artifact tree."""

        path = self._path(relative)
        path.write_bytes(value)
        return path

    def write_dataframe(self, relative: str | Path, frame: pd.DataFrame) -> Path:
        """Write CSV or Parquet according to the explicit filename suffix."""

        path = self._path(relative)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame.to_csv(path, index=False)
        elif suffix in {".parquet", ".pq"}:
            frame.to_parquet(path, index=False)
        else:
            raise ArtifactIntegrityError(
                f"Prediction/table artifact must end in .csv or .parquet: {path}"
            )
        return path

    def write_joblib(self, relative: str | Path, value: Any) -> Path:
        path = self._path(relative)
        joblib.dump(value, path)
        return path

    def copy_file(self, source: str | Path, relative: str | Path) -> Path:
        destination = self._path(relative)
        shutil.copy2(Path(source), destination)
        return destination

    def freeze_final_test(
        self, lock: FinalTestLock, relative: str | Path = "final_test.lock.json"
    ) -> Path:
        payload = {
            "hashes": asdict(lock.hashes),
            "created_at_utc": lock.created_at_utc,
            "run_id": lock.run_id,
        }
        return self.write_json(relative, payload)


def experiment_metadata(
    *,
    command: str,
    run_id: str,
    seed: int,
    hashes: FrozenHashes,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transparent, serializable runtime metadata record."""

    return {
        "command": command,
        "run_id": run_id,
        "seed": int(seed),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hashes": asdict(hashes),
        "software_versions": software_versions(),
        "extra": dict(extra or {}),
    }
