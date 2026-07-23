"""Project-specific exceptions with actionable command-line messages."""

from __future__ import annotations


class UrbanHeatRiskError(RuntimeError):
    """Base class for expected, user-actionable project errors."""


class DataRequiredError(UrbanHeatRiskError):
    """Raised when an observation-requiring command has no readable input file."""


class ConfigurationError(UrbanHeatRiskError):
    """Raised for malformed or internally inconsistent configuration."""


class SchemaValidationError(UrbanHeatRiskError):
    """Raised when a real table fails strict schema validation."""


class LeakageError(UrbanHeatRiskError):
    """Raised when a predictor is banned, suspicious, or undeclared."""


class SplitInvariantError(UrbanHeatRiskError):
    """Raised when a split leaks space, time, or locked-test information."""


class ArtifactIntegrityError(UrbanHeatRiskError):
    """Raised when frozen hashes or artifact metadata do not match."""
