"""Human-centered urban heat-risk modelling framework.

The package intentionally performs no filesystem discovery, training, or artifact
creation at import time.  Use :mod:`urban_heat_risk_ai.cli` with an explicit real
observation table when data become available.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0"
