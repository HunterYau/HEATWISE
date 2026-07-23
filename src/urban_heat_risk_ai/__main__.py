"""Module entry point for ``python -m urban_heat_risk_ai``."""

from __future__ import annotations

from urban_heat_risk_ai.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation
    raise SystemExit(main())
