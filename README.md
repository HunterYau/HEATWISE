# Urban Heat Risk AI

`urban_heat_risk_ai` is a Python 3.12 framework for predicting pedestrian-level Universal
Thermal Climate Index (UTCI) at unsensed urban locations. The continuous target is
`calculated_utci_c`. The framework is deliberately strict about real-data provenance,
spatiotemporal leakage, final-test locking, and fair comparison of operational models.

This repository contains source code, configuration, documentation, and data-independent
tests only. It contains no dataset, trained model, predictions, metrics, plot, or claimed
performance result. Runtime directories are created only after a user supplies a real CSV
or Parquet table and explicitly chooses an output path for a writing command.

## Environment

Use the existing project virtual environment. Do not create, replace, move, rename, repair,
or manually modify `.venv`. Install pinned dependencies only through its interpreter:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency set is pinned for Python 3.12. CPU is the default for XGBoost and the optional
PyTorch neural-network baseline. PyTorch is included because that baseline uses dropout,
AdamW, and validation-based early stopping; the baseline is disabled by default in
`configs/model.yaml`.

For a local editable package install, after dependencies are present:

```powershell
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The module entry point works directly from an editable install:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --help
```

## First command with real data

After placing a real training table anywhere on the computer, run this exact command from
the project root, replacing only the quoted path:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai validate --data "C:\absolute\path\to\real_training_table.parquet"
```

CSV is also accepted. Every observation-dependent command requires an explicit `--data`
path. The application does not search the project, home directory, mounted drives, or any
other location for data. A missing path produces a clear error and no training starts.

Validation is read-only: it reports column, dtype, unit, range, timestamp, duplicate-ID,
missingness, split-role, feature-availability, and leakage problems without cleaning or
rewriting the source table. Read [the schema](docs/DATA_SCHEMA.md) and complete [the real-data
checklist](docs/REAL_DATA_CHECKLIST.md) before making a split manifest.

## CLI workflow

Global options such as `--config` and `--features-config` go before the subcommand. Defaults
are `configs/model.yaml` and `configs/features.yaml`.

```text
validate     Validate a real CSV/Parquet table without modifying it.
make-splits  Create a deterministic spatiotemporal split manifest at an explicit path.
tune         Run blocked inner-fold Optuna tuning on development observations.
train        Refit frozen direct/component models using selected settings.
calibrate    Fit 90% split-conformal intervals on the untouched calibration role.
evaluate     Evaluate stored development out-of-fold predictions and baselines.
final-test   Evaluate the hash-locked final role only with --unlock-final-test.
predict      Predict from an explicit operational feature table to an explicit destination.
```

Get command-specific arguments without supplying data:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai make-splits --help
.venv\Scripts\python.exe -m urban_heat_risk_ai tune --help
.venv\Scripts\python.exe -m urban_heat_risk_ai train --help
```

A typical real-data sequence is documented with full commands in
`docs/REAL_DATA_CHECKLIST.md`. No command should be run merely to produce a demonstration;
use only reviewed real observations.

## Modeling contract

The preregistered primary model is an unweighted `XGBRegressor` that directly predicts
continuous UTCI from the `core` allow-list. It uses histogram trees, MAE evaluation,
deterministic seeds, CPU execution, nested spatiotemporal blocked validation, and inner-fold
early stopping. The outer evaluation fold never controls preprocessing, hyperparameters, or
tree count.

The core model does not require a same-day thermal image. A separately trained
`satellite_enhanced` model adds quality-controlled Landsat or ECOSTRESS LST variables only
on rows satisfying frozen image-age/quality rules. Its comparison core model is fit and
evaluated on exactly the same eligible observations.

A central leakage guard terminates fitting if an undeclared, banned, or suspicious column
enters a model. Prohibited inputs include local measured air/RH/wind/globe temperature,
MRT, UTCI/category, WBGT, uncertainty, transformations of those values, IDs, calibration and
quality metadata, coordinates, timestamps, and split fields. Coordinates are available to
spatial splitting and mapping only.

Four secondary component models predict local air departure, local vapor pressure,
pedestrian-wind log adjustment, and MRT departure. Their reconstructed meteorology passes
through a pinned `pythermalcomfort.models.utci` SI wrapper. Raw physical predictions and
validity flags are retained. Direct and component UTCI are compared on identical held-out
observations and are never silently averaged.

Fixed heat-focused categories are:

- below 26 °C: no heat stress;
- 26 to below 32 °C: moderate;
- 32 to below 38 °C: strong;
- 38 to below 46 °C: very strong; and
- 46 °C or higher: extreme.

See [the methodology](docs/METHODOLOGY.md) for split design, tuning/refit rules, wind-height
conversion, component reconstruction, baselines, block bootstrap, conformal intervals,
subgroups, final-test hashing, and interpretation constraints.

## Project layout

```text
configs/
  features.yaml          Predictor allow-lists and leakage deny-list
  model.yaml             Split, model, physics, evaluation, and artifact policy
docs/
  DATA_SCHEMA.md         Strict real-table columns, units, ranges, and validation rules
  METHODOLOGY.md          Prespecified modeling and evaluation design
  REAL_DATA_CHECKLIST.md Real-data provenance and run checklist
src/urban_heat_risk_ai/
  artifacts.py           Runtime-only artifact metadata and hashing
  baselines.py           Weather, Heat Index, LST, linear, RF, and neural comparisons
  cli.py                 Explicit-path command-line interface
  components.py          Component targets, fitting, and reconstruction
  conformal.py           Split-conformal intervals and coverage diagnostics
  direct_xgb.py          Nested tuning and outer refit for direct XGBoost
  explain.py             SHAP, PDP, extrapolation, and anomaly diagnostics
  features.py            Allow-list preprocessing and central leakage guard
  metrics.py             Continuous/category/subgroup/block-bootstrap metrics
  physics.py             Humidity, wind-height, Heat Index, and UTCI physics
  schema.py              Immutable schema validation and reports
  splits.py              SpatioTemporalBlockedSplit and manifest invariants
  workflows.py           CLI workflow orchestration
tests/                    Pure, data-independent tests only
```

## Development checks

These checks do not train a model or create data/results:

```powershell
.venv\Scripts\python.exe -m compileall -q src tests
.venv\Scripts\python.exe -m urban_heat_risk_ai --help
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests
```

Tests use isolated scalar constants only for pure functions and configuration contracts.
They must not create a synthetic table, train an estimator, or write predictions, models,
plots, metrics, or other generated data artifacts.
