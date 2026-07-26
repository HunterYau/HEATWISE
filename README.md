# Urban Heat Risk AI

`urban_heat_risk_ai` is a Python 3.12 framework for predicting pedestrian-level Universal
Thermal Climate Index (UTCI) at unsensed urban locations. Its staged path separates public
reference learning from later local sensor adaptation:

1. **Stage 1 - public-reference initialization** fits a base XGBoost model to
   `public_reference_utci_c` using only reviewed public online data and the operational
   predictor allow-list. It contains no local sensor measurements.
2. **Stage 2 - local pedestrian adaptation** loads the immutable Stage 1 bundle, derives
   `calculated_utci_c` from collocated raw sensor measurements, and continues boosting on
   the same frozen online-input representation. Sensor values are label-production inputs
   only; they never enter the model matrix.

Stage 1 is an initialization from public reference conditions. It is **not** evidence of
local pedestrian-level accuracy and must not be presented as a substitute for Stage 2
validation on independently collected local observations.

The framework is deliberately strict about provenance, target leakage, frozen feature
contracts, spatiotemporal splitting, lineage hashes, final-test locking, and fair comparison
of operational models.

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

## Two-stage quick start

The framework never downloads public data, queries an API, searches the computer, or joins
sensor and online sources automatically. Prepare each reviewed CSV or Parquet table
yourself and pass its exact path.

Stage 1 consumes an already assembled public-reference table. Validate it first:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai stage1-validate --data "C:\absolute\path\to\public_reference_table.parquet"
```

Then explicitly train and save the base bundle:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai stage1-train --data "C:\absolute\path\to\public_reference_table.parquet" --output-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public"
```

Later, after collecting sensors, prepare one prejoined row-per-observation table containing
the exact same online predictors plus the required raw sensor and provenance fields.
Validate it against the saved Stage 1 contract:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\features.yaml" stage2-validate --data "C:\absolute\path\to\local_sensor_joined.parquet" --stage1-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public"
```

Adapt into a new directory; the Stage 1 directory remains unchanged:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\features.yaml" stage2-adapt --data "C:\absolute\path\to\local_sensor_joined.parquet" --stage1-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public" --output-dir "C:\absolute\path\to\heatwise_artifacts\stage2_local"
```

Predict from an online-input-only table with either bundle:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage2_local\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage2_local\configs\features.yaml" stage-predict --data "C:\absolute\path\to\online_only_prediction_rows.parquet" --model-dir "C:\absolute\path\to\heatwise_artifacts\stage2_local" --output "C:\absolute\path\to\heatwise_artifacts\predictions\local_utci.parquet"
```

Using each bundle's copied configuration snapshots makes later runs independent of
subsequent edits in the checkout. The command refuses a byte-different model configuration
or feature allow-list rather than silently adapting or predicting under a changed contract.

CSV is also accepted for input. Every observation-dependent command requires an explicit
`--data` path. A missing path produces a clear error and no training starts. Validation is
read-only: it reports column, dtype, unit, range, timestamp, duplicate-ID, missingness,
split-role, feature-availability, provenance, and leakage problems without cleaning or
rewriting the source table.

Global overrides precede the subcommand:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config configs\model.yaml --features-config configs\features.yaml stage1-validate --data "C:\absolute\path\to\public_reference_table.parquet"
```

Read [the schema](docs/DATA_SCHEMA.md), [the methodology](docs/METHODOLOGY.md), and
[the real-data checklist](docs/REAL_DATA_CHECKLIST.md) before training.

## Staged artifact contract

Stage commands create an output directory only when training or prediction is intentionally
run. A successful Stage 1 bundle has this layout:

```text
stage1_public/
  configs/
    model.yaml
    features.yaml
  models/
    stage1_base.joblib
  preprocessing/
    stage1_preprocessor.joblib
  schemas/
    input_features.json
  metadata/
    stage1.json
  hashes.json
  artifact_manifest.json
```

Stage 2 writes a different bundle and preserves the parent:

```text
stage2_local/
  configs/
    model.yaml
    features.yaml
  models/
    stage2_adapted.joblib
  preprocessing/
    frozen_stage1_preprocessor.joblib
  schemas/
    input_features.json
  metadata/
    stage2.json
  lineage/
    stage1_parent.json
  hashes.json
  artifact_manifest.json
```

`input_features.json` freezes raw predictor order, numeric/categorical roles, transformed
feature names and order, categorical handling, units, predictor-policy version, and hashes.
Stage 2 transforms with the saved Stage 1 preprocessor; it does not refit that preprocessor
or introduce a new predictor. `stage1_parent.json` records the immutable parent bundle and
its hashes. `artifact_manifest.json` and `hashes.json` make the model, preprocessor, schema,
configuration, input table, and lineage auditable.

## CLI workflow

Global options such as `--config` and `--features-config` go before the subcommand. Defaults
are `configs/model.yaml` and `configs/features.yaml`.

```text
stage1-validate Validate a public-only reference table without modifying it.
stage1-train    Train and save the public-reference base model and frozen input contract.
stage2-validate Validate a sensor-joined table against an immutable Stage 1 bundle.
stage2-adapt    Continue Stage 1 boosting on freshly derived local UTCI labels.
stage-predict   Predict with a Stage 1 or Stage 2 bundle from online-only inputs.

# Legacy single-run research workflow
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
.venv\Scripts\python.exe -m urban_heat_risk_ai stage1-train --help
.venv\Scripts\python.exe -m urban_heat_risk_ai stage2-adapt --help
.venv\Scripts\python.exe -m urban_heat_risk_ai stage-predict --help
.venv\Scripts\python.exe -m urban_heat_risk_ai make-splits --help
.venv\Scripts\python.exe -m urban_heat_risk_ai tune --help
.venv\Scripts\python.exe -m urban_heat_risk_ai train --help
```

The legacy commands remain available for the original single-table nested-CV research
workflow. They do not replace the staged lineage contract. A complete staged sequence and
the legacy workflow are documented in `docs/REAL_DATA_CHECKLIST.md`. No command should be
run merely to produce a demonstration; use only reviewed real observations.

## Modeling contract

Both stages use XGBoost with the same public operational-input contract. Stage 1 fits and
freezes the preprocessor plus base booster. Stage 2 verifies the parent artifact hashes,
derives local UTCI labels from sensors, transforms with the frozen Stage 1 preprocessor, and
continues boosting from the saved base booster. It saves a separate adapted artifact and
never mutates the parent. `stage-predict` accepts either complete bundle but never accepts
sensor measurements as model inputs.

In the legacy research workflow, the preregistered primary model is an unweighted
`XGBRegressor` that directly predicts continuous UTCI from the `core` allow-list. It uses
histogram trees, MAE evaluation, deterministic seeds, CPU execution, nested spatiotemporal
blocked validation, and inner-fold early stopping. The outer evaluation fold never controls
preprocessing, hyperparameters, or tree count.

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

- below 26 degrees C: no heat stress;
- 26 to below 32 degrees C: moderate;
- 32 to below 38 degrees C: strong;
- 38 to below 46 degrees C: very strong; and
- 46 degrees C or higher: extreme.

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
  staged_training.py     Frozen staged schemas, bundle integrity, and lineage
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
plots, metrics, or other generated data artifacts. Staged tests cover public-only Stage 1
column rejection, Stage 2 sensor exclusion and physics, frozen feature-schema compatibility,
lineage configuration and artifact contracts, and command parsing without performing a
training run.
