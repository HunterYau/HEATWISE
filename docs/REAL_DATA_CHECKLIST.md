# Real-data readiness checklist

Use this checklist only with reviewed real records: genuine public-source observations for
Stage 1 and actual local sensor observations for Stage 2. Do not create a mock, synthetic,
example, or filled-in training table to satisfy a check. If a required item is not
available, stop and document the gap.

## Two-stage lifecycle

The staged path is the recommended operational sequence. Stage 1 can run before local
sensors exist. Stage 2 is an independent later command that consumes, but never overwrites,
the frozen Stage 1 bundle.

The application does not download public data, call an online API, search the project or
computer, or join sensor records. Assemble reviewed tables outside the application and pass
their exact paths.

### A. Prepare the Stage 1 public-reference table

- [ ] Obtain the public sources under licenses that permit the intended use.
- [ ] Record provider, product/collection, immutable version, license, retrieval timestamp,
  spatial/temporal support, update cadence, and quality policy.
- [ ] Verify each row is one public site/grid-time observation with a unique `sample_id`.
- [ ] Include `public_source_name`, `public_source_version`, `public_source_license`,
  `public_retrieved_at_utc`, `public_target_method_version`, `public_quality_flag`, and
  continuous `public_reference_utci_c`.
- [ ] Document exactly how the public reference UTCI was produced, including units,
  implementation/version, wind-height convention, and input-limit behavior.
- [ ] Include the common site/time/block metadata and a reviewed `split_role`. Stage 1
  fitting uses only the configured training role and accepted public quality state.
- [ ] Include the ordered online predictors allowed by `configs/features.yaml`.
- [ ] Confirm every predictor is available from public online sources at an unsensed
  prediction location and time.
- [ ] Confirm the table has no `sensor_id`, measurement height, calibration/quality field
  for local instruments, `measured_*` value, local MRT/UTCI/category, WBGT, uncertainty,
  sensor-derived transformation, or suspicious alias.
- [ ] Archive the immutable table outside the model output directory and record its hash.

Stage 1 is public-reference initialization. Its validation or training does not establish
local pedestrian accuracy.

### B. Validate Stage 1 without writing artifacts

From the project root:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai stage1-validate --data "C:\absolute\path\to\public_reference_table.parquet"
```

- [ ] Review every dtype, unit, range, timestamp, missingness, duplicate-ID, split-role,
  public-provenance, feature-availability, and leakage finding.
- [ ] Correct the upstream table-generation process; the validator does not clean or rewrite
  the table.
- [ ] Confirm a missing path fails clearly and that no runtime directory is created.

CSV input is also supported. If using nondefault checked-in configuration files, global
options precede the command:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config configs\model.yaml --features-config configs\features.yaml stage1-validate --data "C:\absolute\path\to\public_reference_table.parquet"
```

### C. Train and freeze Stage 1

Choose a new explicit destination:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai stage1-train --data "C:\absolute\path\to\public_reference_table.parquet" --output-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public"
```

- [ ] Confirm the command used only accepted public rows in the configured training role.
- [ ] Confirm `public_reference_utci_c` was the target and never appeared in transformed
  feature names.
- [ ] Confirm no local sensor or sensor-derived column entered preprocessing or XGBoost.
- [ ] Freeze and archive the entire bundle; do not replace files in place.
- [ ] Treat Stage 1 predictions as public-reference initialization, not locally validated
  pedestrian predictions.

Expected Stage 1 bundle:

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

`schemas/input_features.json` must freeze the raw predictor order, numeric/categorical
roles, transformed feature names and order, categorical handling, units, policy version,
and hashes. Preserve this file with the model and preprocessor.

### D. Collect and assemble the Stage 2 sensor-joined table

- [ ] Collect real, independently reviewed local pedestrian measurements.
- [ ] Synchronize local air temperature, RH, pedestrian wind, and globe temperature to a
  documented tolerance and averaging period.
- [ ] Record sensor IDs, calibration versions, measurement heights, maintenance events,
  quality flags, and exclusions.
- [ ] Verify the globe inventory matches the configured `globe_diameter_m`,
  `globe_emissivity`, and calculation method. These are frozen physics settings, not model
  inputs.
- [ ] Prejoin exactly one row per observation using stable keys. Resolve one-to-many or
  ambiguous online-source matches before invoking the application.
- [ ] Supply the same online predictor names expected by the Stage 1 schema. Do not add a
  new local feature because it happens to be available near a sensor.
- [ ] Confirm online weather lags and rolling features are computed only from the
  operational background source, never from local sensors.
- [ ] Include raw `measured_air_temperature_c`, `measured_relative_humidity_pct`,
  `measured_pedestrian_wind_speed_m_s`, and `measured_globe_temperature_c` for target
  derivation only.
- [ ] Include the common spatial/time metadata, sensor provenance, and reviewed
  `split_role`; adaptation uses only configured development/quality-accepted rows.
- [ ] Do not use caller-provided MRT, UTCI, categories, or sensor transformations as the
  training target. The staged commands derive the target freshly.
- [ ] Archive and hash the joined table outside both model directories.

### E. Validate Stage 2 against the frozen parent

Use the configuration snapshots stored inside the Stage 1 bundle. This keeps adaptation
bound to the exact contract that trained the parent, even if the project checkout changes
later:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\features.yaml" stage2-validate --data "C:\absolute\path\to\local_sensor_joined.parquet" --stage1-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public"
```

- [ ] Confirm every Stage 1 artifact and manifest hash verifies before label derivation.
- [ ] Confirm raw predictor names, order, roles, units, categorical handling, transformed
  order, policy version, and hashes match `schemas/input_features.json`.
- [ ] Confirm the saved Stage 1 preprocessor is used only to transform Stage 2 predictors.
  It must not be fitted or extended.
- [ ] Review fresh MRT, wind-height, and UTCI applicability failures. Do not clip, fill, or
  silently exclude an invalid eligible target.
- [ ] Confirm raw sensor measurements and all target-production diagnostics are absent from
  the operational predictor matrix.

### F. Adapt into a separate Stage 2 bundle

Use a new output directory. It must not be the Stage 1 directory:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\features.yaml" stage2-adapt --data "C:\absolute\path\to\local_sensor_joined.parquet" --stage1-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public" --output-dir "C:\absolute\path\to\heatwise_artifacts\stage2_local"
```

- [ ] Confirm adaptation continued boosting from `models/stage1_base.joblib`; it was not a
  fresh unrelated fit.
- [ ] Confirm the Stage 1 model, preprocessor, schema, metadata, and hashes are byte-for-byte
  unchanged after adaptation.
- [ ] Confirm Stage 2 stored a copy of the frozen Stage 1 preprocessor/schema rather than a
  refitted preprocessing object.
- [ ] Confirm `lineage/stage1_parent.json` identifies and hashes the exact parent bundle.
- [ ] Confirm all local target derivation settings and software versions are recorded.

Expected Stage 2 bundle:

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

### G. Predict with either complete bundle

Prepare an online-only CSV or Parquet table matching the frozen input schema. Do not include
or expect local sensors at prediction locations.

With the locally adapted bundle:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage2_local\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage2_local\configs\features.yaml" stage-predict --data "C:\absolute\path\to\online_only_prediction_rows.parquet" --model-dir "C:\absolute\path\to\heatwise_artifacts\stage2_local" --output "C:\absolute\path\to\heatwise_artifacts\predictions\local_utci.parquet"
```

To intentionally use public-reference initialization before Stage 2 exists, point
`--model-dir` at the Stage 1 directory:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai --config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\model.yaml" --features-config "C:\absolute\path\to\heatwise_artifacts\stage1_public\configs\features.yaml" stage-predict --data "C:\absolute\path\to\online_only_prediction_rows.parquet" --model-dir "C:\absolute\path\to\heatwise_artifacts\stage1_public" --output "C:\absolute\path\to\heatwise_artifacts\predictions\public_reference_utci.parquet"
```

- [ ] Verify the model bundle and lineage hashes before accepting predictions.
- [ ] Preserve stable row identifiers in the output.
- [ ] Label Stage 1 output as public-reference initialization.
- [ ] Label Stage 2 output as locally adapted only within the documented population and
  support.
- [ ] Treat extrapolation and anomaly flags as diagnostics; do not claim causation.

## Legacy single-table research workflow

The checklist below preserves the original nested blocked-CV, component-model, conformal,
and locked-final-test path. Use it when intentionally running that research workflow. It
does not produce the staged artifact layout or Stage 1-to-Stage 2 lineage above.

### 1. Collection design and governance

- [ ] Confirm permission to use, retain, and model every sensor, weather, imagery, and GIS
  source, including any location-disclosure constraints.
- [ ] Record the study region, campaign dates, site-selection rationale, deployment
  protocol, sensor model/serial mapping, calibration certificates, maintenance events, and
  analyst responsible for label production.
- [ ] Confirm the intended deployment population and location types are represented; note
  known exclusions rather than claiming broader validity.
- [ ] Include observations across multiple complete hot dates and geographically separated
  sites. Before locking the final test, verify at least four separated unseen sites and at
  least one (preferably two) complete hot dates can be reserved.
- [ ] Define `date` as the site's local civil date and preserve a timezone-aware
  `timestamp_utc`. Document the IANA timezone and daylight-saving treatment.
- [ ] Predefine the projected CRS and spatial-block size from scientific/geographic
  considerations. Do not compare block sizes by model performance.

### 2. Sensor and label provenance

- [ ] Synchronize local air temperature, RH, pedestrian wind, and globe temperature to a
  documented tolerance and averaging period.
- [ ] Verify measurement height and wind averaging conventions for each sensor deployment.
- [ ] Record globe diameter, emissivity, radiation/wind correction, and the exact MRT
  calculation.
- [ ] Record sensor calibration version and a row-level `quality_flag`; retain failed rows
  for audit if policy permits, but do not fit on them.
- [ ] Calculate local vapor pressure only from paired measured local temperature and RH.
- [ ] Calculate continuous `calculated_utci_c` with the documented pinned UTCI procedure.
  Confirm units, wind reference height, input-limit handling, and `round_output=False`.
- [ ] Derive `utci_category` from the fixed 26/32/38/46 °C thresholds. Do not hand-label it
  or create a separate classification outcome.
- [ ] Supply nonnegative `label_uncertainty_c` and document whether it is a standard
  uncertainty, confidence half-width, or another quantity.
- [ ] Treat optional WBGT as a comparison label only, never a predictor.

### 3. Operational feature provenance

- [ ] Verify each proposed predictor appears in the exact `core_predictors` allow-list in
  `configs/features.yaml` and is available for a genuinely unsensed location.
- [ ] Confirm all weather history and heating-rate variables use only the operational
  background weather source. Local sensor values must not feed lags, rolling windows, or
  interactions.
- [ ] Confirm weather-source distance and age are nonnegative and that no source observation
  comes from after the prediction instant.
- [ ] Record weather product/station versions, temporal resolution, reference wind height,
  interpolation procedure, and quality controls.
- [ ] Record reflectance product/collection, acquisition or composite date, cloud masking,
  resampling, and equations for NDVI, NDBI, NDWI, and albedo proxy.
- [ ] Record land-cover taxonomy, year, resolution, fraction computation, and treatment of
  mixed/unknown pixels.
- [ ] Record DEM, coastline, road, and building sources and their CRS/resolution/version.
- [ ] Compute 15/30/60/90 m summaries using the prespecified geometries and definitions.
  Do not choose buffers by held-out performance.
- [ ] Verify fractions are on `[0, 1]`, distances/heights are meters, pressure is Pa,
  radiation is W/m², and cyclic variables are within `[-1, 1]`.
- [ ] Generate only the prespecified interaction columns. Record formulas and denominator
  handling for canopy-to-impervious ratios.

### 4. Optional satellite-enhanced inputs

- [ ] Keep the core table usable when no thermal image exists; same-day LST must not become
  a hidden core requirement.
- [ ] Use quality-controlled Landsat or ECOSTRESS LST and record product/collection,
  retrieval version, units, emissivity method, cloud mask, view geometry, pixel support,
  acquisition timestamp, and reprojection/resampling.
- [ ] Compute image age as observation time minus acquisition time; reject future images.
- [ ] Compute LST-minus-air using operational background air temperature, not local measured
  air temperature.
- [ ] Populate source, quality score/flag, cloud fraction, valid-pixel fraction, view angle,
  cyclic overpass time, and explicit missingness.
- [ ] Freeze maximum image age and quality eligibility before tuning.
- [ ] Plan the enhanced-versus-core comparison on exactly the same eligible observations.

### 5. Assemble the immutable table

- [ ] Make each row one site-time observation and assign a globally unique `sample_id`.
- [ ] Supply all identifiers, provenance, sensor/label, and operational columns documented
  in `docs/DATA_SCHEMA.md`.
- [ ] Use exact case-sensitive names and SI units. Do not use textual missing sentinels,
  infinities, mixed scalar types, duplicate columns, or duplicate sample IDs.
- [ ] Set `split_role` to `unassigned` if a frozen manifest has not yet been made. Do not
  assign rows by inspecting model errors.
- [ ] Store the table as CSV or Parquet at a user-chosen location. The project does not
  require or create a `data` directory.
- [ ] Make a read-only archival copy outside the runtime output location and record its
  cryptographic hash.

### 6. Validate without changing the table

From the project root, run the following with the absolute path to the real table:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai validate --data "C:\absolute\path\to\real_training_table.parquet"
```

- [ ] Review every reported error and warning against the original data/provenance.
- [ ] Correct upstream collection or feature-production code; do not ask the validator to
  clip, coerce, fill, deduplicate, or overwrite the source table.
- [ ] Rerun validation until errors are resolved. Preserve validation logs with the study
  record when an explicit runtime output is later created.
- [ ] If predicting on an unlabeled operational table, use `--mode prediction`; this does
  not relax predictor leakage checks.

### 7. Freeze the split manifest

Choose an explicit manifest destination. This command intentionally creates that file only
when you run it with real data:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai make-splits --data "C:\absolute\path\to\real_training_table.parquet" --manifest "C:\absolute\path\to\study_artifacts\split_manifest.json"
```

- [ ] Inspect counts by role, site, block, date, weather event, heat category, and quality.
- [ ] Verify every held-out date/event is complete and no development row shares either a
  block or date with its validation rows after embargo.
- [ ] Verify calibration consists of complete unseen spatial-day groups.
- [ ] Verify the final-test composition target is feasible. If it is not, collect more real
  data rather than weakening the lock silently.
- [ ] Freeze and version the manifest. Do not overwrite it after viewing evaluation results.

### 8. Tune, train, and calibrate intentionally

Select explicit runtime destinations; the application creates them only now:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai tune --data "C:\absolute\path\to\real_training_table.parquet" --manifest "C:\absolute\path\to\study_artifacts\split_manifest.json" --output-dir "C:\absolute\path\to\study_artifacts\tuning"
```

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai train --data "C:\absolute\path\to\real_training_table.parquet" --manifest "C:\absolute\path\to\study_artifacts\split_manifest.json" --tuning-result "C:\absolute\path\to\study_artifacts\tuning\best_trial.json" --output-dir "C:\absolute\path\to\study_artifacts\run_001"
```

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai calibrate --data "C:\absolute\path\to\real_training_table.parquet" --manifest "C:\absolute\path\to\study_artifacts\split_manifest.json" --run-dir "C:\absolute\path\to\study_artifacts\run_001"
```

- [ ] Confirm inner folds alone controlled hyperparameters and early stopping.
- [ ] Confirm every outer refit used the median inner-fold best tree count and all
  outer-training rows.
- [ ] Confirm the calibration role was untouched until conformal calibration.
- [ ] Keep the heat-weighted experiment disabled for the preregistered primary result unless
  an explicit, timestamped configuration amendment says otherwise.
- [ ] Enable satellite-enhanced and neural comparisons only deliberately; label them
  secondary.

### 9. Development evaluation and interpretation

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai evaluate --data "C:\absolute\path\to\real_training_table.parquet" --manifest "C:\absolute\path\to\study_artifacts\split_manifest.json" --run-dir "C:\absolute\path\to\study_artifacts\run_001"
```

- [ ] Report support with every metric and `unavailable` when a threshold subset has no
  positive cases.
- [ ] Use paired whole-block bootstrap intervals and identify observation-weighted versus
  block-balanced results.
- [ ] Compare direct and component UTCI on identical valid observations only; never average
  them. Freeze the development-OOF disagreement-warning threshold.
- [ ] Report conformal coverage/width overall and by prespecified subgroup and state the
  exchangeability limitation.
- [ ] Describe SHAP and partial dependence as predictive associations, not causal effects.
- [ ] Investigate extrapolation and anomaly warnings without automatically deleting rows.

### 10. Lock and open the final test once

- [ ] Before opening the final test, freeze the dataset, checked-in configs, recorded runtime
  choices, predictor allow-list, manifest, code revision, selected model, baselines, subgroup
  definitions, and reporting plan.
- [ ] Confirm their SHA-256 values match the stored lock and archive the lock separately.
- [ ] Obtain the study lead's explicit authorization for the one-time assessment.
- [ ] Run only after all choices are frozen:

```powershell
.venv\Scripts\python.exe -m urban_heat_risk_ai final-test --data "C:\absolute\path\to\real_training_table.parquet" --manifest "C:\absolute\path\to\study_artifacts\split_manifest.json" --run-dir "C:\absolute\path\to\study_artifacts\run_001" --unlock-final-test
```

- [ ] Treat a hash mismatch as a stop condition. Do not update the lock simply to make the
  command pass.
- [ ] Do not retune, select features, change thresholds, or repeatedly inspect the final
  result. Any later analysis must be labeled exploratory or use newly collected real data.
