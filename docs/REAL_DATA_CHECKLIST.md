# Real-data readiness checklist

Use this checklist only with observations collected from real sites. Do not create a mock,
synthetic, example, or filled-in training table to satisfy a check. If a required item is
not available, stop and document the gap.

## 1. Collection design and governance

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

## 2. Sensor and label provenance

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

## 3. Operational feature provenance

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

## 4. Optional satellite-enhanced inputs

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

## 5. Assemble the immutable table

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

## 6. Validate without changing the table

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

## 7. Freeze the split manifest

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

## 8. Tune, train, and calibrate intentionally

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

## 9. Development evaluation and interpretation

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

## 10. Lock and open the final test once

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
