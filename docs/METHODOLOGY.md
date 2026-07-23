# Methodology

## Estimand and operating setting

The primary task is to predict continuous pedestrian-level Universal Thermal Climate Index
(`calculated_utci_c`) for a site-time where no local heat sensor is available. Therefore,
every predictor must be operationally obtainable at an unsensed location. Raw coordinates
may define spatial blocks and maps, but they are not primary predictors. Sensor readings,
labels, identifiers, calibration/quality metadata, raw timestamps, and split membership are
never predictors.

The preregistered primary result is an unweighted direct UTCI XGBoost model using the
`core` feature set. A thermal-image-enhanced model, physical-component system, weighted
experiment, and comparison models are secondary analyses. No category classifier is fit;
heat categories are fixed deterministic functions of continuous predictions.

## Leakage control and preprocessing

`configs/features.yaml` is the versioned feature contract. Before any estimator sees a
matrix, a central guard checks that every column is declared for the requested predictor
set, no exact deny-list name is present, and no suspicious target/sensor-derived pattern is
present. Any violation terminates fitting or prediction. This check is repeated after
feature construction and preprocessing so an unsafe transformation cannot bypass it.

For XGBoost, numerical `NaN` values remain `NaN`, and explicit missingness indicators are
learned from columns that are missing in the current training partition. Categorical
variables are one-hot encoded with a fixed learned category order and unknown-category
handling. The learned feature order is stored with the model and asserted at prediction.
Encoders, imputers, scalers, missingness decisions, anomaly detectors, and category levels
are fit only on the relevant training partition.

Continuous variables are standardized only for the regularized-linear and neural-network
comparisons. XGBoost and Random Forest inputs are not standardized. Any imputation required
by a comparison model is fit on its training rows only.

## Spatiotemporal design

Random row splitting and ordinary grouping on a concatenated site/date key are prohibited.
`SpatioTemporalBlockedSplit` operates on distinct spatial and temporal restrictions:

1. Locations receive stable `spatial_block_id` values. If they are not supplied,
   coordinates are transformed from WGS84 into the configured projected CRS and assigned by
   deterministic floor indexing at the prespecified block size. Block size is never tuned
   against model performance.
2. Complete civil dates are held out. When `weather_event_id` is available, complete events
   are held out too.
3. For each validation fold, any candidate training row sharing either a spatial block or a
   date (or configured weather event) with validation is embargoed.
4. Assertions verify disjoint sample IDs, blocks, dates, events, and roles before fitting.

Development data use nested blocked cross-validation. Outer folds estimate generalization;
inner folds select hyperparameters and objective. A separate calibration role contains
complete unseen spatial-day groups and is untouched by model fitting/tuning. The locked
final test contains complete unseen groups and is designed to include at least four
geographically separated unseen sites plus at least one, preferably two, complete hot
dates. Feasibility failures are reported rather than weakened silently.

The split manifest is a separate immutable artifact. Creating it never edits the real data.
Every result records the dataset, configuration, allow-list, and manifest hashes.

## Primary direct UTCI model

The direct estimator is `xgboost.XGBRegressor` with `tree_method="hist"`,
`eval_metric="mae"`, deterministic seeds, and CPU execution by default. Within each inner
fold, Optuna chooses between `reg:absoluteerror` and `reg:pseudohubererror` and searches the
prespecified space:

- approximately 300–1,500 effective trees;
- maximum depth 3–8;
- log-scaled learning rate 0.01–0.10;
- row and column subsampling 0.60–1.00;
- minimum child weight, L1, L2, optional gamma, and pseudo-Huber slope.

Early stopping (about 50 rounds) sees only the current inner-validation fold. For the
selected trial, the best iteration is recorded separately for each inner fold. The outer
model is then refit on all outer-training rows using the median selected tree count. The
outer evaluation fold never chooses an early-stopping iteration, preprocessing parameter,
threshold, or hyperparameter.

The Optuna objective is mean grouped validation MAE, with spatial-day groups kept visible in
aggregation. High-UTCI MAE (`UTCI >= 32 °C`) is recorded as a secondary metric, not folded
into primary selection. A heat-weighted experiment is disabled by default. Enabling it
creates a clearly labeled secondary result and does not replace the unweighted primary
unless the configuration is explicitly changed and the change is recorded.

## Satellite-enhanced model

The core model never requires same-day land-surface temperature. The enhanced XGBoost model
is trained separately using the core features plus quality-controlled Landsat or ECOSTRESS
LST, LST minus operational background air temperature, image age, cyclic overpass time,
quality variables, and missingness information.

Eligibility is determined before fitting by fixed source, maximum image age, cloud fraction,
quality flag, valid-pixel, and valid-LST requirements. Its fair comparison is a core model
refit and evaluated on exactly the same eligible rows and split groups. An enhanced score
must not be compared with a core score computed on a broader or different sample.

## Physical-component pathway

Four supporting XGBoost models use the same operational feature contract and split manifest:

- **A — local air-temperature departure:**
  `measured_air_temperature_c - background_air_temperature_c`;
- **B — local vapor pressure:** saturation vapor pressure at measured local air
  temperature multiplied by measured RH fraction;
- **C — stable pedestrian-wind adjustment:**
  `log1p(measured_pedestrian_wind_speed_m_s) - log1p(background_wind_speed_m_s)`;
- **D — radiant departure:** `calculated_mrt_c - measured_air_temperature_c`.

These quantities are labels only for their respective component estimators and are blocked
by the leakage guard from all predictor matrices. Inverse equations reconstruct local air
temperature, vapor pressure/RH, pedestrian wind, and MRT. The implementation retains
unclipped raw predictions and emits explicit domain, finite-value, and physical-consistency
flags. Invalid reconstructions follow the configured `flag_and_return_nan` policy rather
than being hidden by silent clipping.

Component inputs are then passed to the pinned UTCI physics wrapper. Direct and
component-derived UTCI are evaluated only on their identical valid held-out rows. Their
estimates are never averaged. A disagreement-warning threshold is learned once from the
absolute difference in development out-of-fold predictions and applied unchanged to later
partitions.

## Humidity and wind physics

Saturation vapor pressure and RH conversions use a documented, numerically stable SI
formula. Predicted vapor pressure is converted back to RH using reconstructed local air
temperature, with raw RH retained and out-of-range values flagged.

UTCI requires wind at 10 m. When a wind value is observed or reconstructed at height `z`,
the configured neutral logarithmic profile is

```text
u10 = uz * log((10 - d) / z0) / log((z - d) / z0)
```

where `z0` is roughness length and `d` is displacement height. The conversion is applicable
only when heights and log arguments are valid and the neutral-profile assumption is
defensible. The software reports invalid/applicability flags, retains the pre-conversion
wind, and computes sensitivity values at the prespecified roughness lengths. It does not
select roughness length based on held-out performance.

The UTCI wrapper calls the version-pinned `pythermalcomfort.models.utci` in SI units with
`round_output=False`. The wrapper applies one explicit set of input limits for air
temperature, MRT–air difference, 10 m wind, and RH. Because the library's implicit limiting
is disabled, the wrapper can distinguish out-of-range from missing inputs and return a
clear flag/`NaN` according to configuration. Scalar regression tests anchor documented UTCI
cases for the pinned version.

## Comparison models

All learned comparisons use the same outer partitions and training-only preprocessing:

- operational background weather-station air temperature;
- a unit-tested Rothfusz Heat Index calculation with documented applicability and humidity
  adjustments;
- satellite LST alone, only on eligible observations;
- regularized linear regression with standardized continuous inputs;
- a 500-tree Random Forest;
- an optional small PyTorch network with hidden widths 128/64/32, GELU (or configured ReLU),
  0.20 dropout, AdamW, Huber/MAE loss, CPU default, and validation-only early stopping.

The neural baseline is fully specified but disabled by default to avoid accidental runtime
cost. None of these comparisons controls the primary model's selection.

## Metrics and subgroup reporting

Continuous metrics are MAE, RMSE, R², Spearman correlation, and mean bias. Fixed-category
metrics are accuracy, macro F1 across all five fixed labels, per-class support counts, and a
fixed-order confusion matrix. Recall is reported for UTCI thresholds of at least 32, 38,
and 46 °C. When a held-out subset contains no positive case for a threshold, recall is
reported as `unavailable`, not zero or one.

Prespecified subgroup reporting covers observed heat category, sun versus shade, land-cover
class, coast-distance group, and time of day. Small or empty groups are shown with their
support and an unavailable metric where appropriate; they are not silently dropped.
The configured `time_of_day_group` is treated as local time and is preferred. If it is
absent, the framework reconstructs local hour from the documented `hour_sin`/`hour_cos`
pair; it never labels the UTC clock hour as local time of day.

Model-versus-baseline improvements use paired bootstrap resampling of whole spatial-day or
site-day blocks. Individual rows are never the bootstrap unit. Both observation-weighted
metrics and block-balanced metrics (equal weight per resampled block) are reported, along
with the resampling unit, random seed, confidence level, and number of usable blocks.

## Split-conformal intervals

After model selection and fitting are complete, absolute residuals on the untouched
calibration partition form a 90% split-conformal interval. With `n` finite calibration
residuals and `alpha = 0.10`, the rank is

```text
k = ceil((n + 1) * (1 - alpha)).
```

If `k <= n`, the radius is the kth smallest residual using one-based indexing. If the
configured minimum sample size is not met or `k > n`, the result is an unbounded interval
with an explicit warning. The code does not substitute a convenient finite quantile.

Empirical coverage and interval width are reported overall and in the prespecified
subgroups. These are descriptive diagnostics; the finite-sample theoretical guarantee
depends on exchangeability between calibration and future observations, which
spatiotemporal shift may weaken.

## Explanations and extrapolation

Runtime explanation artifacts include SHAP global bars, beeswarms, selected local waterfall
plots, and partial-dependence plots for canopy, impervious surface, shortwave radiation,
wind, and coastal distance. Each output is labeled as a predictive association, not evidence
that changing a feature will cause a change in UTCI.

Local waterfalls are requested explicitly by repeating
`--local-explanation-sample-id SAMPLE_ID` on the `evaluate` command. This avoids silently
choosing a favorable or dramatic case after seeing outcomes.

Feature-range checks compare prediction rows with training ranges and report univariate
extrapolation. The optional multivariate anomaly detector is fit only on the relevant
training partition and is diagnostic by default; it does not automatically exclude rows.

## Final-test lock and artifact provenance

The final test is locked by default. Before it can run, SHA-256 hashes of the real dataset
bytes, checked-in model configuration file, predictor allow-list, and split manifest must
match the frozen lock, and the user must supply `--unlock-final-test`. A mismatch stops
evaluation. Runtime choices and selected settings are recorded separately in run metadata.
This mechanism discourages repeated test-set adaptation; process discipline is still
required to preserve a genuinely one-time final assessment.

Models, preprocessing objects, predictions, metrics, plots, hashes, copied checked-in
configurations, selected/runtime settings, experiment metadata, and software versions are
written only during explicit runtime commands with real data. Imports, configuration checks,
schema validation, and this implementation task do not create runtime artifact directories or
claim any result.
