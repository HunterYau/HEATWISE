# Real-data schema

## Scope and row grain

The only supported modeling input is a user-supplied real CSV or Parquet file passed
through `--data`. The software never searches for data. Each row is one pedestrian-level
observation at one site and one instant. Repeated observations at a site are separate rows;
aggregated site-day rows are not valid.

Column names are case-sensitive `snake_case`. Units are SI and are part of the column
contract. The validator reads the table without changing, sorting, imputing, clipping,
casting, deduplicating, or overwriting it. CSV values must already use the documented units.
Parquet unit metadata, when present, must agree with this document.

The schema has three kinds of columns:

1. metadata used for provenance, spatial blocking, and temporal blocking;
2. sensor/label columns used only to construct or verify outcomes; and
3. operational predictors available at an unsensed location.

Only exact names in `configs/features.yaml` may enter a predictor matrix. Metadata,
coordinates, split fields, sensor values, labels, and their transformations are denied by a
central leakage guard.

## Identifiers, provenance, and split metadata

| Column | Type | Required | Contract |
|---|---|---:|---|
| `sample_id` | string | yes | Nonempty and unique over the file; stable across reruns. |
| `site_id` | string | yes | Stable site identifier. Used for evaluation grouping, never prediction. |
| `date` | date/string | yes | Observation's local civil date in `YYYY-MM-DD`; it must agree with the documented site timezone and is held out whole. |
| `timestamp_utc` | timezone-aware datetime/string | yes | ISO-8601 instant with `Z` or explicit `+00:00`; parsed value must be UTC. Naive timestamps are errors. |
| `latitude` | float | yes | WGS84 decimal degrees, `-90` to `90`. Used only for splitting/mapping. |
| `longitude` | float | yes | WGS84 decimal degrees, `-180` to `180`. Used only for splitting/mapping. |
| `spatial_block_id` | string | conditional | Required for training. It may be absent only when `make-splits` is asked to derive deterministic candidate blocks from coordinates and the configured CRS/block size. |
| `weather_event_id` | string | no | Stable synoptic-event identifier. If present, events are held out whole. |
| `sensor_id` | string | yes | Stable instrument identifier; never a predictor. |
| `measurement_height_m` | float | yes | Height above local ground, greater than `0` and at most `50 m`; used by physics only. |
| `calibration_version` | string | yes | Nonempty version or certificate reference; never a predictor. |
| `quality_flag` | categorical string | yes | One of `pass`, `suspect`, or `fail`; only configured values (by default `pass`) are eligible for fitting. |
| `split_role` | categorical string | yes | One of `unassigned`, `development`, `calibration`, or `final_test`. `make-splits` may consume `unassigned`; it writes a separate manifest and never edits this column. |

Optional prespecified evaluation metadata may include `sun_shade_group`,
`coast_distance_group`, and `time_of_day_group`. These are never predictors.
`time_of_day_group` must describe local civil/solar time using a documented fixed rule; if
it is absent, reporting reconstructs local hour from `hour_sin` and `hour_cos`, never from
the UTC clock hour. `land_cover_class` serves both as an allowed operational categorical
predictor and as the prespecified land-cover subgroup.

If local-date semantics cannot be established from the collection protocol, add a
`timezone_name` metadata column containing an IANA timezone (for example,
`America/Los_Angeles`) and resolve the ambiguity before splitting. Daylight-saving folds
must still map to unique UTC instants.

## Sensor measurements and labels

These fields may be used to calculate labels, train the four physical-component targets,
or verify label integrity. They are prohibited from every operational predictor matrix.

| Column | Type/unit | Required | Plausible range and meaning |
|---|---|---:|---|
| `measured_air_temperature_c` | float, °C | yes | `-50` to `65`; calibrated pedestrian-level air temperature. |
| `measured_relative_humidity_pct` | float, % | yes | `0` to `100`; paired in time/height with measured air temperature. |
| `measured_pedestrian_wind_speed_m_s` | float, m/s | yes | `0` to `40`; averaging period must be documented. |
| `measured_globe_temperature_c` | float, °C | yes | `-60` to `120`; globe diameter/emissivity and response correction must be documented. |
| `calculated_mrt_c` | float, °C | yes | `-70` to `120`; mean radiant temperature calculated from the approved globe-temperature method. |
| `calculated_utci_c` | float, °C | yes | `-100` to `100`; the continuous primary target, calculated with the pinned UTCI contract. |
| `utci_category` | categorical string | yes | Must exactly match the fixed thresholds below when the continuous label is present. |
| `optional_wbgt_c` | float, °C | no | `-50` to `65`; optional comparison outcome, never a predictor. |
| `label_uncertainty_c` | float, °C | yes | Nonnegative and at most `30`; documented one-standard-uncertainty or other explicitly named uncertainty measure. |

Accepted UTCI category values are `no_heat_stress`, `moderate`, `strong`,
`very_strong`, and `extreme`. They are derived, not modeled:

| Continuous UTCI | Category |
|---|---|
| `< 26 °C` | `no_heat_stress` |
| `>= 26 °C` and `< 32 °C` | `moderate` |
| `>= 32 °C` and `< 38 °C` | `strong` |
| `>= 38 °C` and `< 46 °C` | `very_strong` |
| `>= 46 °C` | `extreme` |

The full label-production method, pythermalcomfort version, globe corrections, weather
averaging windows, sensor synchronization tolerance, and uncertainty definition must be
recorded outside the training table and cited in experiment metadata.

## Core operational predictors

The canonical allow-list is `core_predictors` in `configs/features.yaml`. Every value must
represent information obtainable at the prediction instant for an unsensed location. A
same-day thermal image is never required. A training run uses a frozen ordered list and
refuses absent, undeclared, extra selected, or all-missing predictors.

### Background weather and temporal history

| Columns | Type/unit | Plausible range or rule |
|---|---|---|
| `background_air_temperature_c`, `background_dew_point_c` | float, °C | `-80` to `65`; dew point must not materially exceed air temperature. |
| `background_relative_humidity_pct` | float, % | `0` to `100`; at least dew point or RH is required. |
| `background_wind_speed_m_s` | float, m/s | `0` to `75`; source reference height must be documented. |
| `background_surface_pressure_pa` | float, Pa | `50000` to `110000`. |
| `background_cloud_cover_pct` | float, % | `0` to `100`. |
| `background_shortwave_radiation_w_m2` | float, W/m² | `0` to `1400`; nighttime values should be zero within instrument tolerance. |
| `background_precipitation_1h_mm`, `background_precipitation_3h_mm` | float, mm | `0` to `500`; accumulation windows end at the observation instant. |
| `background_weather_source_distance_m` | float, m | `0` to `500000`. |
| `background_weather_age_minutes` | float, min | `0` to `10080`; values must not come from the future. |
| `background_air_temperature_lag_1h_c`, `_lag_3h_c`, `_lag_24h_c` | float, °C | `-80` to `65`; operational background series only. |
| `background_temperature_heating_rate_c_per_h` | float, °C/h | `-30` to `30`; causal window only. |
| `background_cumulative_hot_hours_24h` | float, h | `0` to `24`; threshold and missing-window handling must be documented. |

Do not calculate lags or heating features from local measured temperatures. At a partition
boundary, their upstream background observations may precede the row, but no local label
information may cross the boundary.

### Satellite reflectance, cover, terrain, vegetation, and urban form

| Columns/family | Type/unit | Plausible range or rule |
|---|---|---|
| `ndvi`, `ndbi`, `ndwi` | float | `-1` to `1`; processing collection and compositing dates documented. |
| `albedo_proxy` | float | `0` to `1`; calculation and spectral source documented. |
| `land_cover_class` | categorical string | One of the versioned categories in `features.yaml`; unknown values are encoded safely but reported. |
| `impervious_fraction`, `canopy_fraction`, `grass_fraction`, `water_fraction` | float fraction | `0` to `1`; overlapping classification systems must be documented. |
| `elevation_m` | float, m | `-500` to `9000`. |
| `slope_degrees` | float, degrees | `0` to `90`. |
| `aspect_sin`, `aspect_cos` | float | `-1` to `1`; pair norm should be approximately one for defined aspects. |
| `distance_to_coast_m`, `distance_to_major_road_m` | float, m | Nonnegative and at most `5,000,000`. |
| `building_fraction`, `road_fraction` | float fraction | `0` to `1`. |
| `sky_view_factor`, `estimated_shade_fraction` | float fraction | `0` to `1`; time basis for shade must match the observation. |

### Solar and cyclic time

| Columns | Type/unit | Plausible range or rule |
|---|---|---|
| `solar_elevation_degrees` | float, degrees | `-90` to `90`. |
| `solar_azimuth_sin`, `solar_azimuth_cos` | float | Each `-1` to `1`; pair norm approximately one when defined. |
| `hour_sin`, `hour_cos` | float | Each `-1` to `1`; derived from local solar/civil convention documented by the producer. |
| `day_of_year_sin`, `day_of_year_cos` | float | Each `-1` to `1`; leap-year convention documented. |

The raw timestamp and raw solar azimuth are removed before modeling. Cyclic variables must
be reproducible deterministic functions, not target-aware encodings.

### Multiscale buffers and prespecified interactions

For each valid radius `15`, `30`, `60`, and `90 m`, the schema permits:

- `canopy_fraction_{r}m`, `impervious_fraction_{r}m`, `grass_fraction_{r}m`,
  `water_fraction_{r}m`, `building_fraction_{r}m`, and `road_fraction_{r}m`, all
  fractions in `[0, 1]`;
- `mean_building_height_m_{r}m`, nonnegative and no more than `1000 m`;
- `sky_view_factor_{r}m`, `ndvi_mean_{r}m`, and `albedo_mean_{r}m` using the ranges above;
- `canopy_to_impervious_ratio_{r}m`, nonnegative, with a documented denominator epsilon
  and missing value when the ratio is undefined.

The only permitted explicit products are `solar_elevation_x_shortwave`,
`shortwave_x_shade_fraction`, `shortwave_x_canopy_fraction`,
`background_wind_x_sky_view_factor`, `background_wind_x_building_fraction`, and
`distance_to_coast_x_background_wind`. Definitions are frozen in `features.yaml`.

## Satellite-enhanced additions

The enhanced model is a separate model. Its effective allow-list is the exact union of the
core list and `satellite_enhanced_additions`. It may train only on rows meeting the image-age
and quality rules in `model.yaml`.

| Column | Type/unit | Plausible range or rule |
|---|---|---|
| `satellite_lst_c` | float, °C | `-80` to `100`; quality-controlled Landsat or ECOSTRESS surface temperature. |
| `satellite_lst_minus_background_air_temperature_c` | float, °C | `-100` to `100`; uses operational background air, never local measured air. |
| `satellite_image_age_hours` | float, h | Nonnegative; observation time minus acquisition time. Future imagery is an error. |
| `satellite_overpass_hour_sin`, `satellite_overpass_hour_cos` | float | Each `-1` to `1`; raw overpass timestamp is metadata, not a predictor. |
| `satellite_lst_quality_score` | float | `0` to `1`, higher-is-better definition documented. |
| `satellite_cloud_fraction`, `satellite_valid_pixel_fraction` | float fraction | `0` to `1`. |
| `satellite_view_zenith_degrees` | float, degrees | `0` to `90`. |
| `satellite_lst_missing` | integer/bool | Exactly `0` or `1`, consistent with `satellite_lst_c`. |
| `satellite_thermal_source` | categorical string | `landsat` or `ecostress`. |
| `satellite_lst_quality_flag` | categorical string | `good`, `acceptable`, or a documented ineligible value. |

The comparison refits/evaluates a core model on exactly the same eligible observations as
the enhanced model. Results from different row sets are not treated as model improvements.

## Missingness and dtype rules

- Identifiers, timestamps, coordinates, provenance, role, quality, and the primary target
  may not be missing on supervised rows.
- Sensor labels needed by a selected component model may not be missing for that model.
- Numerical predictors may contain true `NaN`. Empty strings, textual sentinels such as
  `-9999`, and infinities are errors.
- By default the validator warns when an allowed predictor exceeds `20%` missingness,
  errors at `80%`, and errors when a selected predictor is absent or entirely missing.
- XGBoost retains numerical `NaN` and receives fitted missingness indicators. Linear and
  neural comparisons use training-fitted imputation/scaling. Categorical nulls receive a
  training-fitted missing token; unseen categories are ignored safely by one-hot encoding.
- Numeric columns must be numeric after parsing without coercing invalid strings. Boolean
  indicators accept only booleans or integer `0/1`. Categorical and identifier columns must
  not mix scalar types.

## Validation report and failure policy

`validate --data PATH` emits an actionable report and does not write a cleaned table. Each
issue contains severity, row/column context when safe, the violated rule, observed value or
count, and a corrective action. Checks include:

- supported extension and readable CSV/Parquet structure;
- exact column names, duplicate names, dtypes, and SI-unit suffixes/metadata;
- physical ranges, finite values, cross-field consistency, and category thresholds;
- parseable UTC-aware timestamps, `date` consistency, future-age errors, and ordering
  assumptions used by lag features;
- duplicate/missing `sample_id`, missing required metadata, quality values, and split roles;
- missingness by column and by split role;
- availability and categorical declarations for the requested predictor set;
- exact allow-list membership plus banned and suspicious target/sensor-derived names; and
- when a manifest exists, disjoint roles and no spatial-block/date leakage.

Errors stop downstream commands. Warnings require review but do not authorize automatic
repair. Source files are always treated as immutable.
