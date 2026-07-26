"""Physical calculations used by the component-based UTCI model.

The functions in this module are deliberately independent of model training.  They
use SI units throughout, preserve unrounded values, and make every applicability
decision explicit.  In particular, :func:`calculate_utci` is a narrow wrapper
around the version-pinned :mod:`pythermalcomfort` implementation rather than a
second, locally maintained UTCI approximation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

type NumericResult = float | NDArray[np.float64]
type OutOfRangePolicy = Literal["raise", "nan", "clip"]

PYTHERMALCOMFORT_GLOBE_MRT_STANDARD = "ISO"

# pythermalcomfort/UTCI polynomial applicability limits.  The installed,
# version-pinned implementation uses the same inclusive numerical bounds when
# ``limit_inputs=True``.
UTCI_AIR_TEMPERATURE_RANGE_C = (-50.0, 50.0)
UTCI_MRT_DELTA_RANGE_C = (-30.0, 70.0)
UTCI_WIND_10M_RANGE_M_S = (0.5, 17.0)
RELATIVE_HUMIDITY_RANGE_PERCENT = (0.0, 100.0)


def _resolve_utci_limits(
    limits: Mapping[str, Sequence[float]] | None,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    defaults = {
        "air_temperature_c": UTCI_AIR_TEMPERATURE_RANGE_C,
        "mrt_minus_air_temperature_c": UTCI_MRT_DELTA_RANGE_C,
        "wind_10m_m_s": UTCI_WIND_10M_RANGE_M_S,
        "relative_humidity_pct": RELATIVE_HUMIDITY_RANGE_PERCENT,
    }
    configured = dict(limits or {})
    resolved: list[tuple[float, float]] = []
    for key, published in defaults.items():
        raw = configured.get(key, published)
        if len(raw) != 2:
            raise ValueError(f"UTCI limit {key} must contain [minimum, maximum]")
        lower, upper = float(raw[0]), float(raw[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"UTCI limit {key} must be finite and increasing")
        if lower < published[0] or upper > published[1]:
            raise ValueError(
                f"Configured UTCI limit {key} cannot extend beyond the pinned "
                f"implementation domain {published}"
            )
        resolved.append((lower, upper))
    return resolved[0], resolved[1], resolved[2], resolved[3]


def _return_scalar_if_scalar(value: NDArray[np.float64]) -> NumericResult:
    """Return a Python float for zero-dimensional results, otherwise an array."""

    if value.ndim == 0:
        return float(value)
    return value


def saturation_vapor_pressure_kpa(air_temperature_c: ArrayLike) -> NumericResult:
    """Return saturation vapour pressure in kPa using the Magnus equation.

    The coefficients are appropriate for liquid water in the environmental
    temperature range used by this project.  Inputs must be greater than
    -243.04 °C, where the parameterisation becomes singular.
    """

    temperature = np.asarray(air_temperature_c, dtype=float)
    if np.any(~np.isfinite(temperature)):
        raise ValueError("air_temperature_c must contain only finite values")
    if np.any(temperature <= -243.04):
        raise ValueError("air_temperature_c must be greater than -243.04 °C")
    pressure = 0.61094 * np.exp((17.625 * temperature) / (temperature + 243.04))
    return _return_scalar_if_scalar(np.asarray(pressure, dtype=float))


def relative_humidity_to_vapor_pressure(
    air_temperature_c: ArrayLike,
    relative_humidity_percent: ArrayLike,
    *,
    validate: bool = True,
) -> NumericResult:
    """Convert air temperature and relative humidity to vapour pressure (kPa)."""

    temperature, rh = np.broadcast_arrays(
        np.asarray(air_temperature_c, dtype=float),
        np.asarray(relative_humidity_percent, dtype=float),
    )
    if np.any(~np.isfinite(rh)):
        raise ValueError("relative_humidity_percent must contain only finite values")
    if validate and np.any((rh < 0.0) | (rh > 100.0)):
        raise ValueError("relative_humidity_percent must be between 0 and 100")
    saturation = np.asarray(saturation_vapor_pressure_kpa(temperature), dtype=float)
    pressure = saturation * rh / 100.0
    return _return_scalar_if_scalar(np.asarray(pressure, dtype=float))


def vapor_pressure_to_relative_humidity(
    air_temperature_c: ArrayLike,
    vapor_pressure_kpa: ArrayLike,
    *,
    clip: bool = False,
) -> NumericResult:
    """Convert vapour pressure (kPa) to relative humidity (percent).

    By default supersaturation and negative model predictions are retained so
    callers can flag them.  Set ``clip=True`` only for an explicitly requested
    sensitivity calculation; component reconstruction never silently clips.
    """

    temperature, pressure = np.broadcast_arrays(
        np.asarray(air_temperature_c, dtype=float),
        np.asarray(vapor_pressure_kpa, dtype=float),
    )
    if np.any(~np.isfinite(pressure)):
        raise ValueError("vapor_pressure_kpa must contain only finite values")
    saturation = np.asarray(saturation_vapor_pressure_kpa(temperature), dtype=float)
    rh = 100.0 * pressure / saturation
    if clip:
        rh = np.clip(rh, 0.0, 100.0)
    return _return_scalar_if_scalar(np.asarray(rh, dtype=float))


# Short aliases used in target-construction code and external notebooks.
vapor_pressure_from_rh = relative_humidity_to_vapor_pressure
rh_from_vapor_pressure = vapor_pressure_to_relative_humidity


def wind_log_adjustment(
    local_wind_m_s: ArrayLike,
    background_wind_m_s: ArrayLike,
) -> NumericResult:
    """Return ``log1p(local wind) - log1p(background wind)``.

    ``log1p`` is well behaved at calm conditions and avoids unstable ratios.
    Negative wind speeds are physically invalid and are rejected.
    """

    local, background = np.broadcast_arrays(
        np.asarray(local_wind_m_s, dtype=float),
        np.asarray(background_wind_m_s, dtype=float),
    )
    if np.any(~np.isfinite(local)) or np.any(~np.isfinite(background)):
        raise ValueError("wind speeds must contain only finite values")
    if np.any(local < 0.0) or np.any(background < 0.0):
        raise ValueError("wind speeds cannot be negative")
    adjustment = np.log1p(local) - np.log1p(background)
    return _return_scalar_if_scalar(np.asarray(adjustment, dtype=float))


def invert_wind_log_adjustment(
    adjustment: ArrayLike,
    background_wind_m_s: ArrayLike,
) -> NumericResult:
    """Reconstruct local wind from a predicted logarithmic adjustment.

    Raw negative reconstructions are retained.  They are a model-validity signal
    and must not be silently floored before evaluation.
    """

    delta, background = np.broadcast_arrays(
        np.asarray(adjustment, dtype=float),
        np.asarray(background_wind_m_s, dtype=float),
    )
    if np.any(~np.isfinite(delta)) or np.any(~np.isfinite(background)):
        raise ValueError("adjustment and background wind must be finite")
    if np.any(background < 0.0):
        raise ValueError("background wind cannot be negative")
    reconstructed = np.expm1(delta + np.log1p(background))
    return _return_scalar_if_scalar(np.asarray(reconstructed, dtype=float))


# Descriptive aliases for configuration and documentation terminology.
stable_wind_adjustment = wind_log_adjustment
reconstruct_wind_from_adjustment = invert_wind_log_adjustment


def calculate_mean_radiant_temperature_from_globe(
    globe_temperature_c: ArrayLike,
    air_temperature_c: ArrayLike,
    air_speed_m_s: ArrayLike,
    *,
    globe_diameter_m: float,
    globe_emissivity: float,
    standard: str = PYTHERMALCOMFORT_GLOBE_MRT_STANDARD,
) -> NumericResult:
    """Derive unrounded MRT from a globe sensor using pinned pythermalcomfort.

    The staged workflow freezes globe diameter and emissivity in configuration.
    Only the ISO formulation is accepted so a configuration change cannot
    silently switch physical methods. Inputs and outputs use SI units, and no
    value is clipped or substituted.
    """

    if str(standard).strip().upper() != PYTHERMALCOMFORT_GLOBE_MRT_STANDARD:
        raise ValueError("standard must be 'ISO' for the frozen Stage 2 target method")
    diameter = float(globe_diameter_m)
    emissivity = float(globe_emissivity)
    if not isfinite(diameter) or diameter <= 0.0:
        raise ValueError("globe_diameter_m must be finite and greater than zero")
    if not isfinite(emissivity) or not 0.0 < emissivity <= 1.0:
        raise ValueError("globe_emissivity must be finite and in (0, 1]")

    globe, air, speed = np.broadcast_arrays(
        np.asarray(globe_temperature_c, dtype=float),
        np.asarray(air_temperature_c, dtype=float),
        np.asarray(air_speed_m_s, dtype=float),
    )
    if np.any(~np.isfinite(globe)) or np.any(~np.isfinite(air)) or np.any(~np.isfinite(speed)):
        raise ValueError("globe temperature, air temperature, and air speed must be finite")
    if np.any(globe <= -273.15) or np.any(air <= -273.15):
        raise ValueError("globe and air temperatures must be above absolute zero")
    if np.any(speed < 0.0):
        raise ValueError("air_speed_m_s cannot be negative")

    try:
        from pythermalcomfort.utilities import mean_radiant_tmp
    except ImportError as exc:  # pragma: no cover - dependency check gives context
        raise RuntimeError(
            "pythermalcomfort is required for globe-derived MRT; install the pinned "
            "project dependencies"
        ) from exc

    result = np.asarray(
        mean_radiant_tmp(
            tg=globe,
            tdb=air,
            v=speed,
            d=diameter,
            emissivity=emissivity,
            standard=PYTHERMALCOMFORT_GLOBE_MRT_STANDARD,
        ),
        dtype=float,
    )
    if np.any(~np.isfinite(result)):
        raise ValueError("the ISO globe-to-MRT calculation returned a non-finite value")
    return _return_scalar_if_scalar(result)


# A readable alias for target-construction callers.
mean_radiant_temperature_from_globe = calculate_mean_radiant_temperature_from_globe


@dataclass(frozen=True)
class WindProfileResult:
    """Result of a neutral logarithmic wind-profile height conversion."""

    wind_speed_10m_m_s: float
    applicable: bool
    flags: tuple[str, ...]
    measurement_height_m: float
    roughness_length_m: float
    displacement_height_m: float
    sensitivity_by_roughness_m: dict[float, float]


def _log_profile_speed(
    wind_speed_m_s: float,
    measurement_height_m: float,
    target_height_m: float,
    roughness_length_m: float,
    displacement_height_m: float,
) -> float:
    source_above_displacement = measurement_height_m - displacement_height_m
    target_above_displacement = target_height_m - displacement_height_m
    return wind_speed_m_s * (
        log(target_above_displacement / roughness_length_m)
        / log(source_above_displacement / roughness_length_m)
    )


def convert_wind_to_10m(
    wind_speed_m_s: float,
    measurement_height_m: float,
    roughness_length_m: float,
    *,
    displacement_height_m: float = 0.0,
    neutral_stability: bool = True,
    minimum_measurement_height_m: float | None = None,
    maximum_measurement_height_m: float | None = None,
    sensitivity_roughness_lengths_m: Sequence[float] = (),
    strict: bool = False,
) -> WindProfileResult:
    """Convert wind measured at ``measurement_height_m`` to 10 m.

    A neutral logarithmic profile is used:

    ``u(10) = u(z) * ln((10-d)/z0) / ln((z-d)/z0)``.

    The conversion is applicable only when the measurement and target heights
    are above both the displacement height and roughness length, wind is
    non-negative, and neutral stability has been declared.  Failed checks return
    ``NaN`` plus flags (or raise when ``strict=True``).  Configured alternative
    roughness lengths are reported as sensitivity values; none is selected using
    model performance.
    """

    values = (
        wind_speed_m_s,
        measurement_height_m,
        roughness_length_m,
        displacement_height_m,
    )
    if minimum_measurement_height_m is not None and not isfinite(
        float(minimum_measurement_height_m)
    ):
        raise ValueError("minimum_measurement_height_m must be finite when configured")
    if maximum_measurement_height_m is not None and not isfinite(
        float(maximum_measurement_height_m)
    ):
        raise ValueError("maximum_measurement_height_m must be finite when configured")
    if (
        minimum_measurement_height_m is not None
        and maximum_measurement_height_m is not None
        and minimum_measurement_height_m > maximum_measurement_height_m
    ):
        raise ValueError("minimum measurement height cannot exceed maximum height")
    flags: list[str] = []
    if not all(isfinite(float(value)) for value in values):
        flags.append("non_finite_input")
    else:
        if wind_speed_m_s < 0.0:
            flags.append("negative_wind_speed")
        if measurement_height_m <= displacement_height_m:
            flags.append("measurement_not_above_displacement")
        if (
            minimum_measurement_height_m is not None
            and measurement_height_m < minimum_measurement_height_m
        ):
            flags.append("measurement_below_configured_minimum")
        if (
            maximum_measurement_height_m is not None
            and measurement_height_m > maximum_measurement_height_m
        ):
            flags.append("measurement_above_configured_maximum")
        if displacement_height_m >= 10.0:
            flags.append("target_not_above_displacement")
        if roughness_length_m <= 0.0:
            flags.append("nonpositive_roughness_length")
        elif measurement_height_m - displacement_height_m <= roughness_length_m:
            flags.append("measurement_not_above_roughness_sublayer")
        elif 10.0 - displacement_height_m <= roughness_length_m:
            flags.append("target_not_above_roughness_sublayer")
        if not neutral_stability:
            flags.append("neutral_stability_not_applicable")

    sensitivity: dict[float, float] = {}
    if not flags:
        converted = _log_profile_speed(
            float(wind_speed_m_s),
            float(measurement_height_m),
            10.0,
            float(roughness_length_m),
            float(displacement_height_m),
        )
        for candidate in sensitivity_roughness_lengths_m:
            candidate = float(candidate)
            if (
                isfinite(candidate)
                and candidate > 0.0
                and measurement_height_m - displacement_height_m > candidate
                and 10.0 - displacement_height_m > candidate
            ):
                sensitivity[candidate] = _log_profile_speed(
                    float(wind_speed_m_s),
                    float(measurement_height_m),
                    10.0,
                    candidate,
                    float(displacement_height_m),
                )
    else:
        converted = float("nan")

    if flags and strict:
        raise ValueError("wind-profile conversion is not applicable: " + ", ".join(flags))

    return WindProfileResult(
        wind_speed_10m_m_s=float(converted),
        applicable=not flags,
        flags=tuple(flags),
        measurement_height_m=float(measurement_height_m),
        roughness_length_m=float(roughness_length_m),
        displacement_height_m=float(displacement_height_m),
        sensitivity_by_roughness_m=sensitivity,
    )


# Backwards-readable name for callers that emphasise the profile assumption.
logarithmic_wind_profile_to_10m = convert_wind_to_10m


@dataclass(frozen=True)
class UTCIApplicability:
    """Scalar UTCI applicability assessment."""

    applicable: bool
    reasons: tuple[str, ...]


def assess_utci_applicability(
    air_temperature_c: float,
    mean_radiant_temperature_c: float,
    wind_speed_10m_m_s: float,
    relative_humidity_percent: float,
    *,
    limits: Mapping[str, Sequence[float]] | None = None,
) -> UTCIApplicability:
    """Explain whether one SI input tuple is inside the UTCI wrapper domain."""

    values = (
        air_temperature_c,
        mean_radiant_temperature_c,
        wind_speed_10m_m_s,
        relative_humidity_percent,
    )
    air_range, mrt_delta_range, wind_range, humidity_range = _resolve_utci_limits(limits)
    reasons: list[str] = []
    if not all(isfinite(float(value)) for value in values):
        reasons.append("non_finite_input")
    else:
        delta_mrt = mean_radiant_temperature_c - air_temperature_c
        if not air_range[0] <= air_temperature_c <= air_range[1]:
            reasons.append("air_temperature_out_of_range")
        if not mrt_delta_range[0] <= delta_mrt <= mrt_delta_range[1]:
            reasons.append("mrt_minus_air_temperature_out_of_range")
        if not wind_range[0] <= wind_speed_10m_m_s <= wind_range[1]:
            reasons.append("wind_10m_out_of_range")
        if not humidity_range[0] <= relative_humidity_percent <= humidity_range[1]:
            reasons.append("relative_humidity_out_of_range")
    return UTCIApplicability(applicable=not reasons, reasons=tuple(reasons))


def _utci_valid_mask(
    air_temperature_c: NDArray[np.float64],
    mean_radiant_temperature_c: NDArray[np.float64],
    wind_speed_10m_m_s: NDArray[np.float64],
    relative_humidity_percent: NDArray[np.float64],
    *,
    limits: Mapping[str, Sequence[float]] | None = None,
) -> NDArray[np.bool_]:
    air_range, mrt_delta_range, wind_range, humidity_range = _resolve_utci_limits(limits)
    finite = (
        np.isfinite(air_temperature_c)
        & np.isfinite(mean_radiant_temperature_c)
        & np.isfinite(wind_speed_10m_m_s)
        & np.isfinite(relative_humidity_percent)
    )
    delta = mean_radiant_temperature_c - air_temperature_c
    return (
        finite
        & (air_temperature_c >= air_range[0])
        & (air_temperature_c <= air_range[1])
        & (delta >= mrt_delta_range[0])
        & (delta <= mrt_delta_range[1])
        & (wind_speed_10m_m_s >= wind_range[0])
        & (wind_speed_10m_m_s <= wind_range[1])
        & (relative_humidity_percent >= humidity_range[0])
        & (relative_humidity_percent <= humidity_range[1])
    )


def calculate_utci(
    air_temperature_c: ArrayLike,
    mean_radiant_temperature_c: ArrayLike,
    wind_speed_10m_m_s: ArrayLike,
    relative_humidity_percent: ArrayLike,
    *,
    out_of_range: OutOfRangePolicy = "raise",
    limits: Mapping[str, Sequence[float]] | None = None,
) -> NumericResult:
    """Calculate unrounded UTCI in °C using pinned ``pythermalcomfort``.

    Parameters are SI values.  ``wind_speed_10m_m_s`` must already be expressed
    at the UTCI reference height; use :func:`convert_wind_to_10m` when needed.

    ``out_of_range`` is explicit:

    * ``"raise"`` (default) rejects any invalid tuple;
    * ``"nan"`` returns ``NaN`` at invalid positions;
    * ``"clip"`` clips finite values to the published approximation domain.

    The underlying call always uses ``units="SI"`` and ``round_output=False``.
    Because this wrapper applies the same limits itself and has already raised,
    masked, or clipped invalid tuples, it passes ``limit_inputs=False`` to avoid
    a second implementation silently changing the chosen policy.  Non-finite
    values cannot be clipped.
    """

    if out_of_range not in {"raise", "nan", "clip"}:
        raise ValueError("out_of_range must be one of: 'raise', 'nan', 'clip'")

    tdb, tr, wind, rh = np.broadcast_arrays(
        np.asarray(air_temperature_c, dtype=float),
        np.asarray(mean_radiant_temperature_c, dtype=float),
        np.asarray(wind_speed_10m_m_s, dtype=float),
        np.asarray(relative_humidity_percent, dtype=float),
    )
    air_range, mrt_delta_range, wind_range, humidity_range = _resolve_utci_limits(limits)
    valid = _utci_valid_mask(tdb, tr, wind, rh, limits=limits)

    if out_of_range == "raise" and not np.all(valid):
        invalid_count = int(np.size(valid) - np.count_nonzero(valid))
        if valid.ndim == 0:
            detail = ", ".join(
                assess_utci_applicability(
                    float(tdb),
                    float(tr),
                    float(wind),
                    float(rh),
                    limits=limits,
                ).reasons
            )
        else:
            detail = "inspect inputs against the documented UTCI domain"
        raise ValueError(
            f"{invalid_count} UTCI input tuple(s) are outside applicability limits: {detail}"
        )

    if out_of_range == "clip":
        finite = np.isfinite(tdb) & np.isfinite(tr) & np.isfinite(wind) & np.isfinite(rh)
        if not np.all(finite):
            raise ValueError("non-finite UTCI inputs cannot be clipped")
        clipped_tdb = np.clip(tdb, *air_range)
        clipped_delta = np.clip(tr - tdb, *mrt_delta_range)
        call_tdb = clipped_tdb
        call_tr = clipped_tdb + clipped_delta
        call_wind = np.clip(wind, *wind_range)
        call_rh = np.clip(rh, *humidity_range)
    elif out_of_range == "nan":
        # Supplying benign values at invalid positions avoids warnings inside the
        # polynomial; the positions are restored to NaN immediately afterwards.
        call_tdb = np.where(valid, tdb, 20.0)
        call_tr = np.where(valid, tr, 20.0)
        call_wind = np.where(valid, wind, 1.0)
        call_rh = np.where(valid, rh, 50.0)
    else:
        call_tdb, call_tr, call_wind, call_rh = tdb, tr, wind, rh

    try:
        from pythermalcomfort.models import utci as pythermalcomfort_utci
    except ImportError as exc:  # pragma: no cover - dependency check gives context
        raise RuntimeError(
            "pythermalcomfort is required for UTCI; install the pinned project dependencies"
        ) from exc

    result = pythermalcomfort_utci(
        tdb=call_tdb,
        tr=call_tr,
        v=call_wind,
        rh=call_rh,
        units="SI",
        limit_inputs=False,
        round_output=False,
    )
    if not hasattr(result, "utci"):
        raise TypeError(
            "the installed pythermalcomfort UTCI API is incompatible with the pinned project API"
        )
    values = np.asarray(result.utci, dtype=float)
    if out_of_range == "nan":
        values = np.where(valid, values, np.nan)
    return _return_scalar_if_scalar(np.asarray(values, dtype=float))


@dataclass(frozen=True)
class SensorUTCITarget:
    """One Stage 2 target derived from raw colocated sensor measurements."""

    calculated_mrt_c: float
    wind_speed_10m_m_s: float
    calculated_utci_c: float
    valid: bool
    flags: tuple[str, ...]
    wind_profile: WindProfileResult

    @property
    def applicable(self) -> bool:
        """Alias describing whether the derived value is training-eligible."""

        return self.valid


def _wind_profile_kwargs(
    wind_profile: Mapping[str, object] | None,
    *,
    roughness_length_m: float | None,
    displacement_height_m: float | None,
    neutral_stability: bool | None,
    minimum_measurement_height_m: float | None,
    maximum_measurement_height_m: float | None,
    sensitivity_roughness_lengths_m: Sequence[float] | None,
) -> tuple[float, float, bool, float | None, float | None, tuple[float, ...]]:
    """Resolve the checked scalar options accepted by ``convert_wind_to_10m``."""

    profile = dict(wind_profile or {})
    applicability_raw = profile.get("applicability", {})
    if applicability_raw is None:
        applicability: Mapping[str, object] = {}
    elif isinstance(applicability_raw, Mapping):
        applicability = applicability_raw
    else:
        raise ValueError("wind_profile.applicability must be a mapping")

    if profile.get("enabled", True) is not True:
        raise ValueError("Stage 2 target derivation requires wind_profile.enabled=true")
    method = str(profile.get("method", "neutral_logarithmic_profile"))
    if method != "neutral_logarithmic_profile":
        raise ValueError("wind_profile.method must be 'neutral_logarithmic_profile'")
    if profile.get("source_height_column", "measurement_height_m") != "measurement_height_m":
        raise ValueError(
            "Stage 2 wind_profile.source_height_column must be measurement_height_m"
        )
    target_height = float(profile.get("target_height_m", 10.0))
    if not isfinite(target_height) or target_height != 10.0:
        raise ValueError("UTCI target derivation requires wind_profile.target_height_m=10.0")
    if applicability.get("require_source_height_above_roughness_and_displacement", True) is not True:
        raise ValueError(
            "Stage 2 requires source height above roughness and displacement"
        )
    if applicability.get("assume_neutral_stability", True) is not True:
        raise ValueError("Stage 2 target derivation requires declared neutral stability")
    if applicability.get("reject_nonpositive_wind", False) is not False:
        raise ValueError(
            "Stage 2 wind policy must retain zero wind for explicit UTCI applicability flags"
        )
    if (
        applicability.get("invalid_value_policy", "flag_and_return_nan")
        != "flag_and_return_nan"
    ):
        raise ValueError(
            "Stage 2 wind invalid_value_policy must be flag_and_return_nan"
        )

    roughness_raw = (
        roughness_length_m
        if roughness_length_m is not None
        else profile.get("roughness_length_m")
    )
    if roughness_raw is None:
        raise ValueError(
            "roughness_length_m is required explicitly or in the wind_profile configuration"
        )
    displacement_raw = (
        displacement_height_m
        if displacement_height_m is not None
        else profile.get("displacement_height_m", 0.0)
    )
    neutral_raw = (
        neutral_stability
        if neutral_stability is not None
        else applicability.get("assume_neutral_stability", True)
    )
    minimum_raw = (
        minimum_measurement_height_m
        if minimum_measurement_height_m is not None
        else applicability.get("minimum_source_height_m")
    )
    maximum_raw = (
        maximum_measurement_height_m
        if maximum_measurement_height_m is not None
        else applicability.get("maximum_source_height_m")
    )
    sensitivity_raw = (
        sensitivity_roughness_lengths_m
        if sensitivity_roughness_lengths_m is not None
        else profile.get("roughness_sensitivity_values_m", ())
    )
    if isinstance(sensitivity_raw, (str, bytes)) or not isinstance(sensitivity_raw, Sequence):
        raise ValueError("wind-profile roughness sensitivity values must be a sequence")

    return (
        float(roughness_raw),
        float(displacement_raw),
        bool(neutral_raw),
        None if minimum_raw is None else float(minimum_raw),
        None if maximum_raw is None else float(maximum_raw),
        tuple(float(value) for value in sensitivity_raw),
    )


def derive_sensor_utci_target(
    measured_air_temperature_c: float,
    measured_relative_humidity_pct: float,
    measured_pedestrian_wind_speed_m_s: float,
    measured_globe_temperature_c: float,
    measurement_height_m: float,
    *,
    globe_diameter_m: float,
    globe_emissivity: float,
    globe_standard: str = PYTHERMALCOMFORT_GLOBE_MRT_STANDARD,
    wind_profile: Mapping[str, object] | None = None,
    roughness_length_m: float | None = None,
    displacement_height_m: float | None = None,
    neutral_stability: bool | None = None,
    minimum_measurement_height_m: float | None = None,
    maximum_measurement_height_m: float | None = None,
    sensitivity_roughness_lengths_m: Sequence[float] | None = None,
    utci_limits: Mapping[str, Sequence[float]] | None = None,
) -> SensorUTCITarget:
    """Derive one local UTCI target from raw sensor measurements.

    Globe MRT uses the frozen ISO method. Pedestrian wind is converted to the
    UTCI 10 m reference height using an explicit neutral logarithmic profile.
    Invalid sensor tuples return flags and ``NaN`` for the unavailable target;
    measured values are never clipped or silently repaired.
    """

    # Validate frozen method parameters independently of row validity.
    diameter = float(globe_diameter_m)
    emissivity = float(globe_emissivity)
    if str(globe_standard).strip().upper() != PYTHERMALCOMFORT_GLOBE_MRT_STANDARD:
        raise ValueError("globe_standard must be 'ISO' for Stage 2")
    if not isfinite(diameter) or diameter <= 0.0:
        raise ValueError("globe_diameter_m must be finite and greater than zero")
    if not isfinite(emissivity) or not 0.0 < emissivity <= 1.0:
        raise ValueError("globe_emissivity must be finite and in (0, 1]")
    _resolve_utci_limits(utci_limits)

    (
        resolved_roughness,
        resolved_displacement,
        resolved_neutral,
        resolved_minimum_height,
        resolved_maximum_height,
        resolved_sensitivity,
    ) = _wind_profile_kwargs(
        wind_profile,
        roughness_length_m=roughness_length_m,
        displacement_height_m=displacement_height_m,
        neutral_stability=neutral_stability,
        minimum_measurement_height_m=minimum_measurement_height_m,
        maximum_measurement_height_m=maximum_measurement_height_m,
        sensitivity_roughness_lengths_m=sensitivity_roughness_lengths_m,
    )

    air = float(measured_air_temperature_c)
    humidity = float(measured_relative_humidity_pct)
    pedestrian_wind = float(measured_pedestrian_wind_speed_m_s)
    globe = float(measured_globe_temperature_c)
    height = float(measurement_height_m)
    flags: list[str] = []
    if not isfinite(air):
        flags.append("sensor:non_finite_air_temperature")
    elif air <= -273.15:
        flags.append("sensor:air_temperature_at_or_below_absolute_zero")
    if not isfinite(humidity):
        flags.append("sensor:non_finite_relative_humidity")
    elif not 0.0 <= humidity <= 100.0:
        flags.append("sensor:relative_humidity_out_of_range")
    if not isfinite(pedestrian_wind):
        flags.append("sensor:non_finite_pedestrian_wind")
    elif pedestrian_wind < 0.0:
        flags.append("sensor:negative_pedestrian_wind")
    if not isfinite(globe):
        flags.append("sensor:non_finite_globe_temperature")
    elif globe <= -273.15:
        flags.append("sensor:globe_temperature_at_or_below_absolute_zero")

    converted_wind = convert_wind_to_10m(
        pedestrian_wind,
        height,
        resolved_roughness,
        displacement_height_m=resolved_displacement,
        neutral_stability=resolved_neutral,
        minimum_measurement_height_m=resolved_minimum_height,
        maximum_measurement_height_m=resolved_maximum_height,
        sensitivity_roughness_lengths_m=resolved_sensitivity,
    )
    flags.extend(f"wind_profile:{flag}" for flag in converted_wind.flags)

    mrt = float("nan")
    raw_mrt_inputs_valid = (
        isfinite(air)
        and air > -273.15
        and isfinite(globe)
        and globe > -273.15
        and isfinite(pedestrian_wind)
        and pedestrian_wind >= 0.0
    )
    if raw_mrt_inputs_valid:
        try:
            mrt = float(
                calculate_mean_radiant_temperature_from_globe(
                    globe,
                    air,
                    pedestrian_wind,
                    globe_diameter_m=diameter,
                    globe_emissivity=emissivity,
                    standard=globe_standard,
                )
            )
        except ValueError:
            flags.append("mrt:iso_calculation_non_finite")

    utci_value = float("nan")
    can_assess_utci = (
        isfinite(air)
        and isfinite(humidity)
        and isfinite(mrt)
        and converted_wind.applicable
    )
    if can_assess_utci:
        applicability = assess_utci_applicability(
            air,
            mrt,
            converted_wind.wind_speed_10m_m_s,
            humidity,
            limits=utci_limits,
        )
        flags.extend(f"utci:{reason}" for reason in applicability.reasons)
        if applicability.applicable:
            utci_value = float(
                calculate_utci(
                    air,
                    mrt,
                    converted_wind.wind_speed_10m_m_s,
                    humidity,
                    out_of_range="raise",
                    limits=utci_limits,
                )
            )

    unique_flags = tuple(dict.fromkeys(flags))
    valid = not unique_flags and isfinite(utci_value)
    return SensorUTCITarget(
        calculated_mrt_c=mrt,
        wind_speed_10m_m_s=converted_wind.wind_speed_10m_m_s,
        calculated_utci_c=utci_value,
        valid=valid,
        flags=unique_flags,
        wind_profile=converted_wind,
    )


# Concise public alias while retaining the unit in the canonical name above.
utci_si = calculate_utci


__all__ = [
    "PYTHERMALCOMFORT_GLOBE_MRT_STANDARD",
    "RELATIVE_HUMIDITY_RANGE_PERCENT",
    "SensorUTCITarget",
    "UTCI_AIR_TEMPERATURE_RANGE_C",
    "UTCI_MRT_DELTA_RANGE_C",
    "UTCI_WIND_10M_RANGE_M_S",
    "UTCIApplicability",
    "WindProfileResult",
    "assess_utci_applicability",
    "calculate_mean_radiant_temperature_from_globe",
    "calculate_utci",
    "convert_wind_to_10m",
    "derive_sensor_utci_target",
    "invert_wind_log_adjustment",
    "logarithmic_wind_profile_to_10m",
    "mean_radiant_temperature_from_globe",
    "relative_humidity_to_vapor_pressure",
    "rh_from_vapor_pressure",
    "saturation_vapor_pressure_kpa",
    "stable_wind_adjustment",
    "utci_si",
    "vapor_pressure_from_rh",
    "vapor_pressure_to_relative_humidity",
    "wind_log_adjustment",
]
