"""Minimal null-boundary tail test for the exterior-supported SdS family.

The experiment compares the dipole field at the cosmological horizon with
uniform Schwarzschild--de Sitter and with an independently evolved
Schwarzschild field at future null infinity.  It tests, without assuming the
outcome, whether an intermediate Schwarzschild Price interval survives the
exterior modification and how the later signal reflects the transition layer
and cosmological asymptotics.

The frozen uniform-SdS and Schwarzschild controls under
``results/large_l_tail`` are read-only.  This module evolves only the missing
exterior-supported cases and writes them to an isolated output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct

from .exterior_sds_model import (
    ExteriorSdSParameters,
    areal_radius,
    background_audit,
    chebyshev_angle,
    compact_radius,
    compactification_scale,
    rescaled_scalar_potential,
    retarded_time_offset,
    transition_compact_radii,
)
from .large_l_tail import (
    LocalFitSettings,
    _interpolate,
    cosmological_rate,
    effective_rates,
    rms_envelope,
)
from .reproducibility import reproducibility_metadata
from .sds_model import ArealVelocityBumpInitialData
from .sds_result import SdSSimulationResult, load_sds_result


OUTPUT_ROOT = Path("results/exterior_tail_feasibility_v2")
CONTROL_ROOT = Path("results/large_l_tail/raw")
LENGTH = 640.0
ELL = 1
REFERENCE_RADIUS = 4.0
FINITE_OBSERVERS = (8.0, 16.0)
EXTERIOR_RESOLUTIONS = (1536, 2048)
CONTROL_RESOLUTIONS = (1536, 2048, 3072)
TIMESTEP = 0.0025
END_U = 1000.0
SIGNAL_DT = 0.05
PRICE_TARGET = 3.0
INITIAL_DATA = ArealVelocityBumpInitialData(
    center_radius=6.0,
    support_half_width=3.0,
    amplitude=1.0,
)


def model() -> ExteriorSdSParameters:
    """Return the fixed background used by every exterior run."""

    return ExteriorSdSParameters(
        mass=1.0,
        cosmological_length=LENGTH,
        ell=ELL,
    )


def observer_coordinates() -> tuple[float, ...]:
    """Return exact interpolation coordinates at 8M, 16M, and Hc+."""

    parameters = model()
    finite = compact_radius(np.asarray(FINITE_OBSERVERS), parameters)
    return tuple(float(value) for value in finite) + (1.0,)


def archive_path(output_dir: Path, resolution: int) -> Path:
    """Return the isolated archive path for one exterior resolution."""

    if resolution not in EXTERIOR_RESOLUTIONS:
        raise ValueError(
            f"Resolution must be one of {EXTERIOR_RESOLUTIONS}."
        )
    return (
        Path(output_dir)
        / "raw"
        / f"exterior_factored_L{LENGTH:g}_N{resolution}_dt0p0025.npz"
    )


def spectral_preflight(
    resolution: int, *, dense_count: int = 20_001
) -> dict[str, float | int | bool]:
    """Check that the transition potential is spectrally represented."""

    if resolution not in EXTERIOR_RESOLUTIONS:
        raise ValueError(
            f"Resolution must be one of {EXTERIOR_RESOLUTIONS}."
        )
    parameters = model()
    rho0, rho1 = transition_compact_radii(parameters)
    theta0 = float(chebyshev_angle(np.array(rho0)))
    theta1 = float(chebyshev_angle(np.array(rho1)))

    theta_nodes = np.pi * (np.arange(resolution) + 0.5) / resolution
    x_nodes = np.cos(theta_nodes)
    rho_nodes = 0.5 * (1.0 + x_nodes)
    scale = 2.0 * parameters.mass / compactification_scale(parameters)
    represented_nodes = scale * rescaled_scalar_potential(rho_nodes, parameters)
    coefficients = dct(represented_nodes, type=2) / resolution
    coefficients[0] *= 0.5

    theta_dense = np.linspace(theta1, theta0, dense_count)
    x_dense = np.cos(theta_dense)
    rho_dense = 0.5 * (1.0 + x_dense)
    analytic = scale * rescaled_scalar_potential(rho_dense, parameters)
    represented = chebval(x_dense, coefficients)
    maximum_error = float(np.max(np.abs(represented - analytic)))
    analytic_scale = float(np.max(np.abs(analytic)))
    relative_error = maximum_error / analytic_scale
    transition_nodes = int(np.count_nonzero((rho_nodes > rho0) & (rho_nodes < rho1)))
    cap_nodes = int(np.count_nonzero(rho_nodes >= rho1))
    passed = bool(
        transition_nodes >= 24
        and cap_nodes >= 24
        and relative_error < 1.0e-4
        and np.all(np.isfinite(represented))
    )
    return {
        "length_over_M": LENGTH,
        "ell": ELL,
        "resolution": resolution,
        "transition_nodes": transition_nodes,
        "outer_cap_nodes": cap_nodes,
        "transition_inner_radius_over_M": float(areal_radius(np.array(rho0), parameters)),
        "transition_outer_radius_over_M": float(areal_radius(np.array(rho1), parameters)),
        "maximum_absolute_potential_error": maximum_error,
        "maximum_relative_potential_error": relative_error,
        "passed": passed,
    }


def _reserve(destination: Path) -> tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    reservation = destination.with_suffix(".running")
    checkpoint = destination.with_suffix(".checkpoint.npz")
    with reservation.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(destination.stem + "\n")
    return reservation, checkpoint


def run_case(
    output_dir: Path,
    resolution: int,
    *,
    resume_interrupted: bool = False,
) -> Path:
    """Evolve and atomically save one exterior-supported tail case."""

    destination = archive_path(output_dir, resolution)
    reservation = destination.with_suffix(".running")
    checkpoint = destination.with_suffix(".checkpoint.npz")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    if resume_interrupted:
        if not reservation.exists() or not checkpoint.exists():
            raise FileNotFoundError(
                "A resumable case requires both its reservation and checkpoint."
            )
    else:
        reservation, checkpoint = _reserve(destination)

    preflight = spectral_preflight(resolution)
    if not preflight["passed"]:
        raise ValueError(f"Spectral preflight failed: {preflight}")
    parameters = model()
    audit = background_audit(parameters)
    required = (
        "finite_coefficients",
        "positive_interior_lapse",
        "spacelike_bridge_interior",
        "nonnegative_scalar_potential",
    )
    if not all(audit[key] for key in required):
        raise ValueError(f"Background audit failed: {audit}")

    # Dedalus is imported only for the simulation command.
    from .sds_solver import SdSNumericalParameters, run_exterior_sds_simulation

    offset = float(retarded_time_offset(parameters, REFERENCE_RADIUS))
    numerical = SdSNumericalParameters(
        resolution=resolution,
        timestep=TIMESTEP,
        end_time=END_U + offset,
        signal_dt=SIGNAL_DT,
        snapshot_dt=50.0,
        observers=observer_coordinates(),
        timestepper="RK222",
        bridge="minimal",
        dealias=1.5,
    )
    result = run_exterior_sds_simulation(
        parameters,
        INITIAL_DATA,
        numerical,
        checkpoint_path=checkpoint,
        checkpoint_dt=250.0,
        explicit_potential=True,
        endpoint_factored_characteristic_variables=True,
        characteristic_constraint_damping=1.0 / parameters.mass,
    )
    result.metadata["retarded_time_offset"] = {
        "q": offset,
        "definition": "lim_(r->r_c)(h_chi+r_*chi)",
        "evaluation": "endpoint-safe numerical quadrature",
    }
    result.metadata["exterior_tail_feasibility"] = {
        "purpose": (
            "test whether the exterior-supported regulator reproduces a "
            "Schwarzschild Price interval and determine how it departs"
        ),
        "length_over_M": LENGTH,
        "ell": ELL,
        "retarded_end_time_over_M": END_U,
        "finite_observers_over_M": list(FINITE_OBSERVERS),
        "outer_observer": "future cosmological horizon",
        "comparison_outer_observer": "Schwarzschild future null infinity",
        "time_translation_fitted": False,
        "amplitude_rescaling_fitted": False,
        "explicit_potential": True,
        "state_variables": "endpoint-factored characteristic variables u, H, J",
    }
    result.metadata["background_audit"] = audit
    result.metadata["spectral_preflight"] = preflight
    result.metadata["simulation_provenance"] = reproducibility_metadata()

    temporary = destination.with_suffix(".incomplete.npz")
    try:
        result.save(temporary)
        temporary.rename(destination)
        checkpoint.unlink(missing_ok=True)
        reservation.unlink()
    except BaseException:
        # Keep the reservation and checkpoint for inspection or explicit resume.
        raise
    return destination


def _retarded_series(
    result: SdSSimulationResult, observer: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    offset = float(result.metadata["retarded_time_offset"]["q"])
    times = result.signal_times - offset
    inside = times <= END_U + 0.5 * SIGNAL_DT
    return times[inside], result.signals[inside, observer]


def _control_path(background: str, resolution: int) -> Path:
    if background == "schwarzschild":
        stem = f"final_schwarzschild_for_L640_N{resolution}_dt0p0025.npz"
    elif background == "uniform":
        stem = f"final_sds_L640_N{resolution}_dt0p0025.npz"
    else:
        raise ValueError(background)
    return CONTROL_ROOT / stem


def _load_ladder(
    output_dir: Path,
    background: str,
    *,
    resolutions: tuple[int, ...],
) -> dict[int, SdSSimulationResult]:
    paths = {
        resolution: (
            archive_path(output_dir, resolution)
            if background == "exterior"
            else _control_path(background, resolution)
        )
        for resolution in resolutions
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing archives: " + ", ".join(map(str, missing)))
    return {resolution: load_sds_result(path) for resolution, path in paths.items()}


def _refinement_floor(
    ladder: dict[int, SdSSimulationResult], settings: LocalFitSettings
) -> dict[str, np.ndarray]:
    """Return the higher-minus-lower-resolution envelope difference.

    The exterior calculation has only two resolutions.  Their difference is
    therefore an observed refinement change, not an estimate of convergence
    order or of the continuum error.
    """

    series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for resolution, result in ladder.items():
        times, signal = _retarded_series(result)
        amplitude = rms_envelope(
            times,
            signal,
            settings.envelope_width,
            floor_multiplier=0.0,
        )
        series[resolution] = times, amplitude
    ordered = sorted(series)
    if len(ordered) < 2:
        raise ValueError("A numerical floor requires at least two resolutions.")
    higher_resolution = ordered[-1]
    lower_resolution = ordered[-2]
    times, higher = series[higher_resolution]
    lower = _interpolate(*series[lower_resolution], times)
    refinement_difference = np.abs(higher - lower)
    return {
        "times": times,
        "amplitude": higher,
        "floor": refinement_difference,
        "higher_resolution": np.array(higher_resolution),
        "lower_resolution": np.array(lower_resolution),
    }


def _diagnostics(
    result: SdSSimulationResult,
    settings: LocalFitSettings,
    *,
    measured_floor: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    times, signal = _retarded_series(result)
    amplitude, power, normalized_gamma = effective_rates(
        times,
        signal,
        settings,
        kappa=cosmological_rate(LENGTH),
        measured_floor=measured_floor,
    )
    return {
        "times": times,
        "signal": signal,
        "amplitude": amplitude,
        "power": power,
        "normalized_gamma": normalized_gamma,
    }


def _longest_interval(
    times: np.ndarray, mask: np.ndarray
) -> tuple[float, float] | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    group = max(groups, key=lambda item: times[item[-1]] - times[item[0]])
    return float(times[group[0]]), float(times[group[-1]])


def _qualified_intervals(
    times: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_duration: float,
) -> list[tuple[float, float]]:
    """Return contiguous intervals whose duration reaches the stated target."""

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [
        (float(times[group[0]]), float(times[group[-1]]))
        for group in groups
        if times[group[-1]] - times[group[0]] >= minimum_duration
    ]


def _diagnostic_supported_to(
    times: np.ndarray,
    fine: np.ndarray,
    medium_times: np.ndarray,
    medium: np.ndarray,
    *,
    tolerance: float = 0.15,
    after: float = 120.0,
    persistence: float = 10.0,
) -> float | None:
    """Return the first persistent two-grid failure of a local diagnostic.

    A waveform can pass through a cancellation and later become large enough
    to clear an amplitude floor again.  That later re-entry does not repair the
    local derivative across the cancellation, so the plotted rate stops at the
    first disagreement lasting ``persistence`` rather than silently returning.
    """

    medium_on_fine = _interpolate(medium_times, medium, times)
    valid = np.isfinite(fine) & np.isfinite(medium_on_fine)
    bad = (times >= after) & (~valid | (np.abs(fine - medium_on_fine) > tolerance))
    indices = np.flatnonzero(bad)
    if indices.size == 0:
        finite = np.flatnonzero(valid & (times >= after))
        return float(times[finite[-1]]) if finite.size else None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    for group in groups:
        if times[group[-1]] - times[group[0]] >= persistence:
            return float(times[max(group[0] - 1, 0)])
    finite = np.flatnonzero(valid & (times >= after))
    return float(times[finite[-1]]) if finite.size else None


PRICE_BAND_RELATIVE_TOLERANCE = 0.10
PRICE_INTERVAL_TARGET = 40.0
DIAGNOSTIC_REFINEMENT_TOLERANCE = 0.10
FIXED_WINDOWS = (
    (160.0, 220.0),
    (220.0, 300.0),
    (300.0, 500.0),
    (500.0, 750.0),
    (750.0, 950.0),
)


def _comparison_mask(
    higher: dict[str, np.ndarray],
    lower: dict[str, np.ndarray],
    quantity: str,
    *,
    tolerance: float = DIAGNOSTIC_REFINEMENT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower-resolution diagnostic on the higher grid and support."""

    times = higher["times"]
    lower_on_higher = _interpolate(lower["times"], lower[quantity], times)
    supported = (
        (times >= 100.0)
        & np.isfinite(higher[quantity])
        & np.isfinite(lower_on_higher)
        & (np.abs(higher[quantity] - lower_on_higher) <= tolerance)
    )
    return lower_on_higher, supported


def _outcome_metrics(
    higher: dict[str, np.ndarray],
    lower: dict[str, np.ndarray],
    *,
    settings: LocalFitSettings,
) -> dict:
    """Apply outcome-neutral Price and cosmological-rate criteria."""

    times = higher["times"]
    lower_power, power_supported = _comparison_mask(higher, lower, "power")
    power_supported_to = _diagnostic_supported_to(
        times,
        higher["power"],
        times,
        lower_power,
        tolerance=DIAGNOSTIC_REFINEMENT_TOLERANCE,
        after=100.0,
        persistence=10.0,
    )
    if power_supported_to is not None:
        power_supported &= times <= power_supported_to
    price_half_width = PRICE_BAND_RELATIVE_TOLERANCE * PRICE_TARGET
    price_mask = power_supported & (
        np.abs(higher["power"] - PRICE_TARGET) <= price_half_width
    )
    price_interval = _longest_interval(times, price_mask)
    price_duration = (
        price_interval[1] - price_interval[0]
        if price_interval is not None
        else 0.0
    )
    price_objective_met = price_duration >= PRICE_INTERVAL_TARGET

    # A departure is called persistent only after the qualifying Price-like
    # interval (if one exists) and only if it lasts another 40M on samples for
    # which the two resolutions still agree at the diagnostic level.
    departure_after = (
        price_interval[1] if price_objective_met and price_interval else 100.0
    )
    outside_price_band = (
        power_supported
        & (times >= departure_after)
        & (np.abs(higher["power"] - PRICE_TARGET) > price_half_width)
    )
    departure_intervals = _qualified_intervals(
        times,
        outside_price_band,
        minimum_duration=PRICE_INTERVAL_TARGET,
    )
    departure_interval = departure_intervals[0] if departure_intervals else None
    if departure_interval is None:
        departure_direction = None
        departure_median = None
    else:
        inside = (
            (times >= departure_interval[0])
            & (times <= departure_interval[1])
            & outside_price_band
        )
        departure_median = float(np.median(higher["power"][inside]))
        departure_direction = (
            "faster_than_U^-3" if departure_median > PRICE_TARGET
            else "slower_than_U^-3"
        )

    # gamma_eff/kappa is plotted as a diagnostic.  It supports an exponential
    # statement only if the two resolutions agree, the result remains within
    # ten percent of the ell=1 SdS target, and that agreement lasts for at
    # least one full semilog fit width.
    lower_gamma, gamma_supported = _comparison_mask(
        higher, lower, "normalized_gamma"
    )
    gamma_supported_to = _diagnostic_supported_to(
        times,
        higher["normalized_gamma"],
        times,
        lower_gamma,
        tolerance=DIAGNOSTIC_REFINEMENT_TOLERANCE,
        after=100.0,
        persistence=10.0,
    )
    if gamma_supported_to is not None:
        gamma_supported &= times <= gamma_supported_to
    gamma_target = float(ELL)
    gamma_claim_duration = max(
        PRICE_INTERVAL_TARGET,
        settings.exponential_scaled_window / cosmological_rate(LENGTH),
    )
    gamma_target_mask = gamma_supported & (
        np.abs(higher["normalized_gamma"] - gamma_target)
        <= PRICE_BAND_RELATIVE_TOLERANCE * gamma_target
    )
    gamma_intervals = _qualified_intervals(
        times,
        gamma_target_mask,
        minimum_duration=gamma_claim_duration,
    )
    gamma_interval = gamma_intervals[0] if gamma_intervals else None

    return {
        "price_band": [
            PRICE_TARGET - price_half_width,
            PRICE_TARGET + price_half_width,
        ],
        "longest_price_band_interval": price_interval,
        "longest_price_band_duration_over_M": float(price_duration),
        "price_interval_target_over_M": PRICE_INTERVAL_TARGET,
        "price_interval_objective_met": bool(price_objective_met),
        "power_refinement_supported_to_U_over_M": power_supported_to,
        "persistent_departure_interval": departure_interval,
        "persistent_departure_direction": departure_direction,
        "persistent_departure_median_p_eff": departure_median,
        "exponential_target_gamma_over_kappa": gamma_target,
        "exponential_minimum_interval_over_M": float(gamma_claim_duration),
        "exponential_target_interval": gamma_interval,
        "exponential_claim_resolved": bool(gamma_interval is not None),
        "gamma_refinement_supported_to_U_over_M": gamma_supported_to,
        "power_lower_on_higher": lower_power,
        "power_supported": power_supported,
        "gamma_lower_on_higher": lower_gamma,
        "gamma_supported": gamma_supported,
    }


def _constraint_probe_rows(
    ladders: dict[str, dict[int, SdSSimulationResult]],
) -> list[dict]:
    """Return global and pointwise constraint diagnostics through U=1000M."""

    rows: list[dict] = []
    for background, ladder in ladders.items():
        for resolution, result in sorted(ladder.items()):
            offset = float(result.metadata["retarded_time_offset"]["q"])
            times = np.asarray(result.snapshot_times) - offset
            in_record = (times >= 0.0) & (times <= END_U)
            late = in_record & (times >= 100.0)
            for label, values in (
                ("global_linf", result.constraint_linf),
                ("global_l2", result.constraint_l2),
            ):
                values = np.asarray(values)
                rows.append(
                    {
                        "background": background,
                        "resolution": resolution,
                        "probe": label,
                        "rho": None,
                        "maximum_abs_through_U1000": (
                            float(np.max(np.abs(values[in_record])))
                            if np.any(in_record)
                            else None
                        ),
                        "maximum_abs_after_U100": (
                            float(np.max(np.abs(values[late])))
                            if np.any(late)
                            else None
                        ),
                        "last_abs_in_record": (
                            float(np.abs(values[np.flatnonzero(in_record)[-1]]))
                            if np.any(in_record)
                            else None
                        ),
                    }
                )

            constraint = result.metadata.get("constraint", {})
            probe_values = np.asarray(constraint.get("probe_values", []))
            probe_labels = constraint.get("probe_labels", [])
            probe_rho = constraint.get("probe_rho", [])
            if (
                probe_values.ndim == 2
                and probe_values.shape[0] == times.size
                and probe_values.shape[1] == len(probe_labels)
            ):
                for index, label in enumerate(probe_labels):
                    values = probe_values[:, index]
                    rows.append(
                        {
                            "background": background,
                            "resolution": resolution,
                            "probe": label,
                            "rho": float(probe_rho[index]),
                            "maximum_abs_through_U1000": (
                                float(np.max(np.abs(values[in_record])))
                                if np.any(in_record)
                                else None
                            ),
                            "maximum_abs_after_U100": (
                                float(np.max(np.abs(values[late])))
                                if np.any(late)
                                else None
                            ),
                            "last_abs_in_record": (
                                float(np.abs(values[np.flatnonzero(in_record)[-1]]))
                                if np.any(in_record)
                                else None
                            ),
                        }
                    )
    return rows


def analyze(output_dir: Path) -> dict:
    """Create one decay figure and the numerical audit tables behind it."""

    output_dir = Path(output_dir)
    settings = LocalFitSettings(
        envelope_width=10.0,
        price_window=40.0,
        floor_multiplier=100.0,
    )
    ladders = {
        "schwarzschild": _load_ladder(
            output_dir,
            "schwarzschild",
            resolutions=CONTROL_RESOLUTIONS,
        ),
        "uniform": _load_ladder(
            output_dir,
            "uniform",
            resolutions=CONTROL_RESOLUTIONS,
        ),
        "exterior": _load_ladder(
            output_dir,
            "exterior",
            resolutions=EXTERIOR_RESOLUTIONS,
        ),
    }
    floors = {
        background: _refinement_floor(ladder, settings)
        for background, ladder in ladders.items()
    }

    diagnostics: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    primary: dict[str, dict] = {}
    for background, ladder in ladders.items():
        ordered = sorted(ladder)
        higher_resolution = ordered[-1]
        lower_resolution = ordered[-2]
        diagnostics[background] = {}
        for resolution in (lower_resolution, higher_resolution):
            diagnostics[background][resolution] = _diagnostics(
                ladder[resolution],
                settings,
                measured_floor=(
                    floors[background]["floor"]
                    if resolution == higher_resolution
                    else None
                ),
            )
        primary[background] = _outcome_metrics(
            diagnostics[background][higher_resolution],
            diagnostics[background][lower_resolution],
            settings=settings,
        )

    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # Fixed, declared windows make the numerical comparison reproducible and
    # keep the conclusion from depending on a visually selected interval.
    window_rows: list[dict] = []
    for background, ladder in ladders.items():
        ordered = sorted(ladders[background])
        higher_resolution = ordered[-1]
        lower_resolution = ordered[-2]
        higher = diagnostics[background][higher_resolution]
        lower = diagnostics[background][lower_resolution]
        times = higher["times"]
        lower_amplitude = _interpolate(
            lower["times"], lower["amplitude"], times
        )
        lower_power = _interpolate(lower["times"], lower["power"], times)
        lower_gamma = _interpolate(
            lower["times"], lower["normalized_gamma"], times
        )
        for left, right in FIXED_WINDOWS:
            window = (times >= left) & (times <= right)
            amplitude_inside = (
                window
                & np.isfinite(higher["amplitude"])
                & np.isfinite(lower_amplitude)
                & (higher["amplitude"] > 0.0)
            )
            power_inside = (
                window
                & np.isfinite(higher["power"])
                & np.isfinite(lower_power)
            )
            gamma_inside = (
                window
                & np.isfinite(higher["normalized_gamma"])
                & np.isfinite(lower_gamma)
            )
            p_difference = np.abs(higher["power"] - lower_power)
            gamma_difference = np.abs(higher["normalized_gamma"] - lower_gamma)
            power_agreement = (
                power_inside
                & (p_difference <= DIAGNOSTIC_REFINEMENT_TOLERANCE)
            )
            count_in_window = int(np.count_nonzero(window))
            window_rows.append(
                {
                    "background": background,
                    "window_start_U_over_M": left,
                    "window_end_U_over_M": right,
                    "higher_resolution": higher_resolution,
                    "lower_resolution": lower_resolution,
                    "median_rms_envelope": (
                        float(np.median(higher["amplitude"][amplitude_inside]))
                        if np.any(amplitude_inside)
                        else None
                    ),
                    "maximum_relative_envelope_refinement_difference": (
                        float(
                            np.max(
                                np.abs(
                                    higher["amplitude"][amplitude_inside]
                                    - lower_amplitude[amplitude_inside]
                                )
                                / higher["amplitude"][amplitude_inside]
                            )
                        )
                        if np.any(amplitude_inside)
                        else None
                    ),
                    "median_p_eff": (
                        float(np.median(higher["power"][power_inside]))
                        if np.any(power_inside)
                        else None
                    ),
                    "p_eff_5th_percentile": (
                        float(np.percentile(higher["power"][power_inside], 5.0))
                        if np.any(power_inside)
                        else None
                    ),
                    "p_eff_95th_percentile": (
                        float(np.percentile(higher["power"][power_inside], 95.0))
                        if np.any(power_inside)
                        else None
                    ),
                    "maximum_abs_p_eff_refinement_difference": (
                        float(np.max(p_difference[power_inside]))
                        if np.any(power_inside)
                        else None
                    ),
                    "fraction_of_window_with_p_eff_refinement_agreement": (
                        float(np.count_nonzero(power_agreement) / count_in_window)
                        if count_in_window
                        else None
                    ),
                    "median_gamma_eff_over_kappa": (
                        float(
                            np.median(
                                higher["normalized_gamma"][gamma_inside]
                            )
                        )
                        if np.any(gamma_inside)
                        else None
                    ),
                    "maximum_abs_gamma_over_kappa_refinement_difference": (
                        float(np.max(gamma_difference[gamma_inside]))
                        if np.any(gamma_inside)
                        else None
                    ),
                    "price_target": PRICE_TARGET,
                }
            )
    windows_path = tables / "outer_decay_fixed_windows.csv"
    with windows_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)

    # Vary the two operations that define p_eff.  These rows are a sensitivity
    # audit, not extra trials from which the most favorable estimator is chosen.
    sensitivity_rows: list[dict] = []
    floor_cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for envelope_width in (5.0, 10.0, 20.0):
        for background, ladder in ladders.items():
            envelope_settings = LocalFitSettings(
                envelope_width=envelope_width,
                price_window=settings.price_window,
                exponential_scaled_window=settings.exponential_scaled_window,
                floor_multiplier=settings.floor_multiplier,
            )
            floor_cache[(background, envelope_width)] = _refinement_floor(
                ladder, envelope_settings
            )
        for price_window in (30.0, 40.0, 60.0):
            varied = LocalFitSettings(
                envelope_width=envelope_width,
                price_window=price_window,
                exponential_scaled_window=settings.exponential_scaled_window,
                floor_multiplier=settings.floor_multiplier,
            )
            for background, ladder in ladders.items():
                ordered = sorted(ladder)
                higher_resolution = ordered[-1]
                lower_resolution = ordered[-2]
                higher = _diagnostics(
                    ladder[higher_resolution],
                    varied,
                    measured_floor=floor_cache[
                        (background, envelope_width)
                    ]["floor"],
                )
                lower = _diagnostics(ladder[lower_resolution], varied)
                metrics = _outcome_metrics(higher, lower, settings=varied)
                interval = metrics["longest_price_band_interval"]
                departure = metrics["persistent_departure_interval"]
                exponential = metrics["exponential_target_interval"]
                sensitivity_rows.append(
                    {
                        "background": background,
                        "rms_envelope_width_over_M": envelope_width,
                        "p_eff_fit_window_over_M": price_window,
                        "price_band_start_U_over_M": (
                            interval[0] if interval is not None else None
                        ),
                        "price_band_end_U_over_M": (
                            interval[1] if interval is not None else None
                        ),
                        "price_band_duration_over_M": metrics[
                            "longest_price_band_duration_over_M"
                        ],
                        "price_interval_objective_met": metrics[
                            "price_interval_objective_met"
                        ],
                        "persistent_departure_start_U_over_M": (
                            departure[0] if departure is not None else None
                        ),
                        "persistent_departure_end_U_over_M": (
                            departure[1] if departure is not None else None
                        ),
                        "persistent_departure_direction": metrics[
                            "persistent_departure_direction"
                        ],
                        "power_refinement_supported_to_U_over_M": metrics[
                            "power_refinement_supported_to_U_over_M"
                        ],
                        "exponential_target_start_U_over_M": (
                            exponential[0] if exponential is not None else None
                        ),
                        "exponential_target_end_U_over_M": (
                            exponential[1] if exponential is not None else None
                        ),
                        "exponential_claim_resolved": metrics[
                            "exponential_claim_resolved"
                        ],
                    }
                )
    sensitivity_path = tables / "tail_estimator_sensitivity.csv"
    with sensitivity_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    constraint_rows = _constraint_probe_rows(ladders)
    constraints_path = tables / "constraint_probe_audit.csv"
    with constraints_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(constraint_rows[0]))
        writer.writeheader()
        writer.writerows(constraint_rows)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    styles = {
        "schwarzschild": ("black", "--", r"Schwarzschild $\mathcal{I}^{+}$"),
        "uniform": ("#D55E00", "-", r"uniform SdS $\mathcal{H}_{c}^{+}$"),
        "exterior": ("#0072B2", "-", r"exterior SdS $\mathcal{H}_{c}^{+}$"),
    }
    for background, (color, linestyle, label) in styles.items():
        ordered = sorted(ladders[background])
        higher = diagnostics[background][ordered[-1]]
        outcome = primary[background]
        axes[0].plot(
            higher["times"],
            higher["amplitude"],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=label,
        )
        axes[1].plot(
            higher["times"],
            np.where(outcome["power_supported"], higher["power"], np.nan),
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )
        axes[2].plot(
            higher["times"],
            np.where(
                outcome["gamma_supported"],
                higher["normalized_gamma"],
                np.nan,
            ),
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )

    exterior_ordered = sorted(ladders["exterior"])
    exterior_lower = diagnostics["exterior"][exterior_ordered[-2]]
    axes[0].plot(
        exterior_lower["times"],
        exterior_lower["amplitude"],
        color="#0072B2",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
        label=fr"exterior lower resolution $N={exterior_ordered[-2]}$",
    )
    axes[1].plot(
        diagnostics["exterior"][exterior_ordered[-1]]["times"],
        np.where(
            primary["exterior"]["power_supported"],
            primary["exterior"]["power_lower_on_higher"],
            np.nan,
        ),
        color="#0072B2",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
    )
    axes[2].plot(
        diagnostics["exterior"][exterior_ordered[-1]]["times"],
        np.where(
            primary["exterior"]["gamma_supported"],
            primary["exterior"]["gamma_lower_on_higher"],
            np.nan,
        ),
        color="#0072B2",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
    )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"RMS envelope $A$")
    axes[0].legend(frameon=False, loc="best", fontsize=8.5)
    axes[0].set_title(
        r"Dipole decay at the outer boundary, $L/M=640$"
    )
    axes[1].axhspan(2.7, 3.3, color="#009E73", alpha=0.12)
    axes[1].axhline(PRICE_TARGET, color="#009E73", linewidth=0.9)
    axes[1].set_ylim(0.0, 8.0)
    axes[1].set_ylabel(r"$p_{\rm eff}=-d\ln A/d\ln U$")
    axes[2].axhspan(0.9, 1.1, color="#CC79A7", alpha=0.12)
    axes[2].axhline(float(ELL), color="#CC79A7", linewidth=0.9)
    axes[2].set_ylim(-1.0, 12.0)
    axes[2].set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
    axes[2].set_xlim(80.0, END_U)
    axes[2].set_xlabel(r"geometric retarded time $U/M$")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    png = output_dir / "outer_power_index_comparison.png"
    pdf = output_dir / "outer_power_index_comparison.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    def public_outcome(metrics: dict) -> dict:
        return {
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "power_lower_on_higher",
                "power_supported",
                "gamma_lower_on_higher",
                "gamma_supported",
            }
        }

    summary = {
        "question": (
            "Does exterior support recover a short Schwarzschild p=3 interval "
            "at the cosmological horizon, or does the asymptotic modification "
            "remain visible in the tail?"
        ),
        "length_over_M": LENGTH,
        "ell": ELL,
        "outer_price_target": PRICE_TARGET,
        "retarded_time_range_over_M": [0.0, END_U],
        "primary_rms_envelope_width_over_M": settings.envelope_width,
        "primary_price_fit_window_over_M": settings.price_window,
        "price_band_relative_tolerance": PRICE_BAND_RELATIVE_TOLERANCE,
        "price_interval_target_over_M": PRICE_INTERVAL_TARGET,
        "refinement_stability_threshold_abs_diagnostic": (
            DIAGNOSTIC_REFINEMENT_TOLERANCE
        ),
        "outcomes": {
            background: public_outcome(metrics)
            for background, metrics in primary.items()
        },
        "exterior_resolutions": list(EXTERIOR_RESOLUTIONS),
        "control_resolutions": list(CONTROL_RESOLUTIONS),
        "exterior_two_grid_interpretation": (
            "The N=1536 versus N=2048 result is an observed refinement "
            "difference; two grids do not establish a convergence order."
        ),
        "exponential_interpretation_rule": (
            "gamma_eff/kappa_c is diagnostic unless two-grid agreement and "
            "10 percent agreement with the ell=1 target persist for at least "
            "one complete semilog fit width."
        ),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "fixed_windows_table": str(windows_path),
        "estimator_sensitivity_table": str(sensitivity_path),
        "constraint_probe_table": str(constraints_path),
        "control_archives_reused_read_only": True,
    }
    summary_path = tables / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("list", "preflight", "run", "analyze")
    )
    parser.add_argument("resolutions", nargs="*", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume-interrupted", action="store_true")
    arguments = parser.parse_args()

    selected = tuple(arguments.resolutions) or EXTERIOR_RESOLUTIONS
    if any(resolution not in EXTERIOR_RESOLUTIONS for resolution in selected):
        raise ValueError(
            f"Resolution must be one of {EXTERIOR_RESOLUTIONS}."
        )
    if arguments.command == "list":
        for resolution in EXTERIOR_RESOLUTIONS:
            print(archive_path(arguments.output_dir, resolution))
    elif arguments.command == "preflight":
        for resolution in selected:
            print(json.dumps(spectral_preflight(resolution), sort_keys=True))
    elif arguments.command == "run":
        for resolution in selected:
            print(
                run_case(
                    arguments.output_dir,
                    resolution,
                    resume_interrupted=arguments.resume_interrupted,
                )
            )
    else:
        print(json.dumps(analyze(arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
