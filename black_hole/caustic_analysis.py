"""Local estimators for localized source caustic pulse observables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import hilbert
from scipy.signal.windows import tukey

from .caustic_study import COSMOLOGICAL_LENGTHS, direction_waveform, load_case
from .source_evolution import SourcedSimulationResult


PULSE_WINDOWS = (
    (18.0, 35.0, 0.0),
    (35.0, 53.0, np.pi),
    (53.0, 69.0, 0.0),
    (69.0, 86.0, np.pi),
)
"""Common windows for the direct pulse and three echoes in existing archives."""


@dataclass(frozen=True)
class LocalPulseEstimate:
    pulse: int
    phi: float
    window_start: float
    window_end: float
    analytic_time: float
    matched_time: float
    matched_time_resolved: bool
    time: float
    timing_systematic: float
    cadence_uncertainty: float
    timing_uncertainty: float
    analytic_amplitude: float
    matched_amplitude: float
    amplitude: float
    amplitude_systematic: float
    signed_amplitude: float
    integrated_field_energy: float
    integrated_flux_energy: float

    def as_dict(self) -> dict:
        row = asdict(self)
        row["phi_over_pi"] = row.pop("phi") / np.pi
        return row


def _windowed_trace(
    times: np.ndarray,
    trace: np.ndarray,
    window: tuple[float, float],
    *,
    taper_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inside = (times >= window[0]) & (times <= window[1])
    local_times = np.asarray(times[inside], dtype=float)
    local_trace = np.asarray(trace[inside], dtype=float)
    if local_times.size < 12:
        raise ValueError("A pulse window must contain at least twelve samples.")
    edge_count = max(3, local_times.size // 6)
    edges = np.r_[0:edge_count, local_times.size - edge_count : local_times.size]
    centered = local_times - 0.5 * sum(window)
    background_design = np.column_stack(
        [np.ones(edges.size), centered[edges]]
    )
    background_coefficients = np.linalg.lstsq(
        background_design, local_trace[edges], rcond=None
    )[0]
    background = background_coefficients[0] + background_coefficients[1] * centered
    detrended = local_trace - background
    taper = tukey(local_times.size, alpha=taper_fraction)
    return local_times, detrended, detrended * taper, background


def _parabolic_peak(
    times: np.ndarray, values: np.ndarray, index: int
) -> tuple[float, float]:
    if index <= 0 or index >= values.size - 1:
        return float(times[index]), float(values[index])
    left, middle, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * middle + right
    if abs(denominator) < np.finfo(float).eps * max(1.0, abs(middle)):
        return float(times[index]), float(middle)
    offset = 0.5 * (left - right) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    step = float(times[index + 1] - times[index])
    value = middle - 0.25 * (left - right) * offset
    return float(times[index] + offset * step), float(value)


def analytic_signal_estimate(
    times: np.ndarray,
    trace: np.ndarray,
    window: tuple[float, float],
) -> dict[str, float]:
    """Measure one pulse with a tapered local analytic signal."""

    local_times, detrended, tapered, _ = _windowed_trace(times, trace, window)
    analytic = hilbert(tapered)
    envelope = np.abs(analytic)
    guard = max(2, local_times.size // 10)
    peak = guard + int(np.argmax(envelope[guard:-guard]))
    pulse_time, amplitude = _parabolic_peak(local_times, envelope, peak)
    signed = float(np.interp(pulse_time, local_times, detrended))
    field_energy = float(np.trapezoid(detrended**2, local_times))
    derivative = np.gradient(detrended, local_times, edge_order=2)
    flux_energy = float(np.trapezoid(derivative**2, local_times))
    return {
        "time": pulse_time,
        "amplitude": amplitude,
        "signed_amplitude": signed,
        "integrated_field_energy": field_energy,
        "integrated_flux_energy": flux_energy,
    }


def matched_template_estimate(
    times: np.ndarray,
    trace: np.ndarray,
    reference_times: np.ndarray,
    reference_trace: np.ndarray,
    window: tuple[float, float],
    *,
    maximum_shift: float = 4.0,
) -> dict[str, float]:
    """Fit amplitude, phase, time shift, constant, and linear background."""

    candidate_times, _, candidate_tapered, _ = _windowed_trace(times, trace, window)
    template_times, _, template_tapered, _ = _windowed_trace(
        reference_times, reference_trace, window
    )
    template_analytic = hilbert(template_tapered)
    midpoint = 0.5 * sum(window)
    taper_weights = tukey(candidate_times.size, alpha=0.25)

    def solve(shift: float) -> tuple[float, np.ndarray]:
        query = candidate_times - shift
        real = np.interp(query, template_times, template_analytic.real, left=0.0, right=0.0)
        imaginary = np.interp(
            query, template_times, template_analytic.imag, left=0.0, right=0.0
        )
        design = np.column_stack(
            [real, imaginary, np.ones(candidate_times.size), candidate_times - midpoint]
        )
        weighted_design = taper_weights[:, None] * design
        weighted_signal = taper_weights * candidate_tapered
        coefficients = np.linalg.lstsq(weighted_design, weighted_signal, rcond=None)[0]
        residual = weighted_signal - weighted_design @ coefficients
        return float(residual @ residual), coefficients

    optimum = minimize_scalar(
        lambda shift: solve(float(shift))[0],
        bounds=(-maximum_shift, maximum_shift),
        method="bounded",
        options={"xatol": 1e-7},
    )
    _, coefficients = solve(float(optimum.x))
    reference = analytic_signal_estimate(reference_times, reference_trace, window)
    scale = float(np.hypot(coefficients[0], coefficients[1]))
    cadence = float(np.median(np.diff(candidate_times)))
    boundary_tolerance = max(5.0 * cadence, 0.01 * maximum_shift)
    lag_at_boundary = abs(float(optimum.x)) >= maximum_shift - boundary_tolerance
    return {
        "time": float(reference["time"] + optimum.x),
        "amplitude": scale * float(reference["amplitude"]),
        "phase": float(np.arctan2(-coefficients[1], coefficients[0])),
        "time_shift": float(optimum.x),
        "constant_offset": float(coefficients[2]),
        "linear_background": float(coefficients[3]),
        "objective": float(optimum.fun),
        "lag_at_boundary": bool(lag_at_boundary),
        "resolved": bool(optimum.success and not lag_at_boundary),
    }


def estimate_pulse(
    *,
    pulse: int,
    phi: float,
    times: np.ndarray,
    trace: np.ndarray,
    reference_times: np.ndarray,
    reference_trace: np.ndarray,
    window: tuple[float, float],
) -> LocalPulseEstimate:
    analytic = analytic_signal_estimate(times, trace, window)
    matched = matched_template_estimate(
        times, trace, reference_times, reference_trace, window
    )
    cadence = float(np.median(np.diff(times)))
    timing_systematic = abs(float(analytic["time"] - matched["time"]))
    cadence_uncertainty = 0.5 * cadence
    timing_uncertainty = float(np.hypot(timing_systematic, cadence_uncertainty))
    amplitude_systematic = abs(float(analytic["amplitude"] - matched["amplitude"]))
    return LocalPulseEstimate(
        pulse=pulse,
        phi=phi,
        window_start=window[0],
        window_end=window[1],
        analytic_time=float(analytic["time"]),
        matched_time=float(matched["time"]),
        matched_time_resolved=bool(matched["resolved"]),
        time=float(analytic["time"]),
        timing_systematic=timing_systematic,
        cadence_uncertainty=cadence_uncertainty,
        timing_uncertainty=timing_uncertainty,
        analytic_amplitude=float(analytic["amplitude"]),
        matched_amplitude=float(matched["amplitude"]),
        amplitude=float(analytic["amplitude"]),
        amplitude_systematic=amplitude_systematic,
        signed_amplitude=float(analytic["signed_amplitude"]),
        integrated_field_energy=float(analytic["integrated_field_energy"]),
        integrated_flux_energy=float(analytic["integrated_flux_energy"]),
    )


def _observer_name(result: SourcedSimulationResult, observer: int) -> str:
    if observer == result.outer_index():
        return "outer"
    return f"r{result.observer_areal_radius[observer]:g}M"


def local_phase_comparison(
    times: np.ndarray,
    first_trace: np.ndarray,
    second_trace: np.ndarray,
    first: LocalPulseEstimate,
    second: LocalPulseEstimate,
    *,
    half_width: float = 7.0,
    samples: int = 401,
) -> dict[str, float | bool]:
    """Measure the complex phase of consecutive pulses using local windows."""

    offsets = np.linspace(-half_width, half_width, samples)
    first_values = np.interp(first.time + offsets, times, first_trace)
    second_values = np.interp(second.time + offsets, times, second_trace)
    matched = matched_template_estimate(
        offsets,
        second_values,
        offsets,
        first_values,
        (-half_width, half_width),
        maximum_shift=2.0,
    )
    candidate_norm = float(np.linalg.norm(second_values) ** 2)
    coherence = np.sqrt(
        max(0.0, 1.0 - float(matched["objective"]) / candidate_norm)
    ) if candidate_norm else np.nan
    # Report the retarded Green function convention, in which the simple
    # caustic Maslov factor is exp(-i pi/2).  SciPy's analytic signal uses
    # the opposite Fourier sign, so its fitted phase is negated here.
    maslov_phase = -float(matched["phase"])
    resolved = bool(matched["resolved"])
    return {
        "phase_radians": maslov_phase if resolved else np.nan,
        "phase_over_half_pi": float(maslov_phase / (0.5 * np.pi)) if resolved else np.nan,
        "raw_phase_radians": maslov_phase,
        "matched_lag_over_M": float(matched["time_shift"]),
        "lag_at_boundary": bool(matched["lag_at_boundary"]),
        "phase_resolved": resolved,
        "coherence": float(coherence),
    }


def analyze_existing_archives(output_dir: Path) -> tuple[list[dict], list[dict]]:
    """Measure all requested pulse observables in the existing main archives."""

    reference = load_case(output_dir, "schwarzschild")
    cases: tuple[str | float, ...] = ("schwarzschild", *COSMOLOGICAL_LENGTHS)
    pulse_rows: list[dict] = []
    phase_rows: list[dict] = []
    for observer in range(reference.observer_areal_radius.size):
        reference_traces = {
            phi: direction_waveform(reference, phi, observer)[1]
            for phi in (0.0, np.pi)
        }
        for case in cases:
            result = load_case(output_dir, case)
            traces = {
                phi: direction_waveform(result, phi, observer)[1]
                for phi in (0.0, np.pi)
            }
            estimates: list[LocalPulseEstimate] = []
            for pulse, (start, end, phi) in enumerate(PULSE_WINDOWS):
                estimate = estimate_pulse(
                    pulse=pulse,
                    phi=phi,
                    times=result.retarded_time,
                    trace=traces[phi],
                    reference_times=reference.retarded_time,
                    reference_trace=reference_traces[phi],
                    window=(start, end),
                )
                estimates.append(estimate)
                row = estimate.as_dict()
                row.update(
                    {
                        "case": "schwarzschild" if case == "schwarzschild" else f"sds_L{case:g}",
                        "cosmological_length_over_M": np.inf if case == "schwarzschild" else float(case),
                        "observer": _observer_name(result, observer),
                        "observer_radius_over_M": float(result.observer_areal_radius[observer]),
                        "output_cadence_over_M": float(np.median(np.diff(result.retarded_time))),
                    }
                )
                if pulse:
                    previous = estimates[pulse - 1]
                    row["delay_over_M"] = estimate.time - previous.time
                    row["delay_uncertainty_over_M"] = float(
                        np.hypot(estimate.timing_uncertainty, previous.timing_uncertainty)
                    )
                    row["amplitude_ratio"] = estimate.amplitude / previous.amplitude
                    row["energy_ratio"] = (
                        estimate.integrated_flux_energy / previous.integrated_flux_energy
                    )
                else:
                    row["delay_over_M"] = np.nan
                    row["delay_uncertainty_over_M"] = np.nan
                    row["amplitude_ratio"] = np.nan
                    row["energy_ratio"] = np.nan
                pulse_rows.append(row)
            for first, second in zip(estimates, estimates[1:]):
                phase = local_phase_comparison(
                    result.retarded_time,
                    traces[first.phi],
                    traces[second.phi],
                    first,
                    second,
                )
                phase.update(
                    {
                        "case": "schwarzschild" if case == "schwarzschild" else f"sds_L{case:g}",
                        "observer": _observer_name(result, observer),
                        "pulse_pair": f"{first.pulse}->{second.pulse}",
                        "delay_over_M": second.time - first.time,
                    }
                )
                phase_rows.append(phase)
    return pulse_rows, phase_rows
