"""Analyze the matched ``L/M=640`` outer-boundary tail campaign.

The analysis compares Schwarzschild at future null infinity with uniform and
exterior-supported Schwarzschild--de Sitter at the cosmological horizon.  It
uses no fitted time or amplitude alignment.  Decay rates are displayed only
where the finest envelope clears the measured refinement floor and the two
finest local-power estimates agree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .large_l_tail import (
    LocalFitSettings,
    cosmological_rate,
    effective_rates,
    rms_envelope,
)
from .sds_result import SdSSimulationResult, load_sds_result


DEFAULT_ROOT = Path("results/curvature_coupling_production_v2")
FORMULATION_CONTROL_ARCHIVES = (
    Path(
        "results/exterior_tail_feasibility_v2/raw/"
        "exterior_factored_L640_N1536_dt0p0025.npz"
    ),
    Path(
        "results/exterior_tail_feasibility_v2/raw/"
        "exterior_factored_L640_N2048_dt0p0025.npz"
    ),
)
REPRODUCTION_SOURCES = (
    (
        Path("black_hole/curvature_coupling_production.py"),
        "matched tail production generator",
    ),
    (
        Path("black_hole/run_curvature_campaign.py"),
        "matched production campaign orchestrator",
    ),
    (
        Path("black_hole/exterior_tail_feasibility.py"),
        "endpoint-factored formulation-control generator and analysis",
    ),
    (
        Path("black_hole/curvature_coupling_tail_analysis.py"),
        "matched tail analysis and manifest generator",
    ),
    (Path("black_hole/large_l_tail.py"), "tail-rate estimators"),
    (Path("black_hole/exterior_sds_model.py"), "exterior-supported model"),
    (Path("black_hole/sds_model.py"), "uniform SdS model and initial data"),
    (Path("black_hole/schwarzschild_scalar.py"), "Schwarzschild control model"),
    (Path("black_hole/sds_solver.py"), "Dedalus scalar evolution solver"),
    (Path("black_hole/sds_result.py"), "simulation archive serializer"),
    (Path("black_hole/regulator_suite.py"), "write-once archive helpers"),
    (Path("black_hole/reproducibility.py"), "runtime provenance collector"),
)
TAIL_LENGTH = 640.0
FULL_TIMESTEP = 0.0025
HALF_TIMESTEP = 0.00125
RESOLUTIONS = (1536, 2048, 3072)
PRIMARY_ENVELOPE_WIDTH = 10.0
PRIMARY_RATE_WIDTH = 40.0
PRICE_TARGET = 3.0
PRICE_TOLERANCE = 0.3
PRICE_MINIMUM_DURATION = 40.0
RATE_REFINEMENT_TOLERANCE = 0.1
ANALYSIS_START = 80.0
EXPONENTIAL_SCALED_WINDOW = 0.25
EXPONENTIAL_RELATIVE_TOLERANCE = 0.1
FLOOR_SAFETY_FACTOR = 10.0
CONVERGENCE_WINDOWS = (
    (160.0, 220.0),
    (220.0, 300.0),
    (300.0, 500.0),
    (500.0, 750.0),
    (750.0, 950.0),
    (950.0, 1000.0),
)


@dataclass(frozen=True)
class TailFamily:
    """One geometry and curvature coupling in the matched tail study."""

    key: str
    background: str
    coupling_label: str
    coupling: float
    label: str


FAMILIES = (
    TailFamily("schwarzschild", "schwarzschild", "xi0", 0.0, "Schwarzschild"),
    TailFamily("uniform_xi0", "uniform", "xi0", 0.0, "uniform SdS"),
    TailFamily("exterior_xi0", "exterior", "xi0", 0.0, "exterior SdS"),
    TailFamily(
        "uniform_xi1o6", "uniform", "xi1o6", 1.0 / 6.0, "uniform SdS"
    ),
    TailFamily(
        "exterior_xi1o6", "exterior", "xi1o6", 1.0 / 6.0, "exterior SdS"
    ),
)

COMMON_INITIAL_DATA = {
    "amplitude": 1.0,
    "center_radius": 6.0,
    "displacement": "u=0",
    "momentum": "pi=G(r)/A",
    "profile": "C-infinity physical areal-radius velocity bump",
    "radial_derivative": "psi=0",
    "support_half_width": 3.0,
}


def archive_path(
    root: Path, family: TailFamily, resolution: int, timestep: float
) -> Path:
    """Return the canonical archive path for one ladder member."""

    length = "schwarzschild" if family.background == "schwarzschild" else "L640"
    step = str(timestep).replace(".", "p")
    return (
        Path(root)
        / "raw"
        / "tail"
        / family.background
        / family.coupling_label
        / length
        / f"N{resolution}_dt{step}.npz"
    )


def load_ladder(
    root: Path, family: TailFamily
) -> dict[tuple[int, float], SdSSimulationResult]:
    """Load and validate the four matched archives for one family."""

    settings = tuple((resolution, FULL_TIMESTEP) for resolution in RESOLUTIONS) + (
        (2048, HALF_TIMESTEP),
    )
    ladder: dict[tuple[int, float], SdSSimulationResult] = {}
    for resolution, timestep in settings:
        path = archive_path(root, family, resolution, timestep)
        result = load_sds_result(path)
        contract = result.metadata.get("physical_contract", {})
        expected = {
            "group": "tail",
            "background": family.background,
            "ell": 1,
            "length": None if family.background == "schwarzschild" else TAIL_LENGTH,
            "resolution": resolution,
            "timestep": timestep,
            "end_time_or_u": 1000.0,
            "equation": "(Box_g-xi*R_g) Phi = 0",
            "reduced_field": "u=r*Phi",
            "gauge": "minimal",
            "timestepper": "RK222",
            "signal_dt": 0.05,
            "end_time_coordinate": "retarded U=tau-q",
            "outer_observer": (
                "future null infinity"
                if family.background == "schwarzschild"
                else "future cosmological horizon"
            ),
            "time_translation_fitted": False,
            "amplitude_rescaling_fitted": False,
            "initial_data": COMMON_INITIAL_DATA,
        }
        for key, value in expected.items():
            actual = contract.get(key)
            if isinstance(value, float):
                valid = np.isclose(float(actual), value)
            else:
                valid = actual == value
            if not valid:
                raise ValueError(
                    f"Physical-contract mismatch for {key!r} in {path}: "
                    f"expected {value!r}, found {actual!r}."
                )
        if not np.isclose(
            float(contract.get("curvature_coupling", np.nan)), family.coupling
        ):
            raise ValueError(f"Curvature-coupling mismatch in {path}.")
        canonical_contract = json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        contract_hash = hashlib.sha256(canonical_contract).hexdigest()
        stored_hash = result.metadata.get("physical_contract_sha256")
        provenance = result.metadata.get("simulation_provenance", {})
        if contract_hash != stored_hash or contract_hash != provenance.get(
            "physical_contract_sha256"
        ):
            raise ValueError(f"Physical-contract checksum mismatch in {path}.")
        audit = result.metadata.get("result_audit", {})
        if audit.get("finite") is not True:
            raise ValueError(f"Failed result audit in {path}.")
        if family.background == "exterior":
            required_flags = (
                audit.get("passed") is True,
                audit.get("base_grid_snapshot_width") is True,
                audit.get("late_growth_gate_passed") is True,
                result.metadata.get("spectral_preflight", {}).get("passed") is True,
                result.metadata.get("background_audit", {}).get(
                    "finite_coefficients"
                )
                is True,
            )
            if not all(required_flags):
                raise ValueError(f"Failed exterior formulation audit in {path}.")
        arrays = (
            result.signal_times,
            result.signals,
            result.constraint_linf,
            result.constraint_l2,
        )
        if not all(
            np.asarray(value).size and np.all(np.isfinite(value))
            for value in arrays
        ):
            raise FloatingPointError(f"Nonfinite or empty waveform archive {path}.")
        if not np.all(np.diff(result.signal_times) > 0.0):
            raise ValueError(f"Nonmonotone signal clock in {path}.")
        if not np.isclose(float(result.observer_rho[-1]), 1.0):
            raise ValueError(f"Outer observer is not at the boundary in {path}.")
        retarded_end = float(result.signal_times[-1]) - float(
            result.metadata["retarded_time_offset"]["q"]
        )
        if not np.isclose(retarded_end, 1000.0, atol=0.051):
            raise ValueError(f"Unexpected retarded-time endpoint in {path}.")
        ladder[(resolution, timestep)] = result
    return ladder


def retarded_series(
    result: SdSSimulationResult, observer: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shifted waveform at one stored observer."""

    offset = float(result.metadata["retarded_time_offset"]["q"])
    return (
        np.asarray(result.signal_times, dtype=float) - offset,
        np.asarray(result.signals[:, observer], dtype=float),
    )


def interpolate(
    source_times: np.ndarray, source: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Interpolate finite samples without extrapolating."""

    finite = np.isfinite(source_times) & np.isfinite(source)
    output = np.interp(target_times, source_times[finite], source[finite])
    output[
        (target_times < source_times[finite][0])
        | (target_times > source_times[finite][-1])
    ] = np.nan
    return output


def envelope_series(
    result: SdSSimulationResult, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unmasked RMS envelope at the outer boundary."""

    times, signal = retarded_series(result)
    return times, rms_envelope(times, signal, width, floor_multiplier=0.0)


def ladder_floor(
    ladder: dict[tuple[int, float], SdSSimulationResult], width: float
) -> dict[str, np.ndarray]:
    """Measure spatial and temporal envelope differences on the fine grid."""

    fine_times, fine = envelope_series(ladder[(3072, FULL_TIMESTEP)], width)
    medium_times, medium = envelope_series(ladder[(2048, FULL_TIMESTEP)], width)
    coarse_times, coarse = envelope_series(ladder[(1536, FULL_TIMESTEP)], width)
    half_times, half = envelope_series(ladder[(2048, HALF_TIMESTEP)], width)
    medium = interpolate(medium_times, medium, fine_times)
    coarse = interpolate(coarse_times, coarse, fine_times)
    half = interpolate(half_times, half, fine_times)
    spatial_fine = np.abs(fine - medium)
    spatial_coarse = np.abs(medium - coarse)
    temporal = np.abs(medium - half)
    return {
        "times": fine_times,
        "amplitude": fine,
        "spatial_fine": spatial_fine,
        "spatial_coarse": spatial_coarse,
        "temporal": temporal,
        "floor": np.maximum(spatial_fine, temporal),
    }


def longest_interval(
    times: np.ndarray, selected: np.ndarray
) -> tuple[float, float] | None:
    """Return the longest contiguous interval selected on a common grid."""

    indices = np.flatnonzero(selected)
    if not indices.size:
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    group = max(groups, key=lambda values: times[values[-1]] - times[values[0]])
    return float(times[group[0]]), float(times[group[-1]])


def away_from_zero_crossings(
    times: np.ndarray, signal: np.ndarray, half_width: float
) -> np.ndarray:
    """Mask neighborhoods in which a local logarithmic rate is ill-defined."""

    safe = np.ones(times.shape, dtype=bool)
    crossing_indices = np.flatnonzero(np.signbit(signal[1:]) != np.signbit(signal[:-1]))
    for index in crossing_indices:
        crossing = 0.5 * (times[index] + times[index + 1])
        safe &= np.abs(times - crossing) > half_width
    return safe


def rate_diagnostic(
    ladder: dict[tuple[int, float], SdSSimulationResult],
    envelope_width: float = PRIMARY_ENVELOPE_WIDTH,
    rate_width: float = PRIMARY_RATE_WIDTH,
) -> dict[str, np.ndarray | tuple[float, float] | None]:
    """Return refinement-supported local power diagnostics."""

    measured = ladder_floor(ladder, envelope_width)
    fine_times, fine_signal = retarded_series(ladder[(3072, FULL_TIMESTEP)])
    medium_times, medium_signal = retarded_series(ladder[(2048, FULL_TIMESTEP)])
    half_times, half_signal = retarded_series(ladder[(2048, HALF_TIMESTEP)])
    settings = LocalFitSettings(
        envelope_width=envelope_width,
        price_window=rate_width,
        exponential_scaled_window=EXPONENTIAL_SCALED_WINDOW,
        floor_multiplier=FLOOR_SAFETY_FACTOR,
    )
    kappa = cosmological_rate(TAIL_LENGTH)
    amplitude, power, gamma = effective_rates(
        fine_times,
        fine_signal,
        settings,
        kappa=kappa,
        measured_floor=measured["floor"],
    )
    _, medium_power, medium_gamma = effective_rates(
        medium_times, medium_signal, settings, kappa=kappa
    )
    _, half_power, half_gamma = effective_rates(
        half_times, half_signal, settings, kappa=kappa
    )
    medium_power = interpolate(medium_times, medium_power, fine_times)
    half_power = interpolate(half_times, half_power, fine_times)
    medium_gamma = interpolate(medium_times, medium_gamma, fine_times)
    half_gamma = interpolate(half_times, half_gamma, fine_times)
    zero_safe_power = away_from_zero_crossings(
        fine_times, fine_signal, half_width=0.5 * rate_width
    )
    supported = (
        np.isfinite(amplitude)
        & np.isfinite(power)
        & np.isfinite(medium_power)
        & np.isfinite(half_power)
        & (np.abs(power - medium_power) <= RATE_REFINEMENT_TOLERANCE)
        & (np.abs(medium_power - half_power) <= RATE_REFINEMENT_TOLERANCE)
        & zero_safe_power
    )
    exponential_width = EXPONENTIAL_SCALED_WINDOW / kappa
    gamma_supported = (
        np.isfinite(amplitude)
        & np.isfinite(gamma)
        & np.isfinite(medium_gamma)
        & np.isfinite(half_gamma)
        & (np.abs(gamma - medium_gamma) <= RATE_REFINEMENT_TOLERANCE)
        & (np.abs(medium_gamma - half_gamma) <= RATE_REFINEMENT_TOLERANCE)
        & away_from_zero_crossings(
            fine_times, fine_signal, half_width=0.5 * exponential_width
        )
    )
    price_samples = (
        supported
        & (fine_times >= ANALYSIS_START)
        & (np.abs(power - PRICE_TARGET) <= PRICE_TOLERANCE)
    )
    return {
        "times": fine_times,
        "signal": fine_signal,
        "amplitude": amplitude,
        "power": power,
        "power_supported": supported,
        "gamma_over_kappa": gamma,
        "gamma_supported": gamma_supported,
        "price_interval": longest_interval(fine_times, price_samples),
        "floor": measured["floor"],
    }


def exponential_row(
    family: TailFamily,
    diagnostic: dict[str, np.ndarray | tuple[float, float] | None],
) -> dict[str, float | str | bool]:
    """Audit the expected uniform-SdS cosmological rate."""

    if family.background != "uniform":
        raise ValueError("An expected exponential-rate target is uniform-SdS only.")
    target = 1.0 if np.isclose(family.coupling, 0.0) else 2.0
    times = np.asarray(diagnostic["times"])
    gamma = np.asarray(diagnostic["gamma_over_kappa"])
    supported = np.asarray(diagnostic["gamma_supported"], dtype=bool)
    selected = (
        supported
        & (times >= ANALYSIS_START)
        & (np.abs(gamma - target) <= EXPONENTIAL_RELATIVE_TOLERANCE * target)
    )
    interval = longest_interval(times, selected)
    if interval is None:
        start = end = duration = None
    else:
        start, end = interval
        duration = end - start
    minimum = EXPONENTIAL_SCALED_WINDOW / cosmological_rate(TAIL_LENGTH)
    return {
        "family": family.key,
        "curvature_coupling": family.coupling,
        "expected_gamma_over_kappa": target,
        "relative_tolerance": EXPONENTIAL_RELATIVE_TOLERANCE,
        "candidate_start_U_over_M": start,
        "candidate_end_U_over_M": end,
        "candidate_duration_over_M": duration,
        "minimum_required_duration_over_M": minimum,
        "passes_exponential_criterion": bool(
            duration is not None and duration >= minimum
        ),
    }


def power_law_fit_row(
    family: TailFamily,
    diagnostic: dict[str, np.ndarray | tuple[float, float] | None],
    start: float,
    end: float,
) -> dict[str, float | str | int]:
    """Fit a finite-interval power law to an exterior RMS envelope."""

    times = np.asarray(diagnostic["times"])
    amplitude = np.asarray(diagnostic["amplitude"])
    selected = (
        (times >= start)
        & (times <= end)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    x = np.log(times[selected])
    y = np.log(amplitude[selected])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "family": family.key,
        "curvature_coupling": family.coupling,
        "fit_start_U_over_M": start,
        "fit_end_U_over_M": end,
        "sample_count": int(selected.sum()),
        "power_law_exponent": float(-slope),
        "amplitude_prefactor": float(np.exp(intercept)),
        "r_squared": float(1.0 - residual / total),
        "status": "finite-interval transition-profile descriptor",
    }


def relative_l2(
    reference_times: np.ndarray,
    reference: np.ndarray,
    candidate_times: np.ndarray,
    candidate: np.ndarray,
    start: float,
    end: float,
) -> float:
    """Return a relative signed-waveform difference on a fixed interval."""

    candidate_on_reference = interpolate(candidate_times, candidate, reference_times)
    selected = (
        (reference_times >= start)
        & (reference_times <= end)
        & np.isfinite(candidate_on_reference)
    )
    return float(
        np.linalg.norm(candidate_on_reference[selected] - reference[selected])
        / np.linalg.norm(reference[selected])
    )


def convergence_rows(
    family: TailFamily,
    ladder: dict[tuple[int, float], SdSSimulationResult],
) -> list[dict[str, float | str]]:
    """Return successive spatial and fixed-grid temporal waveform changes."""

    traces = {key: retarded_series(result) for key, result in ladder.items()}
    rows: list[dict[str, float | str]] = []
    for start, end in CONVERGENCE_WINDOWS:
        coarse_times, coarse = traces[(1536, FULL_TIMESTEP)]
        medium_times, medium = traces[(2048, FULL_TIMESTEP)]
        fine_times, fine = traces[(3072, FULL_TIMESTEP)]
        half_times, half = traces[(2048, HALF_TIMESTEP)]
        rows.append(
            {
                "family": family.key,
                "curvature_coupling": family.coupling,
                "start_U_over_M": start,
                "end_U_over_M": end,
                "delta_1536_2048_relative_L2": relative_l2(
                    medium_times, medium, coarse_times, coarse, start, end
                ),
                "delta_2048_3072_relative_L2": relative_l2(
                    fine_times, fine, medium_times, medium, start, end
                ),
                "delta_dt_0p0025_0p00125_at_N2048_relative_L2": relative_l2(
                    half_times, half, medium_times, medium, start, end
                ),
            }
        )
    return rows


def accepted_interval_refinement_row(
    family: TailFamily,
    ladder: dict[tuple[int, float], SdSSimulationResult],
    diagnostic: dict[str, np.ndarray | tuple[float, float] | None],
) -> dict[str, float | str] | None:
    """Measure waveform refinement over one accepted Price interval."""

    interval = diagnostic["price_interval"]
    if not isinstance(interval, tuple) or interval[1] - interval[0] < PRICE_MINIMUM_DURATION:
        return None
    start, end = interval
    traces = {key: retarded_series(result) for key, result in ladder.items()}
    coarse_times, coarse = traces[(1536, FULL_TIMESTEP)]
    medium_times, medium = traces[(2048, FULL_TIMESTEP)]
    fine_times, fine = traces[(3072, FULL_TIMESTEP)]
    half_times, half = traces[(2048, HALF_TIMESTEP)]
    return {
        "family": family.key,
        "start_U_over_M": start,
        "end_U_over_M": end,
        "delta_1536_2048_relative_L2": relative_l2(
            medium_times, medium, coarse_times, coarse, start, end
        ),
        "delta_2048_3072_relative_L2": relative_l2(
            fine_times, fine, medium_times, medium, start, end
        ),
        "delta_dt_0p0025_0p00125_at_N2048_relative_L2": relative_l2(
            half_times, half, medium_times, medium, start, end
        ),
    }


def price_row(
    family: TailFamily,
    diagnostic: dict[str, np.ndarray | tuple[float, float] | None],
) -> dict[str, float | str | bool]:
    """Return one primary-estimator acceptance row."""

    interval = diagnostic["price_interval"]
    if not isinstance(interval, tuple):
        start = end = duration = np.nan
    else:
        start, end = interval
        duration = end - start
    return {
        "family": family.key,
        "geometry": family.label,
        "curvature_coupling": family.coupling,
        "candidate_start_U_over_M": start,
        "candidate_end_U_over_M": end,
        "candidate_duration_over_M": duration,
        "minimum_required_duration_over_M": PRICE_MINIMUM_DURATION,
        "passes_price_criterion": bool(duration >= PRICE_MINIMUM_DURATION),
    }


def estimator_rows(
    family: TailFamily,
    ladder: dict[tuple[int, float], SdSSimulationResult],
) -> list[dict[str, float | str | bool]]:
    """Sweep the declared envelope and local-fit widths."""

    rows: list[dict[str, float | str | bool]] = []
    for envelope_width in (5.0, 10.0, 20.0):
        for rate_width in (30.0, 40.0, 60.0):
            diagnostic = rate_diagnostic(ladder, envelope_width, rate_width)
            interval = diagnostic["price_interval"]
            if isinstance(interval, tuple):
                start, end = interval
                duration = end - start
            else:
                start = end = duration = np.nan
            rows.append(
                {
                    "family": family.key,
                    "envelope_width_over_M": envelope_width,
                    "rate_fit_width_over_M": rate_width,
                    "candidate_start_U_over_M": start,
                    "candidate_end_U_over_M": end,
                    "candidate_duration_over_M": duration,
                    "passes_price_criterion": bool(
                        duration >= PRICE_MINIMUM_DURATION
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty list of homogeneous dictionaries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def create_figure(
    diagnostics: dict[str, dict[str, np.ndarray | tuple[float, float] | None]],
    output: Path,
) -> None:
    """Create the six-panel outer-boundary tail comparison."""

    colors = {
        "schwarzschild": "#111111",
        "uniform": "#D55E00",
        "exterior": "#0072B2",
    }
    styles = {"schwarzschild": "--", "uniform": "-", "exterior": "-"}
    linewidths = {"schwarzschild": 1.25, "uniform": 1.35, "exterior": 1.35}
    columns = (
        ("xi0", r"minimal coupling, $\xi=0$"),
        ("xi1o6", r"conformal coupling, $\xi=1/6$"),
    )
    panel_labels = (("a", "b"), ("c", "d"), ("e", "f"))

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(7.1, 7.25),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for column, (coupling_label, title) in enumerate(columns):
        selected = [FAMILIES[0]] + [
            family
            for family in FAMILIES[1:]
            if family.coupling_label == coupling_label
        ]
        axes[0, column].set_title(title)
        for family in selected:
            diagnostic = diagnostics[family.key]
            times = np.asarray(diagnostic["times"])
            signal = np.asarray(diagnostic["signal"])
            amplitude = np.asarray(diagnostic["amplitude"])
            power = np.asarray(diagnostic["power"])
            supported = np.asarray(diagnostic["power_supported"], dtype=bool)
            display = (times >= 100.0) & (times <= 1000.0)
            color = colors[family.background]
            style = styles[family.background]
            width = linewidths[family.background]
            axes[0, column].plot(
                times[display],
                1.0e4 * signal[display],
                color=color,
                linestyle=style,
                linewidth=width,
                label=family.label,
            )
            axes[1, column].plot(
                times[display],
                amplitude[display],
                color=color,
                linestyle=style,
                linewidth=width,
            )
            rate_display = display & supported
            displayed_power = np.where(rate_display, power, np.nan)
            axes[2, column].plot(
                times[display],
                displayed_power[display],
                color=color,
                linestyle=style,
                linewidth=width,
            )

        axes[2, column].axhspan(
            PRICE_TARGET - PRICE_TOLERANCE,
            PRICE_TARGET + PRICE_TOLERANCE,
            color="#009E73",
            alpha=0.13,
            linewidth=0,
        )
        axes[2, column].axhline(
            PRICE_TARGET, color="#009E73", linewidth=0.85, linestyle=":"
        )

    for row in range(3):
        for column in range(2):
            axis = axes[row, column]
            axis.set_xlim(100.0, 1000.0)
            axis.grid(True, color="0.85", linewidth=0.55)
            axis.text(
                0.025,
                0.93,
                f"({panel_labels[row][column]})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
    axes[0, 0].set_ylabel(r"$10^{4}u_{\rm out}$")
    axes[0, 0].set_ylim(-0.75, 1.75)
    axes[1, 0].set_ylabel(r"RMS envelope $A(U)$")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(1.0e-8, 2.0e-4)
    axes[2, 0].set_ylabel(r"$p_{\rm eff}=-d\ln A/d\ln U$")
    axes[2, 0].set_ylim(0.0, 6.0)
    axes[2, 0].set_xlabel(r"retarded time $U/M$")
    axes[2, 1].set_xlabel(r"retarded time $U/M$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "black_hole.curvature_coupling_tail_analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "black_hole.curvature_coupling_tail_analysis"},
    )
    plt.close(figure)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_record(path: Path, repository_root: Path, **fields) -> dict:
    """Return a checksum record for one repository-relative file."""

    relative = Path(path)
    absolute = repository_root / relative
    if not absolute.is_file():
        raise FileNotFoundError(f"Missing reproducibility input: {absolute}")
    return {
        "path": relative.as_posix(),
        "bytes": absolute.stat().st_size,
        "sha256": sha256(absolute),
        **fields,
    }


def formulation_control_records(repository_root: Path) -> list[dict]:
    """Describe the two independent endpoint-factored control archives."""

    records = []
    for relative in FORMULATION_CONTROL_ARCHIVES:
        absolute = repository_root / relative
        result = load_sds_result(absolute)
        feasibility = result.metadata.get("exterior_tail_feasibility", {})
        numerical = result.metadata.get("numerical", {})
        if feasibility.get("state_variables") != (
            "endpoint-factored characteristic variables u, H, J"
        ):
            raise ValueError(
                f"Formulation control is not endpoint factored: {absolute}"
            )
        if (
            feasibility.get("length_over_M") != TAIL_LENGTH
            or feasibility.get("ell") != 1
            or feasibility.get("retarded_end_time_over_M") != 1000.0
            or numerical.get("timestep") != FULL_TIMESTEP
        ):
            raise ValueError(
                f"Formulation-control physical contract mismatch: {absolute}"
            )
        records.append(
            _repository_record(
                relative,
                repository_root,
                resolution=int(numerical["resolution"]),
                timestep_over_M=float(numerical["timestep"]),
                retarded_end_time_over_M=float(
                    feasibility["retarded_end_time_over_M"]
                ),
                length_over_M=float(feasibility["length_over_M"]),
                ell=int(feasibility["ell"]),
                curvature_coupling=0.0,
                state_variables=feasibility["state_variables"],
                generating_module="black_hole.exterior_tail_feasibility",
                archived_git_commit=result.metadata["simulation_provenance"].get(
                    "git_commit"
                ),
                archived_worktree_dirty=result.metadata[
                    "simulation_provenance"
                ].get("git_worktree_dirty"),
            )
        )
    return records


def source_records(repository_root: Path) -> list[dict]:
    """Return checksums for the complete matched-tail reproduction path."""

    return [
        _repository_record(path, repository_root, role=role)
        for path, role in REPRODUCTION_SOURCES
    ]


def verify_manifest(path: Path, repository_root: Path | None = None) -> dict:
    """Verify every archive, artifact, and source checksum in a tail manifest."""

    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    package_root = path.parent
    repository_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )

    checked = 0
    for section in ("raw_archives", "derived_artifacts"):
        for row in manifest[section]:
            artifact = package_root / row["path"]
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing manifest entry: {artifact}")
            if artifact.stat().st_size != row["bytes"] or sha256(artifact) != row[
                "sha256"
            ]:
                raise ValueError(f"Checksum mismatch: {artifact}")
            checked += 1
    for section in ("formulation_control_archives", "source_files"):
        for row in manifest[section]:
            artifact = repository_root / row["path"]
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing manifest entry: {artifact}")
            if artifact.stat().st_size != row["bytes"] or sha256(artifact) != row[
                "sha256"
            ]:
                raise ValueError(f"Checksum mismatch: {artifact}")
            checked += 1

    expected_counts = {
        "archive_count": len(manifest["raw_archives"]),
        "formulation_control_archive_count": len(
            manifest["formulation_control_archives"]
        ),
        "source_file_count": len(manifest["source_files"]),
    }
    for key, expected in expected_counts.items():
        if manifest[key] != expected:
            raise ValueError(
                f"Manifest count mismatch for {key}: "
                f"recorded {manifest[key]}, found {expected}"
            )
    return {
        "verified": True,
        "manifest": path.as_posix(),
        "checked_file_count": checked,
        **expected_counts,
    }


def create_submission_figure(root: Path, destination: Path) -> Path:
    """Regenerate only the manuscript figure without rewriting analysis tables."""

    root = Path(root)
    ladders = {family.key: load_ladder(root, family) for family in FAMILIES}
    diagnostics = {
        family.key: rate_diagnostic(ladders[family.key]) for family in FAMILIES
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sds-tail-figure-") as temporary:
        stem = Path(temporary) / "tail_outer_boundary_comparison"
        create_figure(diagnostics, stem)
        shutil.copy2(stem.with_suffix(".pdf"), destination)
    return destination


def create_report(root: Path, paper_figure: Path | None = None) -> list[Path]:
    """Generate tables, figure, summary, and a checksum index."""

    root = Path(root)
    repository_root = Path(__file__).resolve().parents[1]
    ladders = {family.key: load_ladder(root, family) for family in FAMILIES}
    diagnostics = {
        family.key: rate_diagnostic(ladders[family.key]) for family in FAMILIES
    }
    price_rows = [
        price_row(family, diagnostics[family.key]) for family in FAMILIES
    ]
    convergence = [
        row
        for family in FAMILIES
        for row in convergence_rows(family, ladders[family.key])
    ]
    interval_refinement = [
        row
        for family in FAMILIES
        if (
            row := accepted_interval_refinement_row(
                family, ladders[family.key], diagnostics[family.key]
            )
        )
        is not None
    ]
    sensitivity = [
        row
        for family in FAMILIES
        for row in estimator_rows(family, ladders[family.key])
    ]
    exponential = [
        exponential_row(family, diagnostics[family.key])
        for family in FAMILIES
        if family.background == "uniform"
    ]
    fit_windows = {
        "exterior_xi0": (120.0, 250.0),
        "exterior_xi1o6": (160.0, 400.0),
    }
    exterior_fits = [
        power_law_fit_row(
            family,
            diagnostics[family.key],
            *fit_windows[family.key],
        )
        for family in FAMILIES
        if family.key in fit_windows
    ]

    tables = root / "tables"
    price_path = tables / "tail_price_intervals.csv"
    convergence_path = tables / "tail_numerical_refinement.csv"
    interval_refinement_path = tables / "tail_accepted_interval_refinement.csv"
    sensitivity_path = tables / "tail_estimator_sensitivity.csv"
    exponential_path = tables / "tail_uniform_exponential_intervals.csv"
    fit_path = tables / "tail_exterior_component_fits.csv"
    write_csv(price_path, price_rows)
    write_csv(convergence_path, convergence)
    write_csv(interval_refinement_path, interval_refinement)
    write_csv(sensitivity_path, sensitivity)
    write_csv(exponential_path, exponential)
    write_csv(fit_path, exterior_fits)

    figure_stem = root / "tail_outer_boundary_comparison"
    create_figure(diagnostics, figure_stem)
    figure_pdf = figure_stem.with_suffix(".pdf")
    figure_png = figure_stem.with_suffix(".png")
    if paper_figure is not None:
        paper_figure = Path(paper_figure)
        paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_pdf, paper_figure)

    maximum_constraints = {
        family.key: max(
            float(np.max(result.constraint_linf))
            for result in ladders[family.key].values()
        )
        for family in FAMILIES
    }
    summary = {
        "tail_length_over_M": TAIL_LENGTH,
        "mode": 1,
        "outer_observers": {
            "schwarzschild": "future null infinity",
            "finite_L": "future cosmological horizon",
        },
        "primary_estimator": {
            "rms_envelope_width_over_M": PRIMARY_ENVELOPE_WIDTH,
            "local_log_fit_width_over_M": PRIMARY_RATE_WIDTH,
            "price_target": PRICE_TARGET,
            "price_tolerance": PRICE_TOLERANCE,
            "minimum_duration_over_M": PRICE_MINIMUM_DURATION,
            "fine_medium_rate_tolerance": RATE_REFINEMENT_TOLERANCE,
            "medium_half_step_rate_tolerance": RATE_REFINEMENT_TOLERANCE,
            "measured_floor_multiplier": FLOOR_SAFETY_FACTOR,
        },
        "uniform_exponential_audit": {
            "semilog_fit_width_in_kappa_U": EXPONENTIAL_SCALED_WINDOW,
            "relative_rate_tolerance": EXPONENTIAL_RELATIVE_TOLERANCE,
            "rows": exponential,
        },
        "exterior_finite_interval_fits": exterior_fits,
        "kappa_c_M": cosmological_rate(TAIL_LENGTH),
        "kappa_c_U_at_endpoint": cosmological_rate(TAIL_LENGTH) * 1000.0,
        "price_rows": price_rows,
        "accepted_interval_refinement": interval_refinement,
        "maximum_auxiliary_compatibility_residual": maximum_constraints,
        "conclusion": (
            "Uniform conformal SdS resolves an intermediate Schwarzschild "
            "Price interval. Neither exterior-supported coupling passes the "
            "40M Price criterion; both contain convergent transition-supported "
            "late structure. Neither uniform case resolves its expected "
            "cosmological exponential rate through U/M=1000; no asymptotic "
            "rate target is assigned to the exterior construction."
        ),
    }
    summary_path = root / "tail_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    raw_paths = sorted((root / "raw" / "tail").rglob("*.npz"))
    derived_paths = [
        price_path,
        convergence_path,
        interval_refinement_path,
        sensitivity_path,
        exponential_path,
        fit_path,
        figure_pdf,
        figure_png,
        summary_path,
    ]
    campaign_records = {
        (
            result.metadata["simulation_provenance"]["campaign_id"],
            result.metadata["simulation_provenance"]["campaign_source_sha256"],
        )
        for ladder in ladders.values()
        for result in ladder.values()
    }
    manifest = {
        "schema_version": 2,
        "analysis": {
            "command": (
                "python -m black_hole.curvature_coupling_tail_analysis "
                f"--root {root.as_posix()}"
            ),
            "source": "black_hole/curvature_coupling_tail_analysis.py",
            "source_sha256": sha256(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "simulation_campaigns": [
            {"campaign_id": campaign_id, "campaign_source_sha256": source_hash}
            for campaign_id, source_hash in sorted(campaign_records)
        ],
        "archive_count": len(raw_paths),
        "formulation_control_archive_count": len(FORMULATION_CONTROL_ARCHIVES),
        "source_file_count": len(REPRODUCTION_SOURCES),
        "raw_archives": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "physical_contract_sha256": load_sds_result(path).metadata[
                    "physical_contract_sha256"
                ],
                "campaign_id": load_sds_result(path).metadata[
                    "simulation_provenance"
                ]["campaign_id"],
            }
            for path in raw_paths
        ],
        "derived_artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in derived_paths
        ],
        "formulation_control_archives": formulation_control_records(
            repository_root
        ),
        "source_files": source_records(repository_root),
        "verification": {
            "command": (
                "python -m black_hole.curvature_coupling_tail_analysis "
                f"--root {root.as_posix()} --verify-manifest"
            ),
            "path_scopes": {
                "raw_archives": "relative to the matched-tail package",
                "derived_artifacts": "relative to the matched-tail package",
                "formulation_control_archives": "relative to repository root",
                "source_files": "relative to repository root",
            },
        },
        "legacy_control_snapshot_note": (
            "The Schwarzschild and uniform-SdS controls store snapshots on "
            "the 3/2-dealiased grid. Tail conclusions use only their audited "
            "observer waveforms; the exterior archives store base-grid snapshots."
        ),
    }
    manifest_path = root / "tail_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return derived_paths + [manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--paper-figure", type=Path)
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--repository-root", type=Path)
    arguments = parser.parse_args()
    if arguments.verify_manifest:
        print(
            json.dumps(
                verify_manifest(
                    arguments.root / "tail_manifest.json",
                    arguments.repository_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    for output in create_report(arguments.root, arguments.paper_figure):
        print(output)


if __name__ == "__main__":
    main()
