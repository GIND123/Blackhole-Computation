"""Higher-resolution finite-radius tail and SdS crossover study.

The evolution entry points in this module record exact Dedalus interpolation
operators at selected areal radii.  This avoids reconstructing high-frequency
time series from sparsely saved spatial snapshots and provides a clean
resolution comparison for local logarithmic decay rates.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .sds_model import (
    ArealVelocityBumpInitialData,
    SdSParameters,
    compact_radius,
    retarded_time_offset as sds_retarded_time_offset,
    sds_horizons,
)
from .sds_result import load_sds_result
from .sds_solver import (
    SdSNumericalParameters,
    run_schwarzschild_scalar_simulation,
    run_sds_simulation,
)
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    retarded_time_offset as schwarzschild_retarded_time_offset,
)
from .tail_analysis import (
    decay_rate_transition_time,
    json_safe,
    local_decay_rates,
    numerical_amplitude_floor,
)
from .tail_rate_figures import interpolate_chebyshev_snapshots


REFERENCE_RADIUS = 4.0
INITIAL_DATA = ArealVelocityBumpInitialData(
    center_radius=6.0,
    support_half_width=3.0,
    amplitude=1.0,
)
SCHWARZSCHILD_RADII = (4.0, 8.0, 16.0, 20.0, 50.0, 100.0, 200.0)
SDS_RADII = (4.0, 8.0, 16.0)


@dataclass(frozen=True)
class ResolutionCase:
    background: str
    ell: int
    resolution: int
    timestep: float
    end_time: float
    length: float | None = None

    @property
    def filename(self) -> str:
        if self.background == "schwarzschild":
            return f"schwarzschild_ell{self.ell}_N{self.resolution}.npz"
        if self.length is None:
            raise ValueError("An SdS case requires a cosmological length.")
        return (
            f"sds_ell{self.ell}_L{self.length:g}_"
            f"N{self.resolution}.npz"
        )


SCHWARZSCHILD_CASES = (
    ResolutionCase("schwarzschild", 0, 2048, 0.0025, 720.0),
    ResolutionCase("schwarzschild", 1, 2048, 0.0025, 360.0),
    ResolutionCase("schwarzschild", 2, 4096, 0.00125, 440.0),
)


def _sds_ell1_cases() -> tuple[ResolutionCase, ...]:
    cases = []
    for length in (20.0, 40.0, 80.0, 160.0):
        model = SdSParameters(mass=1.0, cosmological_length=length, ell=1)
        kappa = sds_horizons(model).kappa_cosmological
        cases.append(
            ResolutionCase(
                "sds",
                1,
                2048,
                0.0025,
                max(120.0, 5.0 / kappa),
                length,
            )
        )
    return tuple(cases)


SDS_ELL1_CASES = _sds_ell1_cases()
SDS_ELL2_L80_CASE = ResolutionCase(
    "sds",
    2,
    4096,
    0.00125,
    492.594296548683,
    80.0,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _numerical(
    case: ResolutionCase, observers: tuple[float, ...]
) -> SdSNumericalParameters:
    return SdSNumericalParameters(
        resolution=case.resolution,
        timestep=case.timestep,
        end_time=case.end_time,
        signal_dt=0.05,
        snapshot_dt=20.0,
        timestepper="RK222",
        observers=observers,
        bridge="minimal",
        dealias=1.5,
    )


def run_resolution_case(
    case: ResolutionCase,
    output_dir: Path,
    *,
    reuse_existing: bool = False,
) -> Path:
    """Run one exact-observer higher-resolution case and save its archive."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / case.filename
    if reuse_existing and path.exists():
        return path

    if case.background == "schwarzschild":
        model = SchwarzschildScalarParameters(mass=1.0, ell=case.ell)
        compact_observers = tuple(
            1.0 - 2.0 * model.mass / radius
            for radius in SCHWARZSCHILD_RADII
        ) + (1.0,)
        result = run_schwarzschild_scalar_simulation(
            model,
            INITIAL_DATA,
            _numerical(case, compact_observers),
        )
        offset = schwarzschild_retarded_time_offset(model, REFERENCE_RADIUS)
        result.metadata["retarded_time_offset"] = {
            "q": offset,
            "definition": "lim_(r->infinity)(h+r_*)",
            "evaluation": "analytic",
        }
    elif case.background == "sds":
        if case.length is None:
            raise ValueError("An SdS case requires a cosmological length.")
        model = SdSParameters(
            mass=1.0,
            cosmological_length=case.length,
            ell=case.ell,
        )
        horizons = sds_horizons(model)
        if any(
            radius <= horizons.black_hole or radius >= horizons.cosmological
            for radius in SDS_RADII
        ):
            raise ValueError(
                f"Requested observers do not fit inside L={case.length:g}."
            )
        compact_observers = tuple(
            float(value)
            for value in compact_radius(np.asarray(SDS_RADII), model)
        ) + (1.0,)
        result = run_sds_simulation(
            model,
            INITIAL_DATA,
            _numerical(case, compact_observers),
        )
        offset = sds_retarded_time_offset(model, REFERENCE_RADIUS)
        result.metadata["retarded_time_offset"] = {
            "q": offset,
            "definition": "lim_(r->r_c)(h+r_*)",
            "evaluation": "analytic",
        }
    else:
        raise ValueError(f"Unknown background {case.background!r}.")

    result.metadata["finite_radius_rate_study"] = {
        "areal_radii": [
            float(value) for value in result.observer_areal_radius
        ],
        "exact_dedalus_interpolation_operators": True,
    }
    result.save(path)
    return path


def run_suite(
    group: str,
    output_dir: Path,
    *,
    reuse_existing: bool = False,
) -> list[Path]:
    """Run a named higher-resolution production group."""

    if group == "schwarzschild":
        cases = SCHWARZSCHILD_CASES
    elif group == "sds-ell1":
        cases = SDS_ELL1_CASES
    elif group == "all":
        cases = SCHWARZSCHILD_CASES + SDS_ELL1_CASES
    else:
        raise ValueError(f"Unknown suite group {group!r}.")
    return [
        run_resolution_case(case, output_dir, reuse_existing=reuse_existing)
        for case in cases
    ]


def _time_offset(result) -> float:
    return float(result.metadata["retarded_time_offset"]["q"])


def _observer_index(result, radius: float | None) -> int:
    if radius is None:
        return int(np.argmax(result.observer_rho))
    finite = np.isfinite(result.observer_areal_radius)
    candidates = np.flatnonzero(finite)
    return int(
        candidates[
            np.argmin(
                np.abs(result.observer_areal_radius[candidates] - radius)
            )
        ]
    )


def _rate_series(
    result,
    radius: float | None,
    *,
    smoothing_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = _observer_index(result, radius)
    times = np.asarray(result.signal_times, dtype=float) - _time_offset(result)
    signal = np.asarray(result.signals[:, index], dtype=float)
    step = float(np.median(np.diff(times)))
    window = max(7, int(round(smoothing_width / step)))
    if window % 2 == 0:
        window += 1
    _, positive_power, exponential = local_decay_rates(
        times,
        signal,
        window=window,
    )
    floor = numerical_amplitude_floor(signal)
    resolved = np.abs(signal) > 10.0 * floor
    positive_power[~resolved] = np.nan
    exponential[~resolved] = np.nan
    return times, -positive_power, exponential


def _envelope_rate_series(
    result,
    radius: float | None,
    *,
    smoothing_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rates of a phase-insensitive sliding RMS amplitude envelope.

    Direct differentiation of ``log(abs(u))`` is singular at every
    quasinormal-mode zero crossing.  A centered RMS envelope removes that
    phase dependence while preserving a slowly varying power-law or
    exponential decay rate.
    """

    index = _observer_index(result, radius)
    times = np.asarray(result.signal_times, dtype=float) - _time_offset(result)
    signal = np.asarray(result.signals[:, index], dtype=float)
    step = float(np.median(np.diff(times)))
    envelope_window = max(7, int(round(0.5 * smoothing_width / step)))
    if envelope_window % 2 == 0:
        envelope_window += 1
    kernel = np.full(envelope_window, 1.0 / envelope_window)
    envelope = np.sqrt(np.convolve(signal**2, kernel, mode="same"))
    smoothing_window = max(7, int(round(smoothing_width / step)))
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    _, positive_power, exponential = local_decay_rates(
        times,
        envelope,
        window=smoothing_window,
    )
    floor = numerical_amplitude_floor(signal)
    resolved = envelope > 10.0 * floor
    edge = envelope_window // 2 + smoothing_window // 2
    resolved[:edge] = False
    resolved[-edge:] = False
    positive_power[~resolved] = np.nan
    exponential[~resolved] = np.nan
    return times, -positive_power, exponential


def _snapshot_rate_series(
    result,
    radius: float,
    *,
    smoothing_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local rates reconstructed from a saved dealiased snapshot series."""

    background = str(result.metadata["background"])
    model_metadata = result.metadata["model"]
    mass = float(model_metadata["mass"])
    if background == "Schwarzschild":
        target_rho = 1.0 - 2.0 * mass / radius
    else:
        model = SdSParameters(
            mass=mass,
            cosmological_length=float(model_metadata["cosmological_length"]),
            ell=int(model_metadata["ell"]),
        )
        target_rho = float(compact_radius(np.asarray([radius]), model)[0])
    signal = interpolate_chebyshev_snapshots(
        np.asarray(result.u_snapshots, dtype=float),
        target_rho,
    )
    times = np.asarray(result.snapshot_times, dtype=float) - _time_offset(result)
    step = float(np.median(np.diff(times)))
    window = max(7, int(round(smoothing_width / step)))
    if window % 2 == 0:
        window += 1
    _, positive_power, exponential = local_decay_rates(
        times,
        signal,
        window=window,
    )
    floor = numerical_amplitude_floor(signal)
    resolved = np.abs(signal) > 10.0 * floor
    positive_power[~resolved] = np.nan
    exponential[~resolved] = np.nan
    return times, -positive_power, exponential


def _display_mask(
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> np.ndarray:
    return (
        (x >= x_limits[0])
        & (x <= x_limits[1])
        & np.isfinite(y)
        & (y >= y_limits[0])
        & (y <= y_limits[1])
    )


def _plot_values(
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a plotted rate with NaN gaps instead of joining omitted points."""

    x_mask = (x >= x_limits[0]) & (x <= x_limits[1])
    values = np.asarray(y[x_mask], dtype=float).copy()
    values[
        (~np.isfinite(values))
        | (values < y_limits[0])
        | (values > y_limits[1])
    ] = np.nan
    return x[x_mask], values


def plot_high_resolution_schwarzschild(
    raw_dir: Path, output_dir: Path
) -> Path:
    """Plot restricted signed local rates from the refined Schwarzschild runs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = {
        0: ((200.0, 700.0), (-3.08, -1.94), (20.0, 50.0, 100.0, 200.0)),
        1: ((240.0, 340.0), (-5.15, -2.88), (20.0, 50.0, 100.0, 200.0)),
        2: ((310.0, 420.0), (-7.2, -3.85), (20.0, 50.0, 100.0, 200.0)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    rows: list[dict] = []
    for axis, case in zip(axes, SCHWARZSCHILD_CASES):
        result = load_sds_result(Path(raw_dir) / case.filename)
        x_limits, y_limits, radii = configurations[case.ell]
        colors = plt.get_cmap("viridis")(
            np.linspace(0.08, 0.88, len(radii))
        )
        for color, radius in zip(colors, radii):
            times, signed_rate, _ = _rate_series(
                result, radius, smoothing_width=60.0
            )
            mask = _display_mask(times, signed_rate, x_limits, y_limits)
            axis.plot(
                times[mask],
                signed_rate[mask],
                color=color,
                linewidth=1.35,
                label=rf"$r/M={radius:g}$",
            )
            rows.extend(
                {
                    "ell": case.ell,
                    "resolution": case.resolution,
                    "observer": f"r/M={radius:g}",
                    "U_over_M": float(time),
                    "n_eff": float(rate),
                }
                for time, rate in zip(times[mask], signed_rate[mask])
            )
        times, signed_rate, _ = _rate_series(
            result, None, smoothing_width=60.0
        )
        mask = _display_mask(times, signed_rate, x_limits, y_limits)
        axis.plot(
            times[mask],
            signed_rate[mask],
            color="black",
            linewidth=2.0,
            label=r"$\mathscr{I}^{+}$",
        )
        axis.axhline(
            -(case.ell + 2),
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        axis.axhline(
            -(2 * case.ell + 3),
            color="0.4",
            linestyle=":",
            linewidth=0.9,
        )
        axis.set(
            xlim=x_limits,
            ylim=y_limits,
            xlabel=r"$U/M$",
            title=rf"$\ell={case.ell}$, $N={case.resolution}$",
        )
        axis.grid(alpha=0.24)
    axes[0].set_ylabel(r"$n_{\rm eff}=d\ln|u|/d\ln U$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8.5,
    )
    fig.suptitle("Higher-resolution Schwarzschild finite-radius tail rates")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    path = output_dir / "schwarzschild_high_resolution_rates.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    _write_rows(
        output_dir / "schwarzschild_high_resolution_rates.csv", rows
    )
    return path


def plot_schwarzschild_resolution_comparison(
    raw_dir: Path,
    baseline_root: Path,
    output_dir: Path,
) -> Path:
    """Compare the previously plotted finite-radius rate to its refinement."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = {
        0: ((150.0, 700.0), (-3.15, -2.65), 20.0),
        1: ((180.0, 340.0), (-5.3, -4.2), 20.0),
        2: ((260.0, 420.0), (-7.5, -5.5), 20.0),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    rows: list[dict] = []
    for axis, case in zip(axes, SCHWARZSCHILD_CASES):
        high = load_sds_result(Path(raw_dir) / case.filename)
        baseline = load_sds_result(
            Path(baseline_root)
            / f"ell{case.ell}"
            / "schwarzschild.npz"
        )
        x_limits, y_limits, radius = configurations[case.ell]
        base_times, base_rate, _ = _snapshot_rate_series(
            baseline,
            radius,
            smoothing_width=60.0,
        )
        high_times, high_rate, _ = _rate_series(
            high,
            radius,
            smoothing_width=60.0,
        )
        base_mask = _display_mask(
            base_times, base_rate, x_limits, y_limits
        )
        high_mask = _display_mask(
            high_times, high_rate, x_limits, y_limits
        )
        baseline_resolution = int(
            baseline.metadata["numerical"]["resolution"]
        )
        axis.plot(
            base_times[base_mask],
            base_rate[base_mask],
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            label=rf"$N={baseline_resolution}$",
        )
        axis.plot(
            high_times[high_mask],
            high_rate[high_mask],
            color="tab:blue",
            linewidth=1.5,
            label=rf"$N={case.resolution}$",
        )
        axis.axhline(
            -(2 * case.ell + 3),
            color="black",
            linestyle=":",
            linewidth=0.9,
            label="finite-radius target",
        )
        common = base_times[base_mask]
        if common.size:
            interpolated_high = np.interp(
                common,
                high_times[np.isfinite(high_rate)],
                high_rate[np.isfinite(high_rate)],
            )
            difference = np.abs(base_rate[base_mask] - interpolated_high)
            finite = np.isfinite(difference)
            rows.append(
                {
                    "ell": case.ell,
                    "radius_over_M": radius,
                    "baseline_resolution": baseline_resolution,
                    "refined_resolution": case.resolution,
                    "start_U_over_M": x_limits[0],
                    "end_U_over_M": x_limits[1],
                    "median_absolute_rate_difference": float(
                        np.median(difference[finite])
                    )
                    if np.any(finite)
                    else float("nan"),
                    "maximum_absolute_rate_difference": float(
                        np.max(difference[finite])
                    )
                    if np.any(finite)
                    else float("nan"),
                }
            )
        axis.set(
            xlim=x_limits,
            ylim=y_limits,
            xlabel=r"$U/M$",
            title=rf"$\ell={case.ell}$ at $r=20M$",
        )
        axis.grid(alpha=0.24)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"$n_{\rm eff}=d\ln|u|/d\ln U$")
    fig.suptitle("Resolution check for finite-radius Schwarzschild rates")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = output_dir / "schwarzschild_rate_resolution_comparison.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    _write_rows(
        output_dir / "schwarzschild_rate_resolution_comparison.csv",
        rows,
    )
    return path


def analyze_sds_crossover(
    raw_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Create multi-radius ell=1 SdS rate and crossover-time figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = SDS_ELL1_CASES
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharex=True, sharey=True)
    radius_colors = {
        4.0: "tab:purple",
        8.0: "tab:blue",
        16.0: "tab:green",
        None: "black",
    }
    rate_rows: list[dict] = []
    transition_rows: list[dict] = []

    for axis, case in zip(axes.ravel(), cases):
        result = load_sds_result(Path(raw_dir) / case.filename)
        model = SdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=case.ell,
        )
        kappa = sds_horizons(model).kappa_cosmological
        for radius in (*SDS_RADII, None):
            times, _, exponential = _envelope_rate_series(
                result,
                radius,
                smoothing_width=max(20.0, 0.80 / kappa),
            )
            scaled = kappa * times
            normalized = exponential / kappa
            mask = _display_mask(
                scaled, normalized, (0.55, 5.0), (0.35, 6.0)
            )
            label = (
                r"$\mathcal{H}_c^+$"
                if radius is None
                else rf"$r/M={radius:g}$"
            )
            plot_x, plot_rate = _plot_values(
                scaled, normalized, (0.55, 5.0), (0.35, 6.0)
            )
            axis.plot(
                plot_x,
                plot_rate,
                color=radius_colors[radius],
                linewidth=1.25 if radius is not None else 1.9,
                label=label,
            )
            power = case.ell + 2 if radius is None else 2 * case.ell + 3
            crossing = decay_rate_transition_time(
                scaled,
                normalized,
                float(power),
                case.ell,
            )
            if np.isfinite(crossing):
                axis.axvline(
                    crossing,
                    color=radius_colors[radius],
                    linewidth=0.75,
                    alpha=0.45,
                )
            transition_rows.append(
                {
                    "ell": case.ell,
                    "L_over_M": case.length,
                    "observer": "cosmological_horizon"
                    if radius is None
                    else f"r/M={radius:g}",
                    "kappa_c": kappa,
                    "kappa_U_transition": crossing,
                    "U_transition_over_M": crossing / kappa
                    if np.isfinite(crossing)
                    else float("nan"),
                    "criterion": (
                        "first sustained interval closer to ell than to "
                        "p/(kappa_c U), persisting through the resolved tail"
                    ),
                }
            )
            rate_rows.extend(
                {
                    "ell": case.ell,
                    "L_over_M": case.length,
                    "observer": "cosmological_horizon"
                    if radius is None
                    else f"r/M={radius:g}",
                    "kappa_c_U": float(time),
                    "gamma_eff_over_kappa_c": float(rate),
                }
                for time, rate in zip(scaled[mask], normalized[mask])
            )

        x = np.linspace(0.55, 5.0, 500)
        axis.plot(
            x,
            5.0 / x,
            color="0.45",
            linestyle="--",
            linewidth=0.9,
            label=r"Schwarzschild finite-$r$: $5/(\kappa_cU)$",
        )
        axis.plot(
            x,
            3.0 / x,
            color="0.65",
            linestyle=":",
            linewidth=0.9,
            label=r"Schwarzschild outer: $3/(\kappa_cU)$",
        )
        axis.axhline(
            1.0,
            color="black",
            linestyle="-.",
            linewidth=0.9,
            label=r"SdS target: $\gamma/\kappa_c=1$",
        )
        axis.set_title(rf"$L/M={case.length:g}$")
        axis.grid(alpha=0.24)

    for axis in axes[-1]:
        axis.set_xlabel(r"$\kappa_c U$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8.0,
    )
    fig.suptitle(
        r"Finite-$L$ dipole transition: Schwarzschild power tail to SdS exponential"
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.96))
    rates_path = output_dir / "sds_ell1_multiradius_rate_transition.png"
    fig.savefig(rates_path, dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    markers = {"r/M=4": "o", "r/M=8": "s", "r/M=16": "^",
               "cosmological_horizon": "D"}
    for observer in markers:
        rows = [
            row
            for row in transition_rows
            if row["observer"] == observer
            and np.isfinite(row["U_transition_over_M"])
        ]
        if not rows:
            continue
        lengths = np.asarray([row["L_over_M"] for row in rows], dtype=float)
        physical = np.asarray(
            [row["U_transition_over_M"] for row in rows], dtype=float
        )
        scaled = np.asarray(
            [row["kappa_U_transition"] for row in rows], dtype=float
        )
        label = (
            r"$\mathcal{H}_c^+$"
            if observer == "cosmological_horizon"
            else f"${observer}$"
        )
        axes[0].plot(
            lengths,
            physical,
            marker=markers[observer],
            label=label,
        )
        axes[1].plot(
            lengths,
            scaled,
            marker=markers[observer],
            label=label,
        )
    axes[0].set(
        xlabel=r"$L/M$",
        ylabel=r"$U_{\rm cross}/M$",
        title="Unscaled crossover time",
    )
    axes[1].set(
        xlabel=r"$L/M$",
        ylabel=r"$\kappa_c U_{\rm cross}$",
        title="Crossover in cosmological units",
    )
    for axis in axes:
        axis.grid(alpha=0.24)
        axis.legend(fontsize=8)
    fig.tight_layout()
    summary_path = output_dir / "sds_ell1_crossover_times.png"
    fig.savefig(summary_path, dpi=260)
    plt.close(fig)
    _write_rows(output_dir / "sds_ell1_local_rates.csv", rate_rows)
    _write_rows(output_dir / "sds_ell1_crossover_times.csv", transition_rows)
    return rates_path, summary_path


def plot_sds_resolution_comparison(
    raw_dir: Path,
    baseline_root: Path,
    output_dir: Path,
) -> Path:
    """Compare N=1024 and N=2048 SdS dipole rates at the outer horizon."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), sharex=True, sharey=True)
    rows: list[dict] = []
    for axis, case in zip(axes.ravel(), SDS_ELL1_CASES):
        high = load_sds_result(Path(raw_dir) / case.filename)
        baseline = load_sds_result(
            Path(baseline_root)
            / "ell1"
            / f"sds_L{case.length:g}.npz"
        )
        model = SdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=1,
        )
        kappa = sds_horizons(model).kappa_cosmological
        smoothing = max(20.0, 0.80 / kappa)
        base_times, _, base_exponential = _envelope_rate_series(
            baseline,
            None,
            smoothing_width=smoothing,
        )
        high_times, _, high_exponential = _envelope_rate_series(
            high,
            None,
            smoothing_width=smoothing,
        )
        base_x = kappa * base_times
        high_x = kappa * high_times
        base_rate = base_exponential / kappa
        high_rate = high_exponential / kappa
        base_mask = _display_mask(
            base_x, base_rate, (0.55, 5.0), (0.35, 6.0)
        )
        high_mask = _display_mask(
            high_x, high_rate, (0.55, 5.0), (0.35, 6.0)
        )
        baseline_resolution = int(
            baseline.metadata["numerical"]["resolution"]
        )
        base_plot_x, base_plot_rate = _plot_values(
            base_x, base_rate, (0.55, 5.0), (0.35, 6.0)
        )
        high_plot_x, high_plot_rate = _plot_values(
            high_x, high_rate, (0.55, 5.0), (0.35, 6.0)
        )
        axis.plot(
            base_plot_x,
            base_plot_rate,
            color="tab:orange",
            linestyle="--",
            linewidth=1.1,
            label=rf"$N={baseline_resolution}$",
        )
        axis.plot(
            high_plot_x,
            high_plot_rate,
            color="tab:blue",
            linewidth=1.45,
            label=rf"$N={case.resolution}$",
        )
        x = np.linspace(0.55, 5.0, 400)
        axis.plot(
            x,
            5.0 / x,
            color="0.45",
            linestyle=":",
            linewidth=0.9,
            label="Schwarzschild",
        )
        axis.axhline(
            1.0,
            color="black",
            linestyle="-.",
            linewidth=0.9,
            label="SdS",
        )
        common = base_x[base_mask]
        if common.size:
            interpolated_high = np.interp(
                common,
                high_x[np.isfinite(high_rate)],
                high_rate[np.isfinite(high_rate)],
            )
            difference = np.abs(
                base_rate[base_mask] - interpolated_high
            )
            finite = np.isfinite(difference)
            rows.append(
                {
                    "ell": 1,
                    "L_over_M": case.length,
                    "observer": "cosmological_horizon",
                    "baseline_resolution": baseline_resolution,
                    "refined_resolution": case.resolution,
                    "median_absolute_normalized_rate_difference": float(
                        np.median(difference[finite])
                    )
                    if np.any(finite)
                    else float("nan"),
                }
            )
        axis.set_title(rf"$L/M={case.length:g}$")
        axis.grid(alpha=0.24)
        axis.legend(fontsize=7.5)
    for axis in axes[-1]:
        axis.set_xlabel(r"$\kappa_c U$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
    fig.suptitle(r"Finite-$L$ dipole rate refinement at $\mathcal{H}_c^+$")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = output_dir / "sds_ell1_rate_resolution_comparison.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    _write_rows(
        output_dir / "sds_ell1_rate_resolution_comparison.csv",
        rows,
    )
    return path


def plot_sds_ell2_l80_refinement(
    raw_dir: Path,
    baseline_root: Path,
    output_dir: Path,
) -> Path:
    """Plot the refined L=80 quadrupole rates and its resolution check."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case = SDS_ELL2_L80_CASE
    high = load_sds_result(Path(raw_dir) / case.filename)
    baseline = load_sds_result(
        Path(baseline_root) / "ell2" / "sds_L80.npz"
    )
    model = SdSParameters(
        mass=1.0,
        cosmological_length=80.0,
        ell=2,
    )
    kappa = sds_horizons(model).kappa_cosmological
    smoothing = max(20.0, 0.80 / kappa)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    colors = {
        4.0: "tab:purple",
        8.0: "tab:blue",
        16.0: "tab:green",
        None: "black",
    }
    rows: list[dict] = []
    for radius in (*SDS_RADII, None):
        times, _, exponential = _envelope_rate_series(
            high,
            radius,
            smoothing_width=smoothing,
        )
        scaled = kappa * times
        normalized = exponential / kappa
        mask = _display_mask(
            scaled,
            normalized,
            (0.55, 5.8),
            (0.45, 8.5),
        )
        label = (
            r"$\mathcal{H}_c^+$"
            if radius is None
            else rf"$r/M={radius:g}$"
        )
        plot_x, plot_rate = _plot_values(
            scaled, normalized, (0.55, 5.8), (0.45, 8.5)
        )
        axes[0].plot(
            plot_x,
            plot_rate,
            color=colors[radius],
            linewidth=1.3 if radius is not None else 1.9,
            label=label,
        )
        power = 4.0 if radius is None else 7.0
        crossing = decay_rate_transition_time(
            scaled,
            normalized,
            power,
            ell=2,
        )
        rows.extend(
            {
                "ell": 2,
                "L_over_M": 80.0,
                "resolution": case.resolution,
                "observer": "cosmological_horizon"
                if radius is None
                else f"r/M={radius:g}",
                "kappa_c_U": float(time),
                "gamma_eff_over_kappa_c": float(rate),
                "kappa_U_transition": crossing,
            }
            for time, rate in zip(scaled[mask], normalized[mask])
        )
    x = np.linspace(0.55, 5.8, 500)
    axes[0].plot(
        x,
        7.0 / x,
        color="0.45",
        linestyle="--",
        linewidth=0.9,
        label=r"Schwarzschild finite-$r$: $7/(\kappa_cU)$",
    )
    axes[0].plot(
        x,
        4.0 / x,
        color="0.65",
        linestyle=":",
        linewidth=0.9,
        label=r"Schwarzschild outer: $4/(\kappa_cU)$",
    )
    axes[0].axhline(
        2.0,
        color="black",
        linestyle="-.",
        linewidth=0.9,
        label=r"SdS target: $\gamma/\kappa_c=2$",
    )
    axes[0].set(
        xlim=(0.55, 5.8),
        ylim=(0.45, 8.5),
        xlabel=r"$\kappa_c U$",
        ylabel=r"$\gamma_{\rm eff}/\kappa_c$",
        title=r"Multi-radius rate, $N=4096$",
    )
    axes[0].legend(fontsize=7.5)

    base_times, _, base_exponential = _envelope_rate_series(
        baseline,
        None,
        smoothing_width=smoothing,
    )
    high_times, _, high_exponential = _envelope_rate_series(
        high,
        None,
        smoothing_width=smoothing,
    )
    base_x = kappa * base_times
    high_x = kappa * high_times
    base_rate = base_exponential / kappa
    high_rate = high_exponential / kappa
    base_mask = _display_mask(
        base_x, base_rate, (0.55, 5.8), (0.45, 8.5)
    )
    high_mask = _display_mask(
        high_x, high_rate, (0.55, 5.8), (0.45, 8.5)
    )
    base_plot_x, base_plot_rate = _plot_values(
        base_x, base_rate, (0.55, 5.8), (0.45, 8.5)
    )
    high_plot_x, high_plot_rate = _plot_values(
        high_x, high_rate, (0.55, 5.8), (0.45, 8.5)
    )
    axes[1].plot(
        base_plot_x,
        base_plot_rate,
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label=r"$N=2048$",
    )
    axes[1].plot(
        high_plot_x,
        high_plot_rate,
        color="tab:blue",
        linewidth=1.45,
        label=r"$N=4096$",
    )
    axes[1].plot(
        x,
        7.0 / x,
        color="0.45",
        linestyle=":",
        linewidth=0.9,
        label="Schwarzschild",
    )
    axes[1].axhline(
        2.0,
        color="black",
        linestyle="-.",
        linewidth=0.9,
        label="SdS",
    )
    axes[1].set(
        xlim=(0.55, 5.8),
        ylim=(0.45, 8.5),
        xlabel=r"$\kappa_c U$",
        title=r"Resolution comparison at $\mathcal{H}_c^+$",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.24)
    fig.suptitle(r"Refined finite-$L$ quadrupole transition, $L/M=80$")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = output_dir / "sds_ell2_L80_rate_refinement.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    _write_rows(output_dir / "sds_ell2_L80_local_rates.csv", rows)
    return path


def create_report(
    raw_dir: Path,
    output_dir: Path,
    *,
    baseline_root: Path = Path("results/sds_scalar/tails/raw"),
) -> list[Path]:
    """Create all figures and tables from completed higher-resolution cases."""

    schwarzschild = plot_high_resolution_schwarzschild(raw_dir, output_dir)
    schwarzschild_resolution = plot_schwarzschild_resolution_comparison(
        raw_dir,
        baseline_root,
        output_dir,
    )
    rates, summary = analyze_sds_crossover(raw_dir, output_dir)
    sds_resolution = plot_sds_resolution_comparison(
        raw_dir,
        baseline_root,
        output_dir,
    )
    sds_ell2 = plot_sds_ell2_l80_refinement(
        raw_dir,
        baseline_root,
        output_dir,
    )
    case_rows = []
    for case in (
        *SCHWARZSCHILD_CASES,
        *SDS_ELL1_CASES,
        SDS_ELL2_L80_CASE,
    ):
        result = load_sds_result(Path(raw_dir) / case.filename)
        case_rows.append(
            {
                "background": case.background,
                "ell": case.ell,
                "L_over_M": case.length,
                "resolution": case.resolution,
                "timestep": case.timestep,
                "end_time": float(result.signal_times[-1]),
                "maximum_constraint_linf": float(
                    np.max(result.constraint_linf)
                ),
                "wall_seconds": float(result.metadata["wall_seconds"]),
                "observer_areal_radii": [
                    float(value)
                    for value in result.observer_areal_radius
                ],
            }
        )
    diagnostics = {
        "definition_signed_power_rate": "d ln|u| / d ln U",
        "definition_normalized_exponential_rate": (
            "-d ln|u| / d(kappa_c U)"
        ),
        "finite_L_rate_estimator": (
            "derivative of a centered sliding RMS amplitude envelope; "
            "phase-insensitive at waveform zero crossings"
        ),
        "schwarzschild_finite_radius_target": "-(2 ell + 3)",
        "schwarzschild_outer_target": "-(ell + 2)",
        "sds_minimal_late_target": "gamma/kappa_c = ell for ell > 0",
        "transition_criterion": (
            "first sustained interval in which gamma_eff/kappa_c is closer "
            "to ell than to p/(kappa_c U), with that classification "
            "dominating the remaining resolved tail"
        ),
        "cases": case_rows,
    }
    path = Path(output_dir) / "diagnostics.json"
    path.write_text(
        json.dumps(json_safe(diagnostics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return [
        schwarzschild,
        schwarzschild_resolution,
        rates,
        summary,
        sds_resolution,
        sds_ell2,
        path,
    ]


def _case_from_args(args: argparse.Namespace) -> ResolutionCase:
    return ResolutionCase(
        background=args.background,
        ell=args.ell,
        resolution=args.resolution,
        timestep=args.timestep,
        end_time=args.end_time,
        length=args.length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Higher-resolution finite-radius tail-rate study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one configured evolution")
    run.add_argument(
        "--background", choices=("schwarzschild", "sds"), required=True
    )
    run.add_argument("--ell", type=int, required=True)
    run.add_argument("--length", type=float)
    run.add_argument("--resolution", type=int, required=True)
    run.add_argument("--timestep", type=float, required=True)
    run.add_argument("--end-time", type=float, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--reuse-existing", action="store_true")

    suite = subparsers.add_parser("suite", help="run a production group")
    suite.add_argument(
        "--group",
        choices=("schwarzschild", "sds-ell1", "all"),
        default="all",
    )
    suite.add_argument("--output-dir", type=Path, required=True)
    suite.add_argument("--reuse-existing", action="store_true")

    report = subparsers.add_parser("report", help="create crossover reports")
    report.add_argument("--raw-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("results/sds_scalar/tails/raw"),
    )
    report_schwarzschild = subparsers.add_parser(
        "report-schwarzschild",
        help="create the Schwarzschild refinement figures only",
    )
    report_schwarzschild.add_argument("--raw-dir", type=Path, required=True)
    report_schwarzschild.add_argument(
        "--output-dir", type=Path, required=True
    )
    report_schwarzschild.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("results/sds_scalar/tails/raw"),
    )
    args = parser.parse_args()

    if args.command == "run":
        print(
            run_resolution_case(
                _case_from_args(args),
                args.output_dir,
                reuse_existing=args.reuse_existing,
            )
        )
    elif args.command == "suite":
        for path in run_suite(
            args.group,
            args.output_dir,
            reuse_existing=args.reuse_existing,
        ):
            print(path)
    elif args.command == "report":
        for path in create_report(
            args.raw_dir,
            args.output_dir,
            baseline_root=args.baseline_root,
        ):
            print(path)
    else:
        print(
            plot_high_resolution_schwarzschild(
                args.raw_dir,
                args.output_dir,
            )
        )
        print(
            plot_schwarzschild_resolution_comparison(
                args.raw_dir,
                args.baseline_root,
                args.output_dir,
            )
        )


if __name__ == "__main__":
    main()
