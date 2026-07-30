"""Pure-mode 3D validation suite and publication-quality comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

from .crossover_final import (
    EnvelopeSettings,
    SweepGrid,
    cosmological_rate,
    envelope_rate,
    observer_index,
    rate_pair,
    retarded_series,
    sweep_transition,
)
from .sds_result import load_sds_result
from .tail_analysis import json_safe
from .three_d_solver import (
    PureMode,
    RealSphericalHarmonicBasis,
    ThreeDNumericalParameters,
    ThreeDSimulationResult,
    load_three_d_result,
    real_spherical_harmonic,
    run_pure_mode_simulation,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("results/three_d_validation")
LENGTH = 80.0
MODES = (PureMode(0, 0), PureMode(1, 1), PureMode(2, 2))
PRODUCTION_RESOLUTION = {0: 2048, 1: 2048, 2: 4096}
PRODUCTION_TIMESTEP = {0: 0.0025, 1: 0.0025, 2: 0.00125}
PRODUCTION_END_TIME = {
    ("sds", 0): 410.4952471239025,
    ("sds", 1): 410.4952471239025,
    ("sds", 2): 492.594296548683,
    ("schwarzschild", 0): 720.0,
    ("schwarzschild", 1): 830.0,
    ("schwarzschild", 2): 440.0,
}
CONVERGENCE_RESOLUTIONS = {
    0: (512, 1024, 2048),
    1: (512, 1024, 2048),
    2: (1024, 2048, 4096),
}
CONVERGENCE_END_TIME = 180.0
COLORS = {0: "#7b3294", 1: "#0571b0", 2: "#ca562c"}
BACKGROUND_LABEL = {
    "schwarzschild": "Schwarzschild",
    "sds": r"Schwarzschild--de Sitter, $L/M=80$",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
        }
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def archive_name(background: str, ell: int, resolution: int) -> str:
    prefix = "sds_L80" if background == "sds" else "schwarzschild"
    return f"{prefix}_ell{ell}_m{ell}_N{resolution}.npz"


def production_path(output_dir: Path, background: str, ell: int) -> Path:
    return (
        Path(output_dir)
        / "raw"
        / archive_name(background, ell, PRODUCTION_RESOLUTION[ell])
    )


def convergence_path(
    output_dir: Path,
    background: str,
    ell: int,
    resolution: int,
) -> Path:
    if resolution == PRODUCTION_RESOLUTION[ell]:
        return production_path(output_dir, background, ell)
    return (
        Path(output_dir)
        / "convergence"
        / "raw"
        / archive_name(background, ell, resolution)
    )


def reference_path(background: str, ell: int) -> Path:
    """Locate the completed 1D run corresponding to one 3D case."""

    if background == "schwarzschild":
        candidates = {
            0: [
                Path(
                    "results/sds_scalar/tails/high_resolution_rates/raw/"
                    "schwarzschild_ell0_N2048.npz"
                )
            ],
            1: [
                Path(
                    "results/sds_scalar/tails/crossover_final/raw/"
                    "schwarzschild_ell1_N2048.npz"
                ),
                Path(
                    "results/sds_scalar/tails/high_resolution_rates/raw/"
                    "schwarzschild_ell1_N2048.npz"
                ),
            ],
            2: [
                Path(
                    "results/sds_scalar/tails/high_resolution_rates/raw/"
                    "schwarzschild_ell2_N4096.npz"
                )
            ],
        }[ell]
    else:
        candidates = {
            0: [Path("results/sds_scalar/tails/raw/ell0/sds_L80.npz")],
            1: [
                Path(
                    "results/sds_scalar/tails/high_resolution_rates/raw/"
                    "sds_ell1_L80_N2048.npz"
                )
            ],
            2: [
                Path(
                    "results/sds_scalar/tails/high_resolution_rates/raw/"
                    "sds_ell2_L80_N4096.npz"
                )
            ],
        }[ell]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No completed 1D reference was found for {background}, ell={ell}."
    )


def _run_case(
    output: Path,
    *,
    background: str,
    mode: PureMode,
    resolution: int,
    timestep: float,
    end_time: float,
    force: bool,
) -> Path:
    if output.exists() and not force:
        LOGGER.info("reusing %s", output)
        return output
    numerical = ThreeDNumericalParameters(
        radial_resolution=resolution,
        angular_ell_max=2,
        timestep=timestep,
        end_time=end_time,
        signal_dt=0.05,
        diagnostic_dt=1.0,
        snapshot_dt=max(20.0, end_time / 8.0),
    )
    result = run_pure_mode_simulation(
        background=background,
        mode=mode,
        numerical=numerical,
        cosmological_length=LENGTH,
    )
    result.save(output)
    return output


def run_suite(output_dir: Path, *, force: bool = False) -> list[Path]:
    """Run production pure modes and shorter radial convergence ladders."""

    output_dir = Path(output_dir)
    paths: list[Path] = []
    for background in ("schwarzschild", "sds"):
        for mode in MODES:
            resolution = PRODUCTION_RESOLUTION[mode.ell]
            paths.append(
                _run_case(
                    production_path(output_dir, background, mode.ell),
                    background=background,
                    mode=mode,
                    resolution=resolution,
                    timestep=PRODUCTION_TIMESTEP[mode.ell],
                    end_time=PRODUCTION_END_TIME[(background, mode.ell)],
                    force=force,
                )
            )
            for lower_resolution in CONVERGENCE_RESOLUTIONS[mode.ell][:-1]:
                paths.append(
                    _run_case(
                        convergence_path(
                            output_dir,
                            background,
                            mode.ell,
                            lower_resolution,
                        ),
                        background=background,
                        mode=mode,
                        resolution=lower_resolution,
                        timestep=PRODUCTION_TIMESTEP[mode.ell],
                        end_time=CONVERGENCE_END_TIME,
                        force=force,
                    )
                )
    return paths


def _observer_for(background: str, ell: int) -> float | None:
    # The archived finite-L monopole has compact-coordinate rather than exact
    # finite-radius observers.  Its cosmological-horizon observer is exact.
    if background == "sds" and ell == 0:
        return None
    return 8.0


def _observer_name(radius: float | None, background: str) -> str:
    if radius is not None:
        return rf"$r/M={radius:g}$"
    return r"$\mathcal{H}_c^+$" if background == "sds" else r"$\mathscr{I}^+$"


def _interpolate_signal(
    source_times: np.ndarray,
    source_signal: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    return np.interp(
        target_times,
        source_times,
        source_signal,
        left=np.nan,
        right=np.nan,
    )


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    denominator = np.linalg.norm(reference[finite])
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(candidate[finite] - reference[finite]) / denominator)


def waveform_error_row(
    result: ThreeDSimulationResult,
    reference,
    *,
    radius: float | None,
    end_time: float | None = None,
) -> dict:
    """Compare one target modal coefficient with its 1D counterpart."""

    three_d = result.as_1d_result()
    times, signal = retarded_series(three_d, radius)
    reference_times, reference_signal = retarded_series(reference, radius)
    aligned = _interpolate_signal(reference_times, reference_signal, times)
    mask = (times >= 20.0) & np.isfinite(aligned)
    if end_time is not None:
        mask &= times <= end_time
    peak = float(np.max(np.abs(reference_signal)))
    difference = signal - aligned
    return {
        "background": result.metadata["background_key"],
        "ell": result.metadata["target_mode"]["ell"],
        "m": result.metadata["target_mode"]["m"],
        "radial_resolution": result.metadata["numerical"]["radial_resolution"],
        "observer": "outer_boundary" if radius is None else f"r/M={radius:g}",
        "comparison_start_U_over_M": 20.0,
        "comparison_end_U_over_M": float(times[mask][-1]),
        "relative_waveform_L2": _relative_l2(aligned[mask], signal[mask]),
        "maximum_error_over_reference_peak": float(
            np.max(np.abs(difference[mask])) / peak
        ),
        "maximum_constraint_linf": float(np.max(result.constraint_linf)),
        "minimum_mode_purity": float(np.min(result.mode_purity)),
        "maximum_off_mode_amplitude": float(
            np.max(result.maximum_off_mode_amplitude)
        ),
    }


def plot_waveforms(
    output_dir: Path,
    results: dict[tuple[str, int], ThreeDSimulationResult],
    references: dict[tuple[str, int], object],
) -> Path:
    """Plot direct 3D-vs-1D waveform overlays for all six cases."""

    _style()
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.0), sharex="col")
    for row, ell in enumerate(range(3)):
        for column, background in enumerate(("schwarzschild", "sds")):
            axis = axes[row, column]
            result = results[(background, ell)]
            reference = references[(background, ell)]
            radius = _observer_for(background, ell)
            times, signal = retarded_series(result.as_1d_result(), radius)
            reference_times, reference_signal = retarded_series(reference, radius)
            peak = float(np.max(np.abs(reference_signal)))
            axis.plot(
                reference_times,
                reference_signal / peak,
                color="0.72",
                linewidth=3.0,
                label="1D Chebyshev/Dedalus",
                zorder=1,
            )
            axis.plot(
                times,
                signal / peak,
                color=COLORS[ell],
                linewidth=1.05,
                label=rf"3D $Y_{{{ell}{ell}}}^{{\rm R}}$ coefficient",
                zorder=2,
            )
            row_data = waveform_error_row(result, reference, radius=radius)
            axis.text(
                0.975,
                0.92,
                rf"$\|\Delta u\|_2/\|u\|_2={row_data['relative_waveform_L2']:.1e}$",
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=9,
                color=COLORS[ell],
            )
            axis.set_yscale("symlog", linthresh=2e-8, linscale=0.7)
            axis.set_yticks((-1.0, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 1.0))
            axis.set_ylim(-1.5, 1.5)
            axis.grid(alpha=0.18)
            axis.set_ylabel(rf"$u_{{{ell}{ell}}}/\max|u^{{1D}}|$")
            if row == 0:
                axis.set_title(BACKGROUND_LABEL[background])
            if row == 2:
                axis.set_xlabel(r"geometric retarded time $U/M$")
            axis.text(
                0.025,
                0.08,
                _observer_name(radius, background),
                transform=axis.transAxes,
                fontsize=9,
            )
            if row == 0:
                axis.legend(loc="lower right", fontsize=8.5)
    fig.suptitle(
        "Pure spherical-harmonic packets: independent 3D evolution against 1D",
        y=1.002,
    )
    fig.tight_layout()
    path = Path(output_dir) / "pure_mode_waveform_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _rate_settings(ell: int) -> EnvelopeSettings:
    return EnvelopeSettings(45.0 if ell in (0, 2) else 30.0, 0.5)


def _clipped(values: np.ndarray, low: float, high: float) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    output[(output < low) | (output > high) | ~np.isfinite(output)] = np.nan
    return output


def plot_schwarzschild_decay_rates(
    output_dir: Path,
    results: dict[tuple[str, int], ThreeDSimulationResult],
    references: dict[tuple[str, int], object],
) -> tuple[Path, list[dict]]:
    """Compare finite-radius Price indices at ``r=8M``."""

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.75), sharex=False)
    rows: list[dict] = []
    fit_windows = {0: (350.0, 650.0), 1: (230.0, 480.0), 2: (270.0, 410.0)}
    display_windows = {
        0: ((120.0, 680.0), (2.72, 3.20)),
        1: ((200.0, 520.0), (4.62, 5.28)),
        2: ((260.0, 420.0), (6.45, 7.35)),
    }
    for axis, ell in zip(axes, range(3)):
        result = results[("schwarzschild", ell)].as_1d_result()
        reference = references[("schwarzschild", ell)]
        settings = _rate_settings(ell)
        target = float(2 * ell + 3)
        measured: dict[str, float] = {}
        for label, data, style, width in (
            ("3D", result, "-", 1.6),
            ("1D", reference, (0, (4, 2)), 1.25),
        ):
            times, signal = retarded_series(data, 8.0)
            rate, _ = envelope_rate(times, signal, settings)
            index = times * rate
            axis.plot(
                times,
                _clipped(index, 0.0, target + 2.5),
                color=COLORS[ell] if label == "3D" else "0.25",
                linestyle=style,
                linewidth=width,
                label=label,
            )
            start, stop = fit_windows[ell]
            mask = (
                (times >= start)
                & (times <= stop)
                & np.isfinite(index)
                & (index > 0.0)
                & (index < target + 2.5)
            )
            measured[label] = float(np.median(index[mask]))
        axis.axhline(target, color="black", linewidth=0.9, linestyle=":")
        axis.axhspan(0.95 * target, 1.05 * target, color=COLORS[ell], alpha=0.08)
        axis.set(
            title=rf"$\ell={ell}$, target $p={int(target)}$",
            xlabel=r"$U/M$",
            ylabel=r"$p_{\rm eff}=U\gamma_{\rm eff}$",
            xlim=display_windows[ell][0],
            ylim=display_windows[ell][1],
        )
        axis.grid(alpha=0.2)
        axis.text(
            0.96,
            0.08,
            rf"3D {measured['3D']:.3f}" + "\n" + rf"1D {measured['1D']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        axis.legend(fontsize=8.5, loc="upper right")
        rows.append(
            {
                "background": "schwarzschild",
                "ell": ell,
                "observer": "r/M=8",
                "target_power_index": target,
                "measurement_start_U_over_M": fit_windows[ell][0],
                "measurement_end_U_over_M": fit_windows[ell][1],
                "three_d_median_power_index": measured["3D"],
                "one_d_median_power_index": measured["1D"],
            }
        )
    fig.suptitle(
        r"Finite-radius Price decay from the $Y_{\ell\ell}^{\rm R}$ coefficients",
        y=1.02,
    )
    fig.tight_layout()
    path = Path(output_dir) / "schwarzschild_decay_rate_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path, rows


def _transition_rows_and_summaries(
    results: dict[tuple[str, int], ThreeDSimulationResult],
    references: dict[tuple[str, int], object],
) -> tuple[list[dict], dict[tuple[str, int], object]]:
    rows: list[dict] = [
        {
            "implementation": "3D",
            "ell": 0,
            "m": 0,
            "L_over_M": LENGTH,
            "observer": "cosmological_horizon",
            "status": "not_applicable_nonzero_monopole_limit",
            "reason": (
                "The minimally coupled ell=0 field approaches a nonzero "
                "constant, so a decay-rate transition interval is undefined."
            ),
        }
    ]
    summaries: dict[tuple[str, int], object] = {}
    kappa = cosmological_rate(LENGTH)
    grid = SweepGrid()
    for ell in (1, 2):
        three_d_summary, _ = sweep_transition(
            results[("sds", ell)].as_1d_result(),
            results[("schwarzschild", ell)].as_1d_result(),
            8.0,
            ell=ell,
            length=LENGTH,
            kappa=kappa,
            grid=grid,
        )
        one_d_summary, _ = sweep_transition(
            references[("sds", ell)],
            references[("schwarzschild", ell)],
            8.0,
            ell=ell,
            length=LENGTH,
            kappa=kappa,
            grid=grid,
        )
        summaries[("3D", ell)] = three_d_summary
        summaries[("1D", ell)] = one_d_summary
        for implementation, summary in (
            ("3D", three_d_summary),
            ("1D", one_d_summary),
        ):
            row = summary.as_row()
            row["implementation"] = implementation
            row["m"] = ell if implementation == "3D" else "mode-reduced"
            rows.append(row)
    return rows, summaries


def plot_transition_intervals(
    output_dir: Path,
    results: dict[tuple[str, int], ThreeDSimulationResult],
    references: dict[tuple[str, int], object],
    summaries: dict[tuple[str, int], object],
) -> Path:
    """Compare the 3D and 1D finite-L transition intervals at ``r=8M``."""

    _style()
    kappa = cosmological_rate(LENGTH)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.25))
    for axis, ell in zip(axes, (1, 2)):
        settings = _rate_settings(ell)
        scaled_3d, rate_3d, reference_3d = rate_pair(
            results[("sds", ell)].as_1d_result(),
            results[("schwarzschild", ell)].as_1d_result(),
            8.0,
            kappa,
            settings,
        )
        scaled_1d, rate_1d, _ = rate_pair(
            references[("sds", ell)],
            references[("schwarzschild", ell)],
            8.0,
            kappa,
            settings,
        )
        axis.axhspan(
            0.9 * ell,
            1.1 * ell,
            color=COLORS[ell],
            alpha=0.10,
            label=r"$\ell\pm10\%$",
        )
        axis.axhline(float(ell), color="black", linewidth=0.9, linestyle=":")
        axis.plot(
            scaled_1d,
            _clipped(rate_1d, -0.2, ell + 3.5),
            color="0.45",
            linewidth=3.0,
            alpha=0.55,
            label="1D SdS",
        )
        axis.plot(
            scaled_3d,
            _clipped(rate_3d, -0.2, ell + 3.5),
            color=COLORS[ell],
            linewidth=1.45,
            label=rf"3D $Y_{{{ell}{ell}}}^{{\rm R}}$",
        )
        axis.plot(
            scaled_3d,
            _clipped(reference_3d, -0.2, ell + 3.5),
            color=COLORS[ell],
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            alpha=0.65,
            label="3D Schwarzschild reference",
        )
        for implementation, y_location, color in (
            ("1D", 0.03, "0.35"),
            ("3D", 0.095, COLORS[ell]),
        ):
            summary = summaries[(implementation, ell)]
            if summary.departures and summary.entries:
                departure = float(np.median(summary.departures))
                entry = float(np.median(summary.entries))
                axis.plot(
                    [departure, entry],
                    [y_location, y_location],
                    transform=axis.get_xaxis_transform(),
                    color=color,
                    linewidth=5.0,
                    solid_capstyle="butt",
                    label=f"{implementation} transition interval",
                )
        axis.set(
            xlim=(0.35, 5.2 if ell == 2 else 4.6),
            ylim=(-0.1, ell + 3.1),
            xlabel=r"$\kappa_c U$",
            ylabel=r"$\gamma_{\rm eff}/\kappa_c$",
            title=rf"$\ell={ell}$ at $r=8M$",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7.8, ncol=2, loc="upper right")
    fig.suptitle(
        r"Schwarzschild-to-de Sitter transition: 3D reproduces the 1D interval",
        y=1.02,
    )
    fig.tight_layout()
    path = Path(output_dir) / "transition_interval_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def convergence_rows(
    output_dir: Path,
    references: dict[tuple[str, int], object],
) -> list[dict]:
    """Measure every radial ladder against 1D and against the finest 3D run."""

    rows: list[dict] = []
    for background in ("schwarzschild", "sds"):
        for ell in range(3):
            radius = _observer_for(background, ell)
            finest = load_three_d_result(
                convergence_path(
                    output_dir,
                    background,
                    ell,
                    PRODUCTION_RESOLUTION[ell],
                )
            )
            finest_1d = finest.as_1d_result()
            finest_times, finest_signal = retarded_series(finest_1d, radius)
            for resolution in CONVERGENCE_RESOLUTIONS[ell]:
                result = load_three_d_result(
                    convergence_path(
                        output_dir,
                        background,
                        ell,
                        resolution,
                    )
                )
                row = waveform_error_row(
                    result,
                    references[(background, ell)],
                    radius=radius,
                    end_time=CONVERGENCE_END_TIME - 5.0,
                )
                times, signal = retarded_series(result.as_1d_result(), radius)
                common_end = min(
                    CONVERGENCE_END_TIME - 5.0,
                    float(times[-1]),
                    float(finest_times[-1]),
                )
                mask = (times >= 20.0) & (times <= common_end)
                aligned_finest = _interpolate_signal(
                    finest_times, finest_signal, times[mask]
                )
                row["relative_L2_to_finest_3D"] = _relative_l2(
                    aligned_finest,
                    signal[mask],
                )
                row["comparison_kind"] = "matched-timestep radial refinement"
                rows.append(row)
    return rows


def plot_convergence_constraints_purity(
    output_dir: Path,
    rows: list[dict],
) -> Path:
    """Summarize radial convergence, constraints, and angular leakage."""

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.15))
    axis = axes[0]
    for background, linestyle, marker in (
        ("schwarzschild", "-", "o"),
        ("sds", "--", "s"),
    ):
        for ell in range(3):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["background"] == background and int(row["ell"]) == ell
                ],
                key=lambda row: int(row["radial_resolution"]),
            )
            axis.loglog(
                [row["radial_resolution"] for row in selected],
                [
                    row["relative_L2_to_finest_3D"]
                    if float(row["relative_L2_to_finest_3D"]) > 0.0
                    else np.nan
                    for row in selected
                ],
                color=COLORS[ell],
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
            )
    axis.set(
        xlabel="radial points $N_\\rho$",
        ylabel=r"$\|u_N-u_{\rm finest}\|_2/\|u_{\rm finest}\|_2$",
        title="Matched-timestep radial self-convergence",
    )
    axis.grid(which="both", alpha=0.2)
    mode_handles = [
        Line2D([0], [0], color=COLORS[ell], linewidth=2, label=rf"$\ell={ell}$")
        for ell in range(3)
    ]
    background_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=style,
            marker=marker,
            label=BACKGROUND_LABEL[background],
        )
        for background, style, marker in (
            ("schwarzschild", "-", "o"),
            ("sds", "--", "s"),
        )
    ]
    first_legend = axis.legend(handles=mode_handles, fontsize=8, loc="lower left")
    axis.add_artist(first_legend)
    axis.legend(handles=background_handles, fontsize=7.8, loc="upper right")

    axis = axes[1]
    for background, linestyle, marker in (
        ("schwarzschild", "-", "o"),
        ("sds", "--", "s"),
    ):
        for ell in range(3):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["background"] == background and int(row["ell"]) == ell
                ],
                key=lambda row: int(row["radial_resolution"]),
            )
            axis.loglog(
                [row["radial_resolution"] for row in selected],
                [row["maximum_constraint_linf"] for row in selected],
                color=COLORS[ell],
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
            )
    axis.set(
        xlabel="radial points $N_\\rho$",
        ylabel=r"$\max_\tau\|\psi-\partial_\rho u\|_\infty$",
        title=r"Reduction constraint stays below $4.4\times10^{-10}$",
    )
    axis.grid(which="both", alpha=0.2)
    for plotted_axis in axes[:2]:
        plotted_axis.set_xticks((512, 1024, 2048, 4096))
        plotted_axis.xaxis.set_major_formatter(ScalarFormatter())
        plotted_axis.xaxis.set_minor_formatter(NullFormatter())

    axis = axes[2]
    production = [
        row
        for row in rows
        if int(row["radial_resolution"]) == PRODUCTION_RESOLUTION[int(row["ell"])]
    ]
    x = np.arange(3)
    width = 0.34
    for offset, background, hatch in (
        (-width / 2, "schwarzschild", ""),
        (width / 2, "sds", "//"),
    ):
        values = []
        for ell in range(3):
            row = next(
                item
                for item in production
                if item["background"] == background and int(item["ell"]) == ell
            )
            values.append(max(float(row["maximum_off_mode_amplitude"]), 1e-18))
        axis.bar(
            x + offset,
            values,
            width,
            color=[COLORS[ell] for ell in range(3)],
            alpha=0.78,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
            label=BACKGROUND_LABEL[background],
        )
    axis.axhline(np.finfo(float).eps, color="black", linestyle=":", linewidth=1.0)
    axis.set_yscale("log")
    axis.set(
        xticks=x,
        xticklabels=[r"$Y_{00}$", r"$Y_{11}^{\rm R}$", r"$Y_{22}^{\rm R}$"],
        ylabel="largest non-target modal amplitude",
        title="Angular purity at roundoff",
        ylim=(1e-18, 5e-14),
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", which="both", alpha=0.2)
    fig.tight_layout()
    path = Path(output_dir) / "convergence_constraints_mode_purity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_angular_modes(output_dir: Path) -> Path:
    """Visualize the three real angular profiles and their transform leakage."""

    _style()
    longitude = np.linspace(-np.pi, np.pi, 361)
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    theta = np.pi / 2.0 - latitude_grid
    basis = RealSphericalHarmonicBasis(2)

    fig = plt.figure(figsize=(13.0, 6.8))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.58), hspace=0.22)
    leakage = []
    for column, mode in enumerate(MODES):
        axis = fig.add_subplot(grid[0, column], projection="mollweide")
        values = real_spherical_harmonic(
            mode.ell,
            mode.m,
            theta,
            longitude_grid,
        )
        scale = float(np.max(np.abs(values)))
        image = axis.pcolormesh(
            longitude_grid,
            latitude_grid,
            values,
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            shading="auto",
            rasterized=True,
        )
        axis.grid(alpha=0.22)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        title = (
            r"$Y_{00}$"
            if mode.ell == 0
            else rf"$Y_{{{mode.ell}{mode.m}}}^{{\rm R}}$"
        )
        axis.set_title(title, pad=8)
        fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.05, shrink=0.72)
        leakage.append(
            basis.roundtrip_diagnostics(
                mode.ell, mode.m
            )["maximum_off_mode_amplitude"]
        )

    axis = fig.add_subplot(grid[1, :])
    positions = np.arange(3)
    axis.bar(
        positions,
        np.maximum(leakage, 1e-18),
        color=[COLORS[ell] for ell in range(3)],
        width=0.58,
    )
    axis.axhline(
        np.finfo(float).eps,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="double-precision epsilon",
    )
    axis.set_yscale("log")
    axis.set(
        xticks=positions,
        xticklabels=[r"$Y_{00}$", r"$Y_{11}^{\rm R}$", r"$Y_{22}^{\rm R}$"],
        ylabel=r"$\max_{\rm wrong}|c_{\ell m}|$",
        ylim=(1e-18, 5e-14),
        title=(
            "Gauss--Legendre/Fourier synthesis followed by spherical-harmonic "
            "projection"
        ),
    )
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend()
    fig.suptitle("Pure angular initial data and measured mode purity", y=1.01)
    path = Path(output_dir) / "pure_spherical_harmonics_and_mode_purity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def create_report(output_dir: Path) -> list[Path]:
    """Analyze every completed 3D archive and write plots, tables, diagnostics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        (background, ell): load_three_d_result(
            production_path(output_dir, background, ell)
        )
        for background in ("schwarzschild", "sds")
        for ell in range(3)
    }
    references = {
        (background, ell): load_sds_result(reference_path(background, ell))
        for background in ("schwarzschild", "sds")
        for ell in range(3)
    }

    waveform_rows = [
        waveform_error_row(
            results[(background, ell)],
            references[(background, ell)],
            radius=_observer_for(background, ell),
        )
        for background in ("schwarzschild", "sds")
        for ell in range(3)
    ]
    _write_rows(output_dir / "waveform_agreement.csv", waveform_rows)

    waveform_figure = plot_waveforms(output_dir, results, references)
    rate_figure, rate_rows = plot_schwarzschild_decay_rates(
        output_dir, results, references
    )
    _write_rows(output_dir / "decay_rate_summary.csv", rate_rows)

    transition_rows, summaries = _transition_rows_and_summaries(
        results, references
    )
    _write_rows(output_dir / "transition_intervals.csv", transition_rows)
    transition_figure = plot_transition_intervals(
        output_dir, results, references, summaries
    )

    radial_rows = convergence_rows(output_dir, references)
    _write_rows(output_dir / "radial_convergence.csv", radial_rows)
    convergence_figure = plot_convergence_constraints_purity(
        output_dir, radial_rows
    )
    angular_figure = plot_angular_modes(output_dir)

    diagnostics = {
        "purpose": (
            "First 3D validation stage requested by Professor Zenginoglu: pure "
            "Y_lm packets before angularly mixed data."
        ),
        "independence": (
            "3D uses real spherical harmonics, eighth-order uniform-rho finite "
            "differences, and explicit RK4.  The 1D references use Chebyshev "
            "collocation and Dedalus RK222."
        ),
        "cases": [
            {
                "background": background,
                "ell": ell,
                "m": ell,
                "archive": str(production_path(output_dir, background, ell)),
                "one_d_reference": str(reference_path(background, ell)),
                "metadata": results[(background, ell)].metadata,
            }
            for background in ("schwarzschild", "sds")
            for ell in range(3)
        ],
        "waveform_agreement": waveform_rows,
        "schwarzschild_decay_rates": rate_rows,
        "transition_intervals": transition_rows,
        "radial_convergence": radial_rows,
        "scope_boundary": (
            "No mixed-mode result is claimed here.  That is the next stage "
            "after these pure-mode comparisons pass."
        ),
    }
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(json_safe(diagnostics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return [
        waveform_figure,
        rate_figure,
        transition_figure,
        convergence_figure,
        angular_figure,
        diagnostics_path,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and analyze the pure-mode 3D black-hole validation suite."
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite = subparsers.add_parser("suite", help="run every production and convergence case")
    suite.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    suite.add_argument("--force", action="store_true")
    suite.add_argument(
        "--skip-analysis",
        action="store_true",
        help="only create raw archives",
    )

    case = subparsers.add_parser("run", help="run one pure-mode case")
    case.add_argument("--background", choices=("sds", "schwarzschild"), required=True)
    case.add_argument("--ell", type=int, choices=(0, 1, 2), required=True)
    case.add_argument("--m", type=int)
    case.add_argument("--resolution", type=int, required=True)
    case.add_argument("--timestep", type=float, required=True)
    case.add_argument("--end-time", type=float, required=True)
    case.add_argument("--output", type=Path, required=True)
    case.add_argument("--force", action="store_true")

    analyze = subparsers.add_parser("analyze", help="rebuild all plots and tables")
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    if args.command == "suite":
        for path in run_suite(args.output_dir, force=args.force):
            print(path)
        if not args.skip_analysis:
            for path in create_report(args.output_dir):
                print(path)
    elif args.command == "run":
        mode = PureMode(args.ell, args.ell if args.m is None else args.m)
        path = _run_case(
            args.output,
            background=args.background,
            mode=mode,
            resolution=args.resolution,
            timestep=args.timestep,
            end_time=args.end_time,
            force=args.force,
        )
        print(path)
    else:
        for path in create_report(args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
