"""Diagnostics and plots for Schwarzschild-de Sitter scalar evolutions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .sds_model import (
    BRIDGE_LABELS,
    SdSParameters,
    areal_radius,
    bridge_boost,
    characteristic_speeds,
    propagation_coefficient,
    sds_horizons,
    turning_radius,
)
from .sds_solver import SdSSimulationResult


def _observer_index(result: SdSSimulationResult, location: float) -> int:
    return int(np.argmin(np.abs(result.observer_rho - location)))


def _model_from_result(result: SdSSimulationResult) -> SdSParameters:
    return SdSParameters(**result.metadata["model"])


def _bridge_from_result(result: SdSSimulationResult) -> str:
    return str(result.metadata["numerical"]["bridge"])


def fit_log_decay(
    result: SdSSimulationResult,
    observer: float = 1.0,
    start_fraction: float = 0.55,
    end_fraction: float = 0.95,
) -> dict[str, float | int]:
    """Fit log(abs(u)) = slope*t + intercept over the late signal."""

    index = _observer_index(result, observer)
    end_time = result.signal_times[-1]
    start = start_fraction * end_time
    end = end_fraction * end_time
    mask = (result.signal_times >= start) & (result.signal_times <= end)
    times = result.signal_times[mask]
    amplitude = np.abs(result.signals[mask, index])
    floor = 100.0 * np.finfo(float).eps * np.max(
        np.abs(result.signals[:, index])
    )
    valid = amplitude > floor
    times = times[valid]
    amplitude = amplitude[valid]
    if len(times) < 20:
        return {
            "observer": float(result.observer_rho[index]),
            "start": float(start),
            "end": float(end),
            "slope": float("nan"),
            "r_squared": float("nan"),
            "points": int(len(times)),
        }
    slope, intercept = np.polyfit(times, np.log(amplitude), 1)
    fitted = slope * times + intercept
    y = np.log(amplitude)
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return {
        "observer": float(result.observer_rho[index]),
        "start": float(start),
        "end": float(end),
        "slope": float(slope),
        "r_squared": float(r_squared),
        "points": int(len(times)),
    }


def write_sds_diagnostics(
    result: SdSSimulationResult, output_dir: Path
) -> dict:
    """Write JSON diagnostics and waveform CSV for one SdS result."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = _bridge_from_result(result)
    peaks = []
    for index, rho in enumerate(result.observer_rho):
        peak = int(np.argmax(np.abs(result.signals[:, index])))
        peaks.append(
            {
                "rho": float(rho),
                "areal_radius": float(result.observer_areal_radius[index]),
                "time": float(result.signal_times[peak]),
                "value": float(result.signals[peak, index]),
                "abs_value": float(abs(result.signals[peak, index])),
            }
        )

    diagnostics = {
        "bridge": bridge,
        "label": BRIDGE_LABELS.get(bridge, bridge),
        "model": result.metadata["model"],
        "horizons": result.metadata["horizons"],
        "constraint": {
            "maximum_linf": float(np.max(result.constraint_linf)),
            "final_linf": float(result.constraint_linf[-1]),
            "maximum_l2": float(np.max(result.constraint_l2)),
        },
        "peaks": peaks,
        "late_log_decay_cosmological_horizon": fit_log_decay(result, 1.0),
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2)

    with (output_dir / "waveforms.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "tau",
                *[
                    f"u_rho_{rho:g}_r_{radius:.8g}"
                    for rho, radius in zip(
                        result.observer_rho, result.observer_areal_radius
                    )
                ],
            ]
        )
        writer.writerows(
            np.column_stack((result.signal_times, result.signals)).tolist()
        )
    return diagnostics


def create_sds_plots(result: SdSSimulationResult, output_dir: Path) -> None:
    """Create single-run spacetime, waveform, and constraint plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = _bridge_from_result(result)
    label = BRIDGE_LABELS.get(bridge, bridge)

    order = np.argsort(result.areal_radius)
    radius = result.areal_radius[order]
    snapshots = result.u_snapshots[:, order]
    time_end = min(result.snapshot_times[-1], 150.0)
    mask = result.snapshot_times <= time_end
    plotted = snapshots[mask]
    color_limit = float(np.max(np.abs(plotted))) if plotted.size else 1.0
    if color_limit == 0:
        color_limit = 1.0

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    image = axis.pcolormesh(
        radius,
        result.snapshot_times[mask],
        plotted,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
        rasterized=True,
    )
    fig.colorbar(image, ax=axis, label=r"$u(\tau,r)$")
    axis.set(
        xlabel=r"areal radius $r/M$",
        ylabel=r"$\tau/M$",
        title=f"SdS scalar field, {label}",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "spacetime.png", dpi=220)
    plt.close(fig)

    black_hole = _observer_index(result, 0.0)
    middle = _observer_index(result, 0.5)
    cosmological = _observer_index(result, 1.0)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for index in (black_hole, middle, cosmological):
        axes[0].plot(
            result.signal_times,
            result.signals[:, index],
            linewidth=1,
            label=rf"$\rho={result.observer_rho[index]:g}$",
        )
        axes[1].semilogy(
            result.signal_times,
            np.maximum(np.abs(result.signals[:, index]), np.finfo(float).tiny),
            linewidth=1,
            label=rf"$\rho={result.observer_rho[index]:g}$",
        )
    axes[0].set(ylabel=r"$u$")
    axes[1].set(xlabel=r"$\tau/M$", ylabel=r"$|u|$")
    axes[0].set_title(f"Scalar waveforms, {label}")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "waveforms.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.semilogy(
        result.snapshot_times,
        np.maximum(result.constraint_linf, np.finfo(float).tiny),
        label=r"$\|\psi-\partial_\rho u\|_\infty$",
    )
    axis.semilogy(
        result.snapshot_times,
        np.maximum(result.constraint_l2, np.finfo(float).tiny),
        label="RMS",
    )
    axis.set(
        xlabel=r"$\tau/M$",
        ylabel="constraint error",
        title=f"First-order constraint, {label}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "constraint.png", dpi=220)
    plt.close(fig)


def create_sds_comparison_plots(
    results: dict[str, SdSSimulationResult], output_dir: Path
) -> None:
    """Create multi-bridge comparison figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    first = next(iter(results.values()))
    model = _model_from_result(first)
    horizons = sds_horizons(model)
    rho_dense = np.linspace(0.0, 1.0, 2000)
    radius_dense = areal_radius(rho_dense, model)
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for color_index, bridge in enumerate(results):
        color = colors(color_index % 10)
        label = BRIDGE_LABELS.get(bridge, bridge)
        boost = bridge_boost(rho_dense, model, bridge)
        ingoing, outgoing = characteristic_speeds(rho_dense, model, bridge)
        axes[0].plot(radius_dense, boost, color=color, linewidth=1.5, label=label)
        axes[0].plot(
            [turning_radius(model, bridge)],
            [0.0],
            marker="o",
            color=color,
            markersize=4,
        )
        axes[1].plot(
            radius_dense,
            outgoing,
            color=color,
            linewidth=1.4,
            label=label,
        )
        axes[1].plot(
            radius_dense,
            ingoing,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[0].axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    axes[0].axhline(-1.0, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set(
        xlabel=r"areal radius $r/M$",
        ylabel=r"bridge boost $B=f\,h'(r)$",
        title="Future-directed bridge boosts",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].set(
        xlabel=r"areal radius $r/M$",
        ylabel=r"light speed $d\rho/d\tau$",
        title="Characteristic speeds",
    )
    for axis in axes:
        axis.axvline(horizons.black_hole, color="black", alpha=0.2)
        axis.axvline(horizons.cosmological, color="black", alpha=0.2)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "bridge_boost_characteristics.png", dpi=240)
    plt.close(fig)

    max_time = min(
        150.0, max(result.snapshot_times[-1] for result in results.values())
    )
    color_limit = 0.0
    for result in results.values():
        mask = result.snapshot_times <= max_time
        if np.any(mask):
            color_limit = max(
                color_limit, float(np.max(np.abs(result.u_snapshots[mask])))
            )
    if color_limit == 0:
        color_limit = 1.0

    bridges = list(results)
    columns = 2
    rows = int(np.ceil(len(bridges) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.5, 3.5 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, bridge in zip(axes.ravel(), bridges):
        result = results[bridge]
        order = np.argsort(result.areal_radius)
        mask = result.snapshot_times <= max_time
        image = axis.pcolormesh(
            result.areal_radius[order],
            result.snapshot_times[mask],
            result.u_snapshots[mask][:, order],
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=-color_limit, vcenter=0.0, vmax=color_limit
            ),
            rasterized=True,
        )
        axis.set_title(BRIDGE_LABELS.get(bridge, bridge))
        axis.grid(alpha=0.12)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\tau/M$")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"areal radius $r/M$")
    for axis in axes.ravel()[len(bridges) :]:
        axis.axis("off")
    fig.subplots_adjust(right=0.88)
    cbar_axis = fig.add_axes([0.9, 0.16, 0.02, 0.7])
    fig.colorbar(image, cax=cbar_axis, label=r"$u(\tau,r)$")
    fig.suptitle("Scalar pulse propagation across SdS bridge foliations", y=0.995)
    fig.savefig(output_dir / "spacetime_bridge_gallery.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True)
    for color_index, (bridge, result) in enumerate(results.items()):
        color = colors(color_index % 10)
        label = BRIDGE_LABELS.get(bridge, bridge)
        index = _observer_index(result, 1.0)
        axes[0].plot(
            result.signal_times,
            result.signals[:, index],
            color=color,
            linewidth=1.0,
            label=label,
        )
        axes[1].semilogy(
            result.signal_times,
            np.maximum(np.abs(result.signals[:, index]), np.finfo(float).tiny),
            color=color,
            linewidth=1.0,
            label=label,
        )
    axes[0].set(ylabel=r"$u(\mathcal{H}_c^+)$")
    axes[1].set(xlabel=r"$\tau/M$", ylabel=r"$|u(\mathcal{H}_c^+)|$")
    axes[0].set_title("Signal at the cosmological horizon")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "cosmological_horizon_waveforms.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharex=True)
    for color_index, (bridge, result) in enumerate(results.items()):
        color = colors(color_index % 10)
        label = BRIDGE_LABELS.get(bridge, bridge)
        black_hole = _observer_index(result, 0.0)
        cosmological = _observer_index(result, 1.0)
        axes[0].plot(
            result.signal_times,
            result.signals[:, black_hole],
            color=color,
            linewidth=1.0,
            label=label,
        )
        axes[1].plot(
            result.signal_times,
            result.signals[:, cosmological],
            color=color,
            linewidth=1.0,
            label=label,
        )
    axes[0].set(ylabel=r"$u(\mathcal{H}^+)$", title="Black-hole horizon")
    axes[1].set(
        ylabel=r"$u(\mathcal{H}_c^+)$", title="Cosmological horizon"
    )
    for axis in axes:
        axis.set_xlabel(r"$\tau/M$")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_signal_comparison.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for color_index, (bridge, result) in enumerate(results.items()):
        axis.semilogy(
            result.snapshot_times,
            np.maximum(result.constraint_linf, np.finfo(float).tiny),
            color=colors(color_index % 10),
            linewidth=1.1,
            label=BRIDGE_LABELS.get(bridge, bridge),
        )
    axis.set(
        xlabel=r"$\tau/M$",
        ylabel=r"$\|\psi-\partial_\rho u\|_\infty$",
        title="Constraint preservation across bridge choices",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "constraint_bridge_comparison.png", dpi=240)
    plt.close(fig)

    rows = []
    for bridge, result in results.items():
        coefficient = propagation_coefficient(rho_dense, model, bridge)
        boost = bridge_boost(rho_dense, model, bridge)
        rows.append(
            {
                "bridge": bridge,
                "label": BRIDGE_LABELS.get(bridge, bridge),
                "turning_radius": turning_radius(model, bridge),
                "max_abs_boost_interior": float(np.max(np.abs(boost[1:-1]))),
                "min_A": float(np.min(coefficient)),
                "max_A": float(np.max(coefficient)),
                "max_constraint_linf": float(np.max(result.constraint_linf)),
                "wall_seconds": float(result.metadata["wall_seconds"]),
            }
        )
    with (output_dir / "bridge_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
