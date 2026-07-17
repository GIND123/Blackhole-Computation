"""Bridge-coordinate studies for Schwarzschild-de Sitter scalar waves."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .sds_analysis import (
    create_sds_comparison_plots,
    create_sds_plots,
    write_sds_diagnostics,
)
from .sds_model import (
    ArealBumpInitialData,
    BRIDGE_CHOICES,
    ScalarInitialData,
    SdSParameters,
)
from .sds_solver import (
    SdSNumericalParameters,
    SdSSimulationResult,
    run_sds_simulation,
)


def relative_l2(
    reference: SdSSimulationResult,
    candidate: SdSSimulationResult,
    observer: float = 1.0,
) -> float:
    """Relative L2 difference between observer waveforms on reference times."""

    reference_index = int(np.argmin(np.abs(reference.observer_rho - observer)))
    candidate_index = int(np.argmin(np.abs(candidate.observer_rho - observer)))
    candidate_signal = np.interp(
        reference.signal_times,
        candidate.signal_times,
        candidate.signals[:, candidate_index],
    )
    difference = candidate_signal - reference.signals[:, reference_index]
    denominator = np.linalg.norm(reference.signals[:, reference_index])
    return float(np.linalg.norm(difference) / denominator)


def run_sds_bridge_suite(
    model: SdSParameters,
    initial: ScalarInitialData,
    base: SdSNumericalParameters,
    output_dir: Path,
    bridges: tuple[str, ...] = BRIDGE_CHOICES,
) -> dict[str, SdSSimulationResult]:
    """Run a scalar evolution for each selected bridge coordinate."""

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, SdSSimulationResult] = {}

    for bridge in bridges:
        settings = replace(base, bridge=bridge)
        result = run_sds_simulation(model, initial, settings)
        result.save(raw_dir / f"{bridge}.npz")
        run_dir = output_dir / bridge
        write_sds_diagnostics(result, run_dir)
        create_sds_plots(result, run_dir)
        results[bridge] = result

    create_sds_comparison_plots(results, output_dir)
    return results


def run_sds_convergence_study(
    model: SdSParameters,
    initial: ScalarInitialData | ArealBumpInitialData,
    base: SdSNumericalParameters,
    output_dir: Path,
    bridge: str = "minimal",
    resolutions: tuple[int, ...] = (32, 48, 64, 96),
    temporal_resolution: int = 256,
    timesteps: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025),
    end_time: float = 120.0,
    spatial_timestep: float | None = None,
) -> list[dict]:
    """Run spatial and temporal convergence checks for one bridge."""

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("spatial_*.npz", "temporal_*.npz"):
        for stale_path in raw_dir.glob(pattern):
            stale_path.unlink()
    rows: list[dict] = []

    if spatial_timestep is None:
        spatial_timestep = min(base.timestep, min(timesteps))
    if spatial_timestep <= 0.0:
        raise ValueError("The spatial-study timestep must be positive.")
    spatial_results = []
    for resolution in resolutions:
        settings = replace(
            base,
            bridge=bridge,
            resolution=resolution,
            timestep=spatial_timestep,
            end_time=end_time,
            signal_dt=max(base.signal_dt, spatial_timestep),
            snapshot_dt=max(base.snapshot_dt, spatial_timestep),
        )
        result = run_sds_simulation(model, initial, settings)
        result.save(raw_dir / f"spatial_N{resolution}.npz")
        spatial_results.append(result)
    spatial_reference = spatial_results[-1]
    for resolution, result in zip(resolutions, spatial_results):
        rows.append(
            {
                "study": "spatial",
                "bridge": bridge,
                "resolution": resolution,
                "timestep": spatial_timestep,
                "end_time": end_time,
                "relative_l2": relative_l2(spatial_reference, result),
                "difference_to_next": "",
                "observed_order": "",
            }
        )

    temporal_results = []
    for timestep in timesteps:
        settings = replace(
            base,
            bridge=bridge,
            resolution=temporal_resolution,
            timestep=timestep,
            end_time=end_time,
            signal_dt=max(base.signal_dt, timestep),
            snapshot_dt=max(base.snapshot_dt, timestep),
        )
        result = run_sds_simulation(model, initial, settings)
        result.save(raw_dir / f"temporal_dt{timestep:g}.npz")
        temporal_results.append(result)
    temporal_reference = temporal_results[-1]
    for timestep, result in zip(timesteps, temporal_results):
        rows.append(
            {
                "study": "temporal",
                "bridge": bridge,
                "resolution": temporal_resolution,
                "timestep": timestep,
                "end_time": end_time,
                "relative_l2": relative_l2(temporal_reference, result),
                "difference_to_next": "",
                "observed_order": "",
            }
        )

    spatial_rows = [row for row in rows if row["study"] == "spatial"]
    for index in range(len(spatial_rows) - 1):
        spatial_rows[index]["difference_to_next"] = relative_l2(
            spatial_results[index + 1], spatial_results[index]
        )

    temporal_rows = [row for row in rows if row["study"] == "temporal"]
    temporal_differences = []
    for index in range(len(temporal_rows) - 1):
        difference = relative_l2(
            temporal_results[index + 1], temporal_results[index]
        )
        temporal_rows[index]["difference_to_next"] = difference
        temporal_differences.append(difference)
    for index in range(len(temporal_differences) - 1):
        ratio = timesteps[index] / timesteps[index + 1]
        temporal_rows[index]["observed_order"] = float(
            np.log(temporal_differences[index] / temporal_differences[index + 1])
            / np.log(ratio)
        )

    with (output_dir / "convergence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    spatial_plot_rows = spatial_rows[:-1]
    temporal_plot_rows = temporal_rows[:-1]
    axes[0].semilogy(
        [row["resolution"] for row in spatial_plot_rows],
        [row["difference_to_next"] for row in spatial_plot_rows],
        "o-",
    )
    axes[0].set(
        xlabel="Chebyshev modes",
        ylabel="relative waveform L2 error",
        title=f"Spatial convergence ({bridge})",
    )
    axes[1].loglog(
        [row["timestep"] for row in temporal_plot_rows],
        [row["difference_to_next"] for row in temporal_plot_rows],
        "o-",
    )
    axes[1].set(
        xlabel=r"$\Delta\tau$",
        ylabel="relative waveform L2 error",
        title=f"Timestep convergence ({bridge})",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=220)
    plt.close(fig)
    return rows
