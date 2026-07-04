"""Resolution and timestep convergence studies."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .model import InitialData, ModelParameters
from .solver import NumericalParameters, SimulationResult, run_simulation


def relative_l2(reference: SimulationResult, candidate: SimulationResult) -> float:
    """Relative L2 difference between infinity waveforms on reference times."""

    ref_index = int(np.argmin(np.abs(reference.observer_rho - 1.0)))
    candidate_index = int(np.argmin(np.abs(candidate.observer_rho - 1.0)))
    candidate_signal = np.interp(
        reference.signal_times,
        candidate.signal_times,
        candidate.signals[:, candidate_index],
    )
    difference = candidate_signal - reference.signals[:, ref_index]
    denominator = np.linalg.norm(reference.signals[:, ref_index])
    return float(np.linalg.norm(difference) / denominator)


def run_convergence_study(
    model: ModelParameters,
    initial: InitialData,
    base: NumericalParameters,
    output_dir: Path,
    resolutions: tuple[int, ...] = (48, 64, 96, 128),
    timesteps: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005),
    end_time: float = 200.0,
) -> list[dict]:
    """Run independent spatial- and temporal-convergence sequences."""

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    spatial_results = []
    for resolution in resolutions:
        settings = replace(
            base,
            resolution=resolution,
            timestep=base.timestep,
            end_time=end_time,
            signal_dt=max(base.signal_dt, base.timestep),
            snapshot_dt=max(base.snapshot_dt, base.timestep),
        )
        result = run_simulation(model, initial, settings)
        result.save(raw_dir / f"spatial_N{resolution}.npz")
        spatial_results.append(result)
    spatial_reference = spatial_results[-1]
    for resolution, result in zip(resolutions, spatial_results):
        rows.append(
            {
                "study": "spatial",
                "resolution": resolution,
                "timestep": base.timestep,
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
            resolution=max(resolutions),
            timestep=timestep,
            end_time=end_time,
            signal_dt=max(base.signal_dt, timestep),
            snapshot_dt=max(base.snapshot_dt, timestep),
        )
        result = run_simulation(model, initial, settings)
        result.save(raw_dir / f"temporal_dt{timestep:g}.npz")
        temporal_results.append(result)
    temporal_reference = temporal_results[-1]
    for timestep, result in zip(timesteps, temporal_results):
        rows.append(
            {
                "study": "temporal",
                "resolution": max(resolutions),
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    spatial_plot_rows = spatial_rows[:-1]
    temporal_plot_rows = temporal_rows[:-1]
    axes[0].semilogy(
        [row["resolution"] for row in spatial_plot_rows],
        [row["difference_to_next"] for row in spatial_plot_rows],
        "o-",
    )
    axes[0].set(xlabel="Chebyshev modes", ylabel="relative waveform L2 error", title="Spatial convergence")
    axes[1].loglog(
        [row["timestep"] for row in temporal_plot_rows],
        [row["difference_to_next"] for row in temporal_plot_rows],
        "o-",
    )
    axes[1].set(xlabel=r"$\Delta\tau$", ylabel="relative waveform L2 error", title="Timestep convergence")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=220)
    plt.close(fig)
    return rows
