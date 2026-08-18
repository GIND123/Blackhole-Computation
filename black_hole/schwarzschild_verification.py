"""Standalone checks of the archived Schwarzschild localized-source run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .localized_source import angular_spectral_weights
from .source_evolution import SourcedSimulationResult, load_sourced_result


LEVELS = ("coarse", "medium", "fine")
ELL_MAX = {"coarse": 42, "medium": 46, "fine": 50}
DEFAULT_INPUT = Path("results/regulator_production_v3/raw/source")
DEFAULT_OUTPUT = Path("results/schwarzschild_verification")


def _angular_norm_weights(result: SourcedSimulationResult) -> np.ndarray:
    concentration = float(result.metadata["source"]["angular_concentration"])
    weights = angular_spectral_weights(concentration, int(result.response_ell[-1]))
    ell = result.response_ell.astype(float)
    return (2.0 * ell + 1.0) * weights[result.response_ell] ** 2 / (4.0 * np.pi)


def _common_interval(
    first: SourcedSimulationResult, second: SourcedSimulationResult
) -> tuple[float, float]:
    return (
        max(float(first.retarded_time[0]), float(second.retarded_time[0]), 0.0),
        min(float(first.retarded_time[-1]), float(second.retarded_time[-1])),
    )


def sphere_time_relative_l2(
    candidate: SourcedSimulationResult,
    reference: SourcedSimulationResult,
    *,
    observer: int | None = None,
) -> float:
    """Return the exact Parseval norm of one archived field difference."""

    candidate_observer = (
        candidate.outer_index() if observer is None else int(observer)
    )
    reference_observer = (
        reference.outer_index() if observer is None else int(observer)
    )
    start, end = _common_interval(candidate, reference)
    selected = (
        (reference.retarded_time >= start) & (reference.retarded_time <= end)
    )
    times = reference.retarded_time[selected]
    reference_response = reference.response_signals[selected, reference_observer]
    difference_energy = np.zeros(times.size)
    reference_energy = np.zeros(times.size)
    reference_weights = _angular_norm_weights(reference)
    candidate_lookup = {
        int(ell): index for index, ell in enumerate(candidate.response_ell)
    }
    for reference_index, ell in enumerate(reference.response_ell):
        ell = int(ell)
        reference_mode = reference_response[:, reference_index]
        reference_energy += reference_weights[reference_index] * reference_mode**2
        if ell in candidate_lookup:
            candidate_mode = np.interp(
                times,
                candidate.retarded_time,
                candidate.response_signals[
                    :, candidate_observer, candidate_lookup[ell]
                ],
            )
        else:
            candidate_mode = np.zeros_like(times)
        difference_energy += reference_weights[reference_index] * (
            candidate_mode - reference_mode
        ) ** 2
    numerator = float(np.trapezoid(difference_energy, times))
    denominator = float(np.trapezoid(reference_energy, times))
    return float(np.sqrt(numerator / denominator))


def angular_truncation_l2(
    result: SourcedSimulationResult,
    ell_max: int,
    *,
    observer: int | None = None,
) -> float:
    """Measure an isolated angular cutoff using one fine radial evolution."""

    observer = result.outer_index() if observer is None else int(observer)
    selected = result.retarded_time >= 0.0
    times = result.retarded_time[selected]
    response = result.response_signals[selected, observer]
    weights = _angular_norm_weights(result)
    total = np.sum(weights[None, :] * response**2, axis=1)
    omitted = result.response_ell > ell_max
    residual = np.sum(weights[None, omitted] * response[:, omitted] ** 2, axis=1)
    return float(
        np.sqrt(np.trapezoid(residual, times) / np.trapezoid(total, times))
    )


def create_report(input_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the standalone table and publication-size verification figure."""

    results = {
        level: load_sourced_result(Path(input_dir) / level / "schwarzschild.npz")
        for level in LEVELS
    }
    fine = results["fine"]
    rows: list[dict] = []
    for level in LEVELS:
        result = results[level]
        rows.append(
            {
                "check": "combined_refinement",
                "level": level,
                "radial_resolution": result.metadata["numerical"]["radial_resolution"],
                "timestep_over_M": result.metadata["numerical"]["timestep"],
                "ell_max": result.metadata["numerical"]["angular_ell_max"],
                "sphere_time_relative_l2_to_fine": (
                    0.0 if level == "fine" else sphere_time_relative_l2(result, fine)
                ),
                "maximum_constraint_linf": float(np.max(result.constraint_linf)),
                "time_translation_fitted": False,
            }
        )
    for cutoff in (42, 46):
        rows.append(
            {
                "check": "isolated_angular_truncation",
                "level": "fine responses",
                "radial_resolution": 2048,
                "timestep_over_M": 0.0005,
                "ell_max": cutoff,
                "sphere_time_relative_l2_to_fine": angular_truncation_l2(fine, cutoff),
                "maximum_constraint_linf": float(np.max(fine.constraint_linf)),
                "time_translation_fitted": False,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / "schwarzschild_verification.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    combined = rows[:2]
    axes[0].semilogy(
        [row["radial_resolution"] for row in combined],
        [row["sphere_time_relative_l2_to_fine"] for row in combined],
        "o-",
        color="#0072B2",
    )
    axes[0].set(
        xlabel=r"radial points $N_r$",
        ylabel=r"relative sphere-time $L^2$",
        title="combined ladder",
    )
    angular = rows[3:]
    axes[1].semilogy(
        [row["ell_max"] for row in angular],
        [row["sphere_time_relative_l2_to_fine"] for row in angular],
        "s-",
        color="#D55E00",
    )
    axes[1].set(
        xlabel=r"angular cutoff $\ell_{\max}$",
        ylabel=r"isolated omitted-field $L^2$",
        title="angular truncation",
    )
    for level, color in zip(LEVELS, ("#56B4E9", "#009E73", "#CC79A7")):
        result = results[level]
        axes[2].semilogy(
            result.diagnostic_times,
            np.maximum(result.constraint_linf, 1e-20),
            color=color,
            label=rf"$N_r={result.metadata['numerical']['radial_resolution']}$",
        )
    axes[2].set(
        xlabel=r"bridge time $\tau/M$",
        ylabel=r"$\|\psi-\partial_\rho u\|_\infty$",
        title="reduction constraint",
    )
    axes[2].legend(frameon=False)
    for index, axis in enumerate(axes):
        axis.grid(alpha=0.2, which="both")
        axis.text(0.03, 0.96, f"({chr(97 + index)})", transform=axis.transAxes, va="top")
    fig.tight_layout()
    png = output_dir / "schwarzschild_verification.png"
    pdf = output_dir / "schwarzschild_verification.pdf"
    fig.savefig(png, dpi=360)
    fig.savefig(pdf)
    plt.close(fig)
    return table, png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(create_report(arguments.input_dir, arguments.output_dir))


if __name__ == "__main__":
    main()
