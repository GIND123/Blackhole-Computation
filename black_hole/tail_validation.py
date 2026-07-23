"""Post-process saved tail runs for resolution and profile sensitivity."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from .sds_model import (
    SdSParameters,
    retarded_time_offset as sds_retarded_time_offset,
    sds_horizons,
)
from .sds_solver import SdSSimulationResult, load_sds_result
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    retarded_time_offset as schwarzschild_retarded_time_offset,
)
from .tail_analysis import (
    aligned_signals,
    asymptotic_constant,
    json_safe,
    select_stable_exponential_fit,
    select_stable_power_fit,
)


def _outer_signal(result: SdSSimulationResult) -> np.ndarray:
    index = int(np.argmin(np.abs(result.observer_rho - 1.0)))
    return np.asarray(result.signals[:, index], dtype=float)


def _relative_l2(
    reference: SdSSimulationResult,
    candidate: SdSSimulationResult,
    reference_offset: float,
    candidate_offset: float,
    *,
    start: float | None = None,
) -> float:
    times, reference_signal, candidate_signal = aligned_signals(
        reference.signal_times - reference_offset,
        _outer_signal(reference),
        candidate.signal_times - candidate_offset,
        _outer_signal(candidate),
    )
    mask = times >= (times[0] if start is None else start)
    denominator = np.linalg.norm(reference_signal[mask])
    if denominator == 0.0:
        return float("nan")
    return float(
        np.linalg.norm(candidate_signal[mask] - reference_signal[mask]) / denominator
    )


def create_resolution_report(
    *,
    ell: int,
    length: float,
    reference_paths: tuple[Path, ...],
    sds_paths: tuple[Path, ...],
    output_dir: Path,
    reference_radius: float = 4.0,
) -> list[dict]:
    """Compare matched Schwarzschild/SdS runs across spatial resolutions."""

    if len(reference_paths) != len(sds_paths) or len(reference_paths) < 2:
        raise ValueError("At least two matched reference/SdS resolution pairs are required.")
    references = [load_sds_result(path) for path in reference_paths]
    finite = [load_sds_result(path) for path in sds_paths]
    resolutions = [int(result.metadata["numerical"]["resolution"]) for result in finite]
    if resolutions != sorted(resolutions) or len(set(resolutions)) != len(resolutions):
        raise ValueError("Input cases must have unique increasing resolutions.")

    mass = float(finite[-1].metadata["model"]["mass"])
    schwarzschild_model = SchwarzschildScalarParameters(mass=mass, ell=ell)
    sds_model = SdSParameters(mass=mass, cosmological_length=length, ell=ell)
    q_zero = schwarzschild_retarded_time_offset(schwarzschild_model, reference_radius)
    q_sds = sds_retarded_time_offset(sds_model, reference_radius)
    kappa = sds_horizons(sds_model).kappa_cosmological
    finest_reference = references[-1]
    finest_sds = finite[-1]
    rows: list[dict] = []

    for reference, candidate in zip(references, finite):
        reference_times = reference.signal_times - q_zero
        reference_signal = _outer_signal(reference)
        try:
            price_fit = select_stable_power_fit(
                reference_times,
                reference_signal,
                minimum_time=60.0,
                maximum_time=0.9 * float(reference_times[-1]),
            )
            price_rate = price_fit.rate
            price_r_squared = price_fit.r_squared
        except ValueError:
            price_rate = float("nan")
            price_r_squared = float("nan")
        if ell == 0:
            tail_metric = asymptotic_constant(
                candidate.signal_times - q_sds, _outer_signal(candidate)
            )["value"]
            tail_kind = "constant"
        else:
            try:
                tail_metric = select_stable_exponential_fit(
                    candidate.signal_times - q_sds,
                    _outer_signal(candidate),
                    kappa,
                ).rate / kappa
            except ValueError:
                tail_metric = float("nan")
            tail_kind = "gamma_over_kappa"
        rows.append(
            {
                "ell": ell,
                "L": length,
                "resolution": int(candidate.metadata["numerical"]["resolution"]),
                "timestep": float(candidate.metadata["numerical"]["timestep"]),
                "schwarzschild_scri_power": price_rate,
                "schwarzschild_scri_r_squared": price_r_squared,
                "sds_tail_metric": tail_metric,
                "sds_tail_metric_kind": tail_kind,
                "reference_relative_l2_to_finest": _relative_l2(
                    finest_reference, reference, q_zero, q_zero
                ),
                "sds_relative_l2_to_finest": _relative_l2(
                    finest_sds, candidate, q_sds, q_sds
                ),
                "sds_tail_relative_l2_to_finest": _relative_l2(
                    finest_sds,
                    candidate,
                    q_sds,
                    q_sds,
                    start=1.5 / kappa,
                ),
                "reference_max_constraint": float(np.max(reference.constraint_linf)),
                "sds_max_constraint": float(np.max(candidate.constraint_linf)),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolution_convergence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "resolution_convergence.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_safe(rows), stream, indent=2, allow_nan=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    axes[0].semilogx(
        resolutions,
        [row["schwarzschild_scri_power"] for row in rows],
        "o-",
    )
    axes[0].axhline(ell + 2, color="black", linestyle="--", label="Price target")
    axes[1].semilogx(
        resolutions,
        [row["sds_tail_metric"] for row in rows],
        "o-",
    )
    if ell > 0:
        axes[1].axhline(ell, color="black", linestyle="--", label="de Sitter target")
    axes[2].loglog(
        resolutions[:-1],
        [row["sds_tail_relative_l2_to_finest"] for row in rows[:-1]],
        "o-",
    )
    axes[0].set(xlabel="Chebyshev modes", ylabel="Price exponent", title="Schwarzschild tail")
    axes[1].set(xlabel="Chebyshev modes", ylabel=tail_kind, title="SdS late-time metric")
    axes[2].set(
        xlabel="Chebyshev modes",
        ylabel="relative tail-waveform L2 difference",
        title="Convergence to finest run",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        axis.set_xscale("log", base=2)
        axis.set_xlim(resolutions[0] / 1.15, resolutions[-1] * 1.15)
        axis.set_xticks(resolutions)
        axis.xaxis.set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    fig.savefig(output_dir / "resolution_convergence.png", dpi=240)
    plt.close(fig)
    return rows


def create_profile_sensitivity_report(
    *,
    ell: int,
    length: float,
    cases: tuple[tuple[float, Path, Path], ...],
    output_dir: Path,
    reference_radius: float = 4.0,
) -> list[dict]:
    """Compare tail exponents/rates for distinct physical pulse widths."""

    if len(cases) < 2:
        raise ValueError("At least two pulse-width cases are required.")
    rows: list[dict] = []
    for half_width, reference_path, sds_path in cases:
        reference = load_sds_result(reference_path)
        finite = load_sds_result(sds_path)
        mass = float(finite.metadata["model"]["mass"])
        schwarzschild_model = SchwarzschildScalarParameters(mass=mass, ell=ell)
        sds_model = SdSParameters(mass=mass, cosmological_length=length, ell=ell)
        q_zero = schwarzschild_retarded_time_offset(
            schwarzschild_model, reference_radius
        )
        q_sds = sds_retarded_time_offset(sds_model, reference_radius)
        kappa = sds_horizons(sds_model).kappa_cosmological
        price_fit = select_stable_power_fit(
            reference.signal_times - q_zero,
            _outer_signal(reference),
            minimum_time=60.0,
            maximum_time=0.9 * float(reference.signal_times[-1] - q_zero),
        )
        exponential = select_stable_exponential_fit(
            finite.signal_times - q_sds,
            _outer_signal(finite),
            kappa,
        )
        rows.append(
            {
                "ell": ell,
                "L": length,
                "support_half_width": half_width,
                "resolution": int(finite.metadata["numerical"]["resolution"]),
                "timestep": float(finite.metadata["numerical"]["timestep"]),
                "schwarzschild_scri_power": price_fit.rate,
                "schwarzschild_scri_r_squared": price_fit.r_squared,
                "gamma_over_kappa": exponential.rate / kappa,
                "exponential_r_squared": exponential.r_squared,
                "exponential_start": exponential.start,
                "exponential_end": exponential.end,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "profile_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "profile_sensitivity.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_safe(rows), stream, indent=2, allow_nan=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    widths = [row["support_half_width"] for row in rows]
    axes[0].plot(widths, [row["schwarzschild_scri_power"] for row in rows], "o-")
    axes[0].axhline(ell + 2, color="black", linestyle="--")
    axes[1].plot(widths, [row["gamma_over_kappa"] for row in rows], "o-")
    axes[1].axhline(ell, color="black", linestyle="--")
    axes[0].set(xlabel="support half-width / M", ylabel="Price exponent")
    axes[1].set(xlabel="support half-width / M", ylabel=r"$\gamma/\kappa_c$")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(rf"Initial-profile sensitivity, $\ell={ell}$, $L/M={length:g}$")
    fig.tight_layout()
    fig.savefig(output_dir / "profile_sensitivity.png", dpi=240)
    plt.close(fig)
    return rows


def create_timestep_report(
    *,
    ell: int,
    length: float,
    reference_paths: tuple[Path, ...],
    sds_paths: tuple[Path, ...],
    output_dir: Path,
    reference_radius: float = 4.0,
) -> list[dict]:
    """Compare saved runs at fixed resolution with decreasing timesteps."""

    if len(reference_paths) != len(sds_paths) or len(reference_paths) < 2:
        raise ValueError("At least two matched timestep pairs are required.")
    references = [load_sds_result(path) for path in reference_paths]
    finite = [load_sds_result(path) for path in sds_paths]
    resolutions = {
        int(result.metadata["numerical"]["resolution"])
        for result in references + finite
    }
    if len(resolutions) != 1:
        raise ValueError("All timestep-report runs must use the same resolution.")
    timesteps = [float(result.metadata["numerical"]["timestep"]) for result in finite]
    if timesteps != sorted(timesteps, reverse=True):
        raise ValueError("Timestep inputs must be ordered from coarse to fine.")

    mass = float(finite[-1].metadata["model"]["mass"])
    schwarzschild_model = SchwarzschildScalarParameters(mass=mass, ell=ell)
    sds_model = SdSParameters(mass=mass, cosmological_length=length, ell=ell)
    q_zero = schwarzschild_retarded_time_offset(schwarzschild_model, reference_radius)
    q_sds = sds_retarded_time_offset(sds_model, reference_radius)
    kappa = sds_horizons(sds_model).kappa_cosmological
    rows: list[dict] = []
    for reference, candidate in zip(references, finite):
        price = select_stable_power_fit(
            reference.signal_times - q_zero,
            _outer_signal(reference),
            minimum_time=60.0,
            maximum_time=0.9 * float(reference.signal_times[-1] - q_zero),
        )
        exponential = select_stable_exponential_fit(
            candidate.signal_times - q_sds,
            _outer_signal(candidate),
            kappa,
        )
        rows.append(
            {
                "ell": ell,
                "L": length,
                "resolution": next(iter(resolutions)),
                "timestep": float(candidate.metadata["numerical"]["timestep"]),
                "schwarzschild_scri_power": price.rate,
                "gamma_over_kappa": exponential.rate / kappa,
                "reference_relative_l2_to_finest": _relative_l2(
                    references[-1], reference, q_zero, q_zero
                ),
                "sds_relative_l2_to_finest": _relative_l2(
                    finite[-1], candidate, q_sds, q_sds
                ),
                "sds_tail_relative_l2_to_finest": _relative_l2(
                    finite[-1],
                    candidate,
                    q_sds,
                    q_sds,
                    start=1.5 / kappa,
                ),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "timestep_convergence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "timestep_convergence.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_safe(rows), stream, indent=2, allow_nan=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    axes[0].plot(timesteps, [row["schwarzschild_scri_power"] for row in rows], "o-")
    axes[0].axhline(ell + 2, color="black", linestyle="--")
    axes[1].plot(timesteps, [row["gamma_over_kappa"] for row in rows], "o-")
    axes[1].axhline(ell, color="black", linestyle="--")
    axes[0].set(xlabel=r"$\Delta\tau/M$", ylabel="Price exponent")
    axes[1].set(xlabel=r"$\Delta\tau/M$", ylabel=r"$\gamma/\kappa_c$")
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "timestep_convergence.png", dpi=240)
    plt.close(fig)
    return rows
