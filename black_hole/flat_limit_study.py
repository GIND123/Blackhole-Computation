"""Controlled Schwarzschild-de Sitter to Schwarzschild flat-limit study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    minimal_boost as schwarzschild_boost,
    minimal_height as schwarzschild_height,
    propagation_coefficient as schwarzschild_propagation,
    retarded_time_offset as schwarzschild_retarded_time_offset,
)
from .sds_model import (
    ArealBumpInitialData,
    SdSParameters,
    areal_radius,
    bridge_boost,
    compact_areal_profile,
    minimal_height as sds_height,
    propagation_coefficient,
    propagation_endpoint_coefficients,
    retarded_time_offset as sds_retarded_time_offset,
    sds_horizons,
)
from .sds_solver import (
    SdSNumericalParameters,
    SdSSimulationResult,
    run_schwarzschild_scalar_simulation,
    run_sds_simulation,
)
from .sds_study import run_sds_convergence_study


def _horizon_signal(result: SdSSimulationResult) -> np.ndarray:
    index = int(np.argmin(np.abs(result.observer_rho - 1.0)))
    return result.signals[:, index]


def _common_retarded_times(
    reference: SdSSimulationResult,
    finite_results: dict[float, SdSSimulationResult],
    reference_offset: float,
    finite_offsets: dict[float, float],
) -> np.ndarray:
    """Return reference samples in the time interval shared by all runs."""

    starts = [reference.signal_times[0] - reference_offset]
    ends = [reference.signal_times[-1] - reference_offset]
    for length, result in finite_results.items():
        starts.append(result.signal_times[0] - finite_offsets[length])
        ends.append(result.signal_times[-1] - finite_offsets[length])
    lower, upper = max(starts), min(ends)
    reference_u = reference.signal_times - reference_offset
    common = reference_u[(reference_u >= lower) & (reference_u <= upper)]
    if common.size < 2:
        raise ValueError("The retarded-time domains do not have a usable overlap.")
    return common


def _waveform_difference(
    reference: SdSSimulationResult,
    candidate: SdSSimulationResult,
    retarded_times: np.ndarray,
    reference_offset: float,
    candidate_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return geometrically aligned signals and candidate-reference."""

    reference_signal = np.interp(
        retarded_times,
        reference.signal_times - reference_offset,
        _horizon_signal(reference),
    )
    candidate_signal = np.interp(
        retarded_times,
        candidate.signal_times - candidate_offset,
        _horizon_signal(candidate),
    )
    return (
        retarded_times,
        reference_signal,
        candidate_signal,
        candidate_signal - reference_signal,
    )


def _l2_norm(values: np.ndarray, times: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(np.asarray(values) ** 2, x=times)))


def _flat_limit_rows(
    reference: SdSSimulationResult,
    finite_results: dict[float, SdSSimulationResult],
    retarded_times: np.ndarray,
    reference_offset: float,
    finite_offsets: dict[float, float],
) -> list[dict]:
    rows: list[dict] = []
    aligned_reference = np.interp(
        retarded_times,
        reference.signal_times - reference_offset,
        _horizon_signal(reference),
    )
    reference_norm = _l2_norm(aligned_reference, retarded_times)
    for length, result in finite_results.items():
        times, _, _, difference = _waveform_difference(
            reference,
            result,
            retarded_times,
            reference_offset,
            finite_offsets[length],
        )
        difference_l2 = _l2_norm(difference, times)
        horizons = result.metadata["horizons"]
        left_a, right_a = propagation_endpoint_coefficients(
            SdSParameters(**result.metadata["model"]), "minimal"
        )
        rows.append(
            {
                "L": length,
                "Lambda": 3.0 / length**2,
                "r_black_hole": horizons["black_hole"],
                "r_cosmological": horizons["cosmological"],
                "A_black_hole": left_a,
                "A_cosmological": right_a,
                "retarded_time_offset_q": finite_offsets[length],
                "offset_difference_from_schwarzschild": (
                    finite_offsets[length] - reference_offset
                ),
                "difference_l2": difference_l2,
                "difference_relative_l2": difference_l2 / reference_norm,
                "difference_linf": float(np.max(np.abs(difference))),
                "max_constraint_linf": float(np.max(result.constraint_linf)),
            }
        )
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        current["empirical_power_in_L"] = float(
            -np.log(
                current["difference_relative_l2"]
                / previous["difference_relative_l2"]
            )
            / np.log(current["L"] / previous["L"])
        )
    rows[0]["empirical_power_in_L"] = ""
    return rows


def _write_flat_limit_tables(
    reference: SdSSimulationResult,
    finite_results: dict[float, SdSSimulationResult],
    rows: list[dict],
    output_dir: Path,
    reference_radius: float,
    retarded_times: np.ndarray,
    reference_offset: float,
    finite_offsets: dict[float, float],
    initial: ArealBumpInitialData,
) -> None:
    with (output_dir / "flat_limit_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    aligned_reference = np.interp(
        retarded_times,
        reference.signal_times - reference_offset,
        _horizon_signal(reference),
    )
    columns: list[np.ndarray] = [retarded_times, aligned_reference]
    headers = ["U", "schwarzschild_scri_plus"]
    for length, result in finite_results.items():
        _, _, candidate, difference = _waveform_difference(
            reference,
            result,
            retarded_times,
            reference_offset,
            finite_offsets[length],
        )
        label = f"L{length:g}"
        columns.extend((candidate, difference))
        headers.extend((f"sds_{label}_cosmological_horizon", f"difference_{label}"))
    with (output_dir / "waveform_differences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(np.column_stack(columns).tolist())

    with (output_dir / "retarded_time_offsets.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["background", "L", "q", "q_minus_q_schwarzschild"])
        writer.writerow(["Schwarzschild", "infinity", reference_offset, 0.0])
        for length in finite_results:
            writer.writerow(
                [
                    "Schwarzschild-de Sitter",
                    length,
                    finite_offsets[length],
                    finite_offsets[length] - reference_offset,
                ]
            )

    radius = np.linspace(
        initial.center_radius - 1.1 * initial.support_half_width,
        initial.center_radius + 1.1 * initial.support_half_width,
        1001,
    )
    profile, radial_derivative = compact_areal_profile(radius, initial)
    with (output_dir / "initial_profiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["r", "u", "du_dr"])
        writer.writerows(
            np.column_stack((radius, profile, radial_derivative)).tolist()
        )

    diagnostics = {
        "purpose": (
            "Compare finite-L cosmological-horizon signals with the Lambda=0 "
            "Schwarzschild signal at future null infinity."
        ),
        "coordinate": "rho=(1-r_b/r)/(1-r_b/r_c)",
        "schwarzschild_coordinate": "rho=1-2M/r",
        "bridge": "minimal",
        "height_normalization": {
            "reference_radius": reference_radius,
            "condition": "h_L(r_reference)=h_Schwarzschild(r_reference)=0",
        },
        "retarded_time": {
            "definition": "U=tau-q, q=lim(h+r_*)",
            "endpoint_evaluation": "closed-form analytic limit",
            "common_U_interval": [
                float(retarded_times[0]),
                float(retarded_times[-1]),
            ],
            "schwarzschild_q": reference_offset,
            "finite_L_q": {str(key): value for key, value in finite_offsets.items()},
        },
        "initial_data": reference.metadata["initial_data"],
        "numerical": reference.metadata["numerical"],
        "schwarzschild_max_constraint_linf": float(
            np.max(reference.constraint_linf)
        ),
        "finite_L": rows,
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2)


def _create_flat_limit_plots(
    reference: SdSSimulationResult,
    finite_results: dict[float, SdSSimulationResult],
    rows: list[dict],
    output_dir: Path,
    reference_radius: float,
    retarded_times: np.ndarray,
    reference_offset: float,
    finite_offsets: dict[float, float],
    initial: ArealBumpInitialData,
) -> None:
    colors = plt.get_cmap("viridis")(
        np.linspace(0.15, 0.9, len(finite_results))
    )
    reference_signal = np.interp(
        retarded_times,
        reference.signal_times - reference_offset,
        _horizon_signal(reference),
    )

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.plot(
        retarded_times,
        reference_signal,
        color="black",
        linewidth=1.6,
        label=r"Schwarzschild $\mathscr{I}^+$",
    )
    for color, (length, result) in zip(colors, finite_results.items()):
        axis.plot(
            retarded_times,
            np.interp(
                retarded_times,
                result.signal_times - finite_offsets[length],
                _horizon_signal(result),
            ),
            color=color,
            linewidth=1.0,
            label=rf"SdS $L={length:g}$, $\mathcal{{H}}_c^+$",
        )
    axis.set(
        xlabel=r"geometric retarded time $U/M=(\tau-q_L)/M$",
        ylabel=r"reduced scalar field $u$",
        title="Cosmological-horizon waveforms and the Schwarzschild flat limit",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_comparison.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    difference_floor = 1e-14 * max(
        float(row["difference_linf"]) for row in rows
    )
    for color, (length, result) in zip(colors, finite_results.items()):
        times, _, _, difference = _waveform_difference(
            reference,
            result,
            retarded_times,
            reference_offset,
            finite_offsets[length],
        )
        axes[0].plot(
            times,
            difference,
            color=color,
            linewidth=1.0,
            label=rf"$L={length:g}$",
        )
        axes[1].semilogy(
            times,
            np.maximum(np.abs(difference), difference_floor),
            color=color,
            linewidth=1.0,
        )
    axes[0].set(ylabel=r"$u_L(\mathcal{H}_c^+)-u_0(\mathscr{I}^+)$")
    axes[1].set(
        xlabel=r"geometric retarded time $U/M$",
        ylabel="absolute difference",
    )
    axes[0].set_title("Finite-L waveform difference versus time")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_differences.png", dpi=240)
    plt.close(fig)

    lengths = np.asarray([row["L"] for row in rows], dtype=float)
    relative = np.asarray(
        [row["difference_relative_l2"] for row in rows], dtype=float
    )
    linf = np.asarray([row["difference_linf"] for row in rows], dtype=float)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.loglog(lengths, relative, "o-", label="relative time-domain L2")
    axis.loglog(lengths, linf, "s--", label="absolute Linf")
    axis.set(
        xlabel=r"cosmological length $L/M$",
        ylabel="waveform difference norm",
        title="Approach to the Schwarzschild waveform",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "flat_limit_norms.png", dpi=240)
    plt.close(fig)

    lengths = np.asarray(list(finite_results), dtype=float)
    offsets = np.asarray([finite_offsets[length] for length in lengths])
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True)
    axes[0].semilogx(lengths, offsets, "o-", label=r"SdS $q_L$")
    axes[0].axhline(
        reference_offset,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Schwarzschild $q_0$",
    )
    axes[1].loglog(
        lengths,
        np.abs(offsets - reference_offset),
        "o-",
        color="tab:red",
    )
    axes[0].set(ylabel=r"retarded-time offset $q/M$")
    axes[1].set(
        xlabel=r"cosmological length $L/M$",
        ylabel=r"$|q_L-q_0|/M$",
    )
    axes[0].set_title(r"Analytic geometric offsets, $h(4M)=r_*(4M)=0$")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "retarded_time_offsets.png", dpi=240)
    plt.close(fig)

    radius = np.linspace(
        initial.center_radius - 1.1 * initial.support_half_width,
        initial.center_radius + 1.1 * initial.support_half_width,
        1501,
    )
    profile, radial_derivative = compact_areal_profile(radius, initial)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(radius, profile, color="black", linewidth=1.8)
    axes[1].plot(radius, radial_derivative, color="black", linewidth=1.8)
    for color, (length, result) in zip(colors, finite_results.items()):
        initial_values = result.u_snapshots[0]
        order = np.argsort(result.areal_radius)
        support = np.isfinite(result.areal_radius[order]) & (
            result.areal_radius[order] <= radius[-1]
        )
        axes[0].plot(
            result.areal_radius[order][support],
            initial_values[order][support],
            ".",
            color=color,
            markersize=1.7,
            alpha=0.55,
            label=rf"SdS $L={length:g}$",
        )
    axes[0].set(ylabel=r"$u(0,r)=r\Phi(0,r)$")
    axes[1].set(xlabel=r"areal radius $r/M$", ylabel=r"$\partial_r u(0,r)$")
    axes[0].set_title(r"Identical $C^\infty$ compact pulse in areal radius")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "initial_profiles_areal_radius.png", dpi=240)
    plt.close(fig)

    rho = np.linspace(0.0, 1.0, 2001)
    schwarzschild_model = SchwarzschildScalarParameters(
        **reference.metadata["model"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    axes[0].plot(
        rho,
        schwarzschild_boost(rho),
        color="black",
        linewidth=1.7,
        label="Schwarzschild",
    )
    axes[1].plot(
        rho,
        schwarzschild_propagation(rho, schwarzschild_model),
        color="black",
        linewidth=1.7,
        label="Schwarzschild",
    )
    for color, (length, result) in zip(colors, finite_results.items()):
        model = SdSParameters(**result.metadata["model"])
        axes[0].plot(
            rho,
            bridge_boost(rho, model, "minimal"),
            color=color,
            linewidth=1.0,
            label=rf"$L={length:g}$",
        )
        axes[1].plot(
            rho,
            propagation_coefficient(rho, model, "minimal"),
            color=color,
            linewidth=1.0,
            label=rf"$L={length:g}$",
        )
    axes[0].set(xlabel=r"$\rho$", ylabel=r"$B=f h'(r)$", title="Minimal boost")
    axes[1].set(xlabel=r"$\rho$", ylabel=r"$A$", title="Propagation coefficient")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "coordinate_flat_limit.png", dpi=240)
    plt.close(fig)

    # Plot the normalized heights only on the common near-zone interval, away
    # from their expected logarithmic horizon divergences.
    left = max(
        2.05 * schwarzschild_model.mass,
        max(
            sds_horizons(
                SdSParameters(**result.metadata["model"])
            ).black_hole
            for result in finite_results.values()
        )
        + 0.02,
    )
    right = min(
        10.0 * schwarzschild_model.mass,
        min(
            sds_horizons(
                SdSParameters(**result.metadata["model"])
            ).cosmological
            for result in finite_results.values()
        )
        - 0.1,
    )
    radius = np.linspace(left, right, 1000)
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        radius,
        schwarzschild_height(radius, schwarzschild_model, reference_radius),
        color="black",
        linewidth=1.7,
        label="Schwarzschild",
    )
    for color, (length, result) in zip(colors, finite_results.items()):
        model = SdSParameters(**result.metadata["model"])
        axis.plot(
            radius,
            sds_height(radius, model, reference_radius),
            color=color,
            linewidth=1.0,
            label=rf"$L={length:g}$",
        )
    axis.axvline(reference_radius, color="gray", linewidth=0.9, linestyle=":")
    axis.set(
        xlabel=r"areal radius $r/M$",
        ylabel=r"normalized height $h(r)/M$",
        title=rf"Common time normalization at $r_0={reference_radius:g}M$",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "height_alignment.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.semilogy(
        reference.snapshot_times,
        np.maximum(reference.constraint_linf, np.finfo(float).tiny),
        color="black",
        linewidth=1.5,
        label="Schwarzschild",
    )
    for color, (length, result) in zip(colors, finite_results.items()):
        axis.semilogy(
            result.snapshot_times,
            np.maximum(result.constraint_linf, np.finfo(float).tiny),
            color=color,
            linewidth=1.0,
            label=rf"$L={length:g}$",
        )
    axis.set(
        xlabel=r"$\tau/M$",
        ylabel=r"$\|\psi-\partial_\rho u\|_\infty$",
        title="Constraint preservation in the flat-limit sequence",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "constraints.png", dpi=240)
    plt.close(fig)


def _create_convergence_summary(
    studies: dict[float, list[dict]], output_dir: Path
) -> None:
    if not studies:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for length, rows in studies.items():
        spatial = [
            row
            for row in rows
            if row["study"] == "spatial" and row["difference_to_next"] != ""
        ]
        temporal = [
            row
            for row in rows
            if row["study"] == "temporal" and row["difference_to_next"] != ""
        ]
        axes[0].semilogy(
            [row["resolution"] for row in spatial],
            [row["difference_to_next"] for row in spatial],
            "o-",
            label=rf"$L={length:g}$",
        )
        axes[1].loglog(
            [row["timestep"] for row in temporal],
            [row["difference_to_next"] for row in temporal],
            "o-",
            label=rf"$L={length:g}$",
        )
    axes[0].set(
        xlabel="Chebyshev modes",
        ylabel="relative horizon-waveform difference",
        title="Spatial convergence",
    )
    axes[1].set(
        xlabel=r"$\Delta\tau/M$",
        ylabel="relative horizon-waveform difference",
        title="RK222 timestep convergence",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_summary.png", dpi=240)
    plt.close(fig)


def run_flat_limit_study(
    *,
    mass: float,
    ell: int,
    lengths: tuple[float, ...],
    initial: ArealBumpInitialData,
    numerical: SdSNumericalParameters,
    output_dir: Path,
    reference_radius: float,
    convergence_lengths: tuple[float, ...] = (20.0, 160.0),
    convergence_end_time: float = 100.0,
    run_convergence: bool = True,
) -> dict:
    """Run, compare, plot, and validate the complete 1D flat-limit sequence."""

    if not lengths:
        raise ValueError("At least one cosmological length is required.")
    if tuple(sorted(lengths)) != lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Cosmological lengths must be unique and increasing.")
    if numerical.bridge != "minimal":
        raise ValueError("The controlled flat-limit study uses only minimal gauge.")
    if not isinstance(initial, ArealBumpInitialData):
        raise TypeError(
            "The corrected flat-limit study requires compact areal-radius initial data."
        )

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    schwarzschild_model = SchwarzschildScalarParameters(mass=mass, ell=ell)
    schwarzschild_height(
        np.asarray([reference_radius]), schwarzschild_model, reference_radius
    )
    reference = run_schwarzschild_scalar_simulation(
        schwarzschild_model, initial, numerical
    )
    reference.metadata["height_normalization"] = {
        "reference_radius": reference_radius,
        "height_at_reference": 0.0,
    }
    reference_offset = schwarzschild_retarded_time_offset(
        schwarzschild_model, reference_radius
    )
    reference.metadata["retarded_time_offset"] = {
        "q": reference_offset,
        "definition": "lim_(r->infinity) (h+r_*)",
        "evaluation": "analytic",
    }
    reference.save(raw_dir / "schwarzschild_reference.npz")

    finite_results: dict[float, SdSSimulationResult] = {}
    finite_offsets: dict[float, float] = {}
    for length in lengths:
        model = SdSParameters(
            mass=mass, cosmological_length=length, ell=ell
        )
        sds_height(np.asarray([reference_radius]), model, reference_radius)
        result = run_sds_simulation(model, initial, numerical)
        result.metadata["height_normalization"] = {
            "reference_radius": reference_radius,
            "height_at_reference": 0.0,
        }
        finite_offsets[length] = sds_retarded_time_offset(model, reference_radius)
        result.metadata["retarded_time_offset"] = {
            "q": finite_offsets[length],
            "definition": "lim_(r->r_c) (h+r_*)",
            "evaluation": "analytic",
        }
        result.save(raw_dir / f"sds_L{length:g}.npz")
        finite_results[length] = result

    retarded_times = _common_retarded_times(
        reference,
        finite_results,
        reference_offset,
        finite_offsets,
    )
    rows = _flat_limit_rows(
        reference,
        finite_results,
        retarded_times,
        reference_offset,
        finite_offsets,
    )
    _write_flat_limit_tables(
        reference,
        finite_results,
        rows,
        output_dir,
        reference_radius,
        retarded_times,
        reference_offset,
        finite_offsets,
        initial,
    )
    _create_flat_limit_plots(
        reference,
        finite_results,
        rows,
        output_dir,
        reference_radius,
        retarded_times,
        reference_offset,
        finite_offsets,
        initial,
    )

    convergence: dict[float, list[dict]] = {}
    if run_convergence:
        for length in convergence_lengths:
            if length not in lengths:
                raise ValueError(
                    f"Convergence length L={length:g} is not in the production sequence."
                )
            model = SdSParameters(
                mass=mass, cosmological_length=length, ell=ell
            )
            convergence[length] = run_sds_convergence_study(
                model,
                initial,
                numerical,
                output_dir / "convergence" / f"L{length:g}",
                bridge="minimal",
                resolutions=(192, 256, 384, 512),
                temporal_resolution=max(512, numerical.resolution),
                timesteps=(0.04, 0.02, 0.01, 0.005),
                end_time=convergence_end_time,
                spatial_timestep=0.0025,
            )
        _create_convergence_summary(convergence, output_dir / "convergence")

    return {
        "summary": rows,
        "schwarzschild_max_constraint_linf": float(
            np.max(reference.constraint_linf)
        ),
        "retarded_time_offsets": {
            "Schwarzschild": reference_offset,
            **{f"L={length:g}": value for length, value in finite_offsets.items()},
        },
        "convergence_lengths": list(convergence),
    }
