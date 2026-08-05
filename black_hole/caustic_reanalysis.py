"""Reanalysis products requested for the localized source archives."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .caustic_analysis import (
    analyze_existing_archives,
    estimate_pulse,
    local_phase_comparison,
)
from .caustic_study import COSMOLOGICAL_LENGTHS, direction_waveform, load_case
from .localized_source import (
    LocalizedSourceParameters,
    minimum_ell_max,
    retained_angular_fraction,
    weak_source_integral,
)
from .null_geodesics import generic_target_angle, trace_null_ray
from .reproducibility import reproducibility_metadata
from .tail_analysis import json_safe


GENERIC_WINDOWS = {
    np.pi / 3.0: ((20.0, 42.0), (44.0, 62.0), (61.0, 77.0), (77.0, 98.0)),
    np.pi / 2.0: ((22.0, 43.0), (44.0, 61.0), (60.0, 77.0), (76.0, 96.0)),
}


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def delay_shift_rows(pulse_rows: list[dict]) -> list[dict]:
    lookup = {
        (row["case"], row["observer"], row["pulse"]): row for row in pulse_rows
    }
    rows: list[dict] = []
    for length in COSMOLOGICAL_LENGTHS:
        case = f"sds_L{length:g}"
        for observer in ("r8M", "r12M", "outer"):
            reference_direct = lookup[("schwarzschild", observer, 0)]
            reference_echo = lookup[("schwarzschild", observer, 1)]
            candidate_direct = lookup[(case, observer, 0)]
            candidate_echo = lookup[(case, observer, 1)]
            reference_delay = reference_echo["time"] - reference_direct["time"]
            candidate_delay = candidate_echo["time"] - candidate_direct["time"]
            uncertainty = float(
                np.sqrt(
                    reference_direct["timing_uncertainty"] ** 2
                    + reference_echo["timing_uncertainty"] ** 2
                    + candidate_direct["timing_uncertainty"] ** 2
                    + candidate_echo["timing_uncertainty"] ** 2
                )
            )
            shift = candidate_delay - reference_delay
            rows.append(
                {
                    "cosmological_length_over_M": length,
                    "observer": observer,
                    "schwarzschild_delay_over_M": reference_delay,
                    "sds_delay_over_M": candidate_delay,
                    "delay_shift_over_M": shift,
                    "uncertainty_over_M": uncertainty,
                    "resolved_at_existing_cadence": abs(shift) >= uncertainty,
                    "M_over_L": 1.0 / length,
                    "M2_over_L2": 1.0 / length**2,
                }
            )
    return rows


def plot_delay_scaling(path: Path, rows: list[dict]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fixed_colors = {"r8M": "#0b6e75", "r12M": "#d17a22"}
    for observer, color in fixed_colors.items():
        selected = [row for row in rows if row["observer"] == observer]
        x = np.asarray([row["M2_over_L2"] for row in selected])
        y = np.asarray([row["delay_shift_over_M"] for row in selected])
        error = np.asarray([row["uncertainty_over_M"] for row in selected])
        order = np.argsort(x)
        axes[0].errorbar(x, y, yerr=error, fmt="o", color=color, label=observer)
        grid = np.linspace(0.0, 1.05 * np.max(x), 200)
        coefficient_two = float(np.dot(x, y) / np.dot(x, x))
        root = np.sqrt(x)
        coefficient_one = float(np.dot(root, y) / np.dot(root, root))
        axes[0].plot(grid, coefficient_two * grid, color=color, linewidth=1.5)
        axes[0].plot(
            grid,
            coefficient_one * np.sqrt(grid),
            color=color,
            linewidth=1.0,
            linestyle=":",
        )
    axes[0].set_xlabel(r"$(M/L)^2$")
    axes[0].set_ylabel(r"$\delta\Delta U_1/M$")
    axes[0].legend(title="solid: $L^{-2}$\ndotted: $L^{-1}$")
    axes[0].set_title("Fixed radius observers")

    selected = [row for row in rows if row["observer"] == "outer"]
    x = np.asarray([row["M_over_L"] for row in selected])
    y = np.asarray([row["delay_shift_over_M"] for row in selected])
    error = np.asarray([row["uncertainty_over_M"] for row in selected])
    axes[1].errorbar(x, y, yerr=error, fmt="o", color="#b33b2e", label="archives")
    grid = np.linspace(0.0, 1.05 * np.max(x), 200)
    coefficient_one = float(np.dot(x, y) / np.dot(x, x))
    square = x**2
    coefficient_two = float(np.dot(square, y) / np.dot(square, square))
    axes[1].plot(grid, coefficient_one * grid, color="#111111", label=r"$L^{-1}$")
    axes[1].plot(
        grid,
        coefficient_two * grid**2,
        color="#111111",
        linestyle=":",
        label=r"$L^{-2}$",
    )
    axes[1].set_xlabel(r"$M/L$")
    axes[1].set_ylabel(r"$\delta\Delta U_1/M$")
    axes[1].set_title("Outer boundary")
    axes[1].legend()
    figure.suptitle("Local clock correction and global propagation correction")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def ray_timing_rows(pulse_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for pulse in pulse_rows:
        length = pulse["cosmological_length_over_M"]
        cosmological_length = None if np.isinf(length) else float(length)
        observer_radius = (
            None if pulse["observer"] == "outer" else pulse["observer_radius_over_M"]
        )
        target = pulse["pulse"] * np.pi
        ray = trace_null_ray(
            source_radius=6.0,
            observer_radius=observer_radius,
            target_angle=target,
            emission_time=30.0,
            cosmological_length=cosmological_length,
            winding=int(pulse["pulse"]),
        )
        row = ray.as_dict()
        row.update(
            {
                "case": pulse["case"],
                "observer": pulse["observer"],
                "pulse": pulse["pulse"],
                "measured_U_over_M": pulse["time"],
                "measurement_uncertainty_over_M": pulse["timing_uncertainty"],
                "simulation_minus_ray_over_M": pulse["time"] - ray.arrival_u,
            }
        )
        rows.append(row)
    return rows


def plot_ray_residuals(path: Path, rows: list[dict]) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 3.9), sharey=True)
    for axis, observer in zip(axes, ("r8M", "r12M", "outer")):
        for case, marker in (("schwarzschild", "o"), ("sds_L20", "s"), ("sds_L80", "^")):
            selected = [
                row for row in rows if row["observer"] == observer and row["case"] == case
            ]
            axis.errorbar(
                [row["pulse"] for row in selected],
                [row["simulation_minus_ray_over_M"] for row in selected],
                yerr=[row["measurement_uncertainty_over_M"] for row in selected],
                marker=marker,
                linewidth=1.2,
                label=case.replace("sds_", "SdS "),
            )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_title(observer)
        axis.set_xlabel("pulse")
    axes[0].set_ylabel("simulation minus ray prediction  $(M)$")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Finite source and wave optical timing residuals")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def generic_angular_rows(output_dir: Path) -> tuple[list[dict], list[dict]]:
    reference = load_case(output_dir, "schwarzschild")
    cases: tuple[str | float, ...] = ("schwarzschild", *COSMOLOGICAL_LENGTHS)
    pulse_rows: list[dict] = []
    phase_rows: list[dict] = []
    for gamma, windows in GENERIC_WINDOWS.items():
        reference_times, reference_trace = direction_waveform(reference, gamma)
        for case in cases:
            result = load_case(output_dir, case)
            times, trace = direction_waveform(result, gamma)
            estimates = []
            omega = (
                1.0
                if case == "schwarzschild"
                else np.sqrt(1.0 - 27.0 / float(case) ** 2)
            ) / (3.0 * np.sqrt(3.0))
            for pulse, window in enumerate(windows):
                estimate = estimate_pulse(
                    pulse=pulse,
                    phi=gamma,
                    times=times,
                    trace=trace,
                    reference_times=reference_times,
                    reference_trace=reference_trace,
                    window=window,
                )
                estimates.append(estimate)
            reference_u = estimates[0].time
            for estimate in estimates:
                target = generic_target_angle(gamma, estimate.pulse)
                ray = trace_null_ray(
                    source_radius=6.0,
                    observer_radius=None,
                    target_angle=target,
                    emission_time=30.0,
                    cosmological_length=None if case == "schwarzschild" else float(case),
                    winding=estimate.pulse,
                )
                row = estimate.as_dict()
                row.update(
                    {
                        "case": "schwarzschild" if case == "schwarzschild" else f"sds_L{case:g}",
                        "gamma_over_pi": gamma / np.pi,
                        "Omega_ph_M": omega,
                        "rescaled_U": omega * (estimate.time - reference_u),
                        "target_angle_over_pi": target / np.pi,
                        "ray_U_over_M": ray.arrival_u,
                        "simulation_minus_ray_over_M": estimate.time - ray.arrival_u,
                    }
                )
                if estimate.pulse:
                    previous = estimates[estimate.pulse - 1]
                    row["amplitude_ratio"] = estimate.amplitude / previous.amplitude
                    row["flux_energy_ratio"] = (
                        estimate.integrated_flux_energy / previous.integrated_flux_energy
                    )
                else:
                    row["amplitude_ratio"] = np.nan
                    row["flux_energy_ratio"] = np.nan
                pulse_rows.append(row)
            for first, second in zip(estimates, estimates[1:]):
                phase = local_phase_comparison(times, trace, trace, first, second)
                phase.update(
                    {
                        "case": "schwarzschild" if case == "schwarzschild" else f"sds_L{case:g}",
                        "gamma_over_pi": gamma / np.pi,
                        "pulse_pair": f"{first.pulse}->{second.pulse}",
                        "amplitude_ratio": second.amplitude / first.amplitude,
                        "eikonal_amplitude_ratio": np.exp(-0.5 * np.pi),
                        "phase_residual_from_minus_pi_over_2": float(
                            np.angle(
                                np.exp(1j * (phase["phase_radians"] + 0.5 * np.pi))
                            )
                        ),
                    }
                )
                phase_rows.append(phase)
    return pulse_rows, phase_rows


def plot_angular_collapse(path: Path, rows: list[dict]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for axis, gamma in zip(axes, (1.0 / 3.0, 0.5)):
        reference = [
            row for row in rows if row["gamma_over_pi"] == gamma and row["case"] == "schwarzschild"
        ]
        reference_lookup = {row["pulse"]: row["rescaled_U"] for row in reference}
        for case in ("sds_L20", "sds_L40", "sds_L80", "sds_L160"):
            selected = [
                row for row in rows if row["gamma_over_pi"] == gamma and row["case"] == case
            ]
            axis.plot(
                [row["pulse"] for row in selected],
                [row["rescaled_U"] - reference_lookup[row["pulse"]] for row in selected],
                marker="o",
                linewidth=1.2,
                label=case.replace("sds_", "SdS "),
            )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_title(rf"$\gamma={gamma:g}\pi$")
        axis.set_xlabel("pulse")
        axis.set_ylabel(r"rescaled timing residual")
    axes[-1].legend(fontsize=8)
    figure.suptitle(r"Residual after $\widehat U=\Omega_{\rm ph}(U-U_{\rm ref})$")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_damping_phase(path: Path, rows: list[dict]) -> Path:
    selected = [row for row in rows if row["gamma_over_pi"] == 0.5]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for case in ("schwarzschild", "sds_L20", "sds_L80", "sds_L160"):
        case_rows = [row for row in selected if row["case"] == case]
        x = np.arange(1, len(case_rows) + 1)
        axes[0].plot(x, [row["amplitude_ratio"] for row in case_rows], marker="o", label=case)
        axes[1].plot(x, [row["phase_radians"] for row in case_rows], marker="o", label=case)
    axes[0].axhline(np.exp(-0.5 * np.pi), color="#111111", linestyle=":", label=r"$e^{-\pi/2}$")
    axes[1].axhline(-0.5 * np.pi, color="#111111", linestyle=":", label=r"$-\pi/2$")
    axes[0].set_ylabel("matched amplitude ratio")
    axes[1].set_ylabel("local complex phase  (rad)")
    for axis in axes:
        axis.set_xlabel("consecutive pulse pair")
        axis.legend(fontsize=8)
    figure.suptitle("Finite frequency damping and Maslov phase")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def source_convergence_rows() -> list[dict]:
    rows: list[dict] = []
    for scale in (1.0, 0.7, 0.5):
        source = LocalizedSourceParameters(
            radial_half_width=1.5 * scale,
            time_half_width=4.0 * scale,
            angular_concentration=16.0 / scale**2,
        )
        ell_max = minimum_ell_max(
            source.angular_concentration, omitted_power_tolerance=1e-10
        )
        tests = {
            "constant": lambda time, radius, cosine: np.ones_like(
                time + radius + cosine
            ),
            "linear": lambda time, radius, cosine: time + radius + cosine,
            "quadratic": lambda time, radius, cosine: time**2 + radius**2 + cosine**2,
        }
        exact = {
            "constant": 1.0,
            "linear": source.time_center + source.center_radius + 1.0,
            "quadratic": source.time_center**2 + source.center_radius**2 + 1.0,
        }
        for name, function in tests.items():
            measured = weak_source_integral(source, function, quadrature_order=64)
            rows.append(
                {
                    "width_scale": scale,
                    "radial_half_width_over_M": source.radial_half_width,
                    "time_half_width_over_M": source.time_half_width,
                    "angular_sigma_radians": source.angular_width,
                    "ell_max_for_omitted_power_below_1e_10": ell_max,
                    "omitted_angular_power": 1.0
                    - retained_angular_fraction(source.angular_concentration, ell_max),
                    "test_function": name,
                    "integral": measured,
                    "delta_limit": exact[name],
                    "absolute_error": abs(measured - exact[name]),
                }
            )
    return rows


def create_reanalysis(output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    pulse_rows, phase_rows = analyze_existing_archives(output_dir)
    delay_rows = delay_shift_rows(pulse_rows)
    ray_rows = ray_timing_rows(pulse_rows)
    angular_rows, angular_phase_rows = generic_angular_rows(output_dir)
    source_rows = source_convergence_rows()
    written = [
        _write_csv(tables / "local_pulse_observables.csv", pulse_rows),
        _write_csv(tables / "local_phase_observables.csv", phase_rows),
        _write_csv(tables / "local_delay_scaling.csv", delay_rows),
        _write_csv(tables / "exact_null_ray_timing.csv", ray_rows),
        _write_csv(tables / "generic_angle_pulses.csv", angular_rows),
        _write_csv(tables / "generic_angle_phase.csv", angular_phase_rows),
        _write_csv(tables / "normalized_source_weak_convergence.csv", source_rows),
        plot_delay_scaling(output_dir / "local_vs_outer_timing_scaling.png", delay_rows),
        plot_ray_residuals(output_dir / "simulation_minus_null_ray.png", ray_rows),
        plot_angular_collapse(output_dir / "rescaled_caustic_collapse.png", angular_rows),
        plot_damping_phase(output_dir / "damping_and_phase.png", angular_phase_rows),
    ]
    summary = {
        "pulse_observables": pulse_rows,
        "phase_observables": phase_rows,
        "delay_scaling": delay_rows,
        "null_rays": ray_rows,
        "generic_angles": angular_rows,
        "generic_phase": angular_phase_rows,
        "normalized_source": source_rows,
        "analysis_environment": reproducibility_metadata(),
    }
    summary_path = output_dir / "caustic_reanalysis_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    written.append(summary_path)
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/green_function"))
    arguments = parser.parse_args()
    for path in create_reanalysis(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
