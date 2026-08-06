"""Final figures and tables for the normalized caustic production suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .caustic_analysis import PULSE_WINDOWS, estimate_pulse, local_phase_comparison
from .caustic_reanalysis import (
    delay_shift_rows,
    plot_damping_phase,
    plot_delay_scaling,
    plot_ray_residuals,
    ray_timing_rows,
)
from .caustic_study import direction_waveform
from .null_geodesics import generic_target_angle, trace_null_ray
from .production_analysis import (
    convergence_rows,
    d1_convergence_rows,
    pulse_measurements,
)
from .source_evolution import SourcedSimulationResult, load_sourced_result
from .tail_analysis import json_safe
from .three_d_solver import real_spherical_harmonic


CASES = ("schwarzschild", "sds_L12", "sds_L20", "sds_L40", "sds_L80", "sds_L160")
LENGTHS = {
    "schwarzschild": np.inf,
    "sds_L12": 12.0,
    "sds_L20": 20.0,
    "sds_L40": 40.0,
    "sds_L80": 80.0,
    "sds_L160": 160.0,
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


def _observer_label(result: SourcedSimulationResult, index: int) -> str:
    if index == result.outer_index():
        return "outer"
    return f"r{result.observer_areal_radius[index]:g}M"


def _observer_lookup(result: SourcedSimulationResult) -> dict[str, int]:
    return {
        _observer_label(result, index): index
        for index in range(result.observer_areal_radius.size)
    }


def _load_cases(output_dir: Path) -> dict[str, SourcedSimulationResult]:
    return {
        case: load_sourced_result(Path(output_dir) / "raw" / f"{case}.npz")
        for case in CASES
    }


def production_pulse_rows(
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    results = _load_cases(output_dir)
    reference = results["schwarzschild"]
    reference_observers = _observer_lookup(reference)
    pulse_rows: list[dict] = []
    phase_rows: list[dict] = []
    for case, result in results.items():
        for observer_label, observer in _observer_lookup(result).items():
            reference_observer = reference_observers[observer_label]
            traces = {
                phi: direction_waveform(result, phi, observer)[1]
                for phi in (0.0, np.pi)
            }
            reference_traces = {
                phi: direction_waveform(reference, phi, reference_observer)[1]
                for phi in (0.0, np.pi)
            }
            estimates = []
            for pulse, (start, end, phi) in enumerate(PULSE_WINDOWS):
                estimate = estimate_pulse(
                    pulse=pulse,
                    phi=phi,
                    times=result.retarded_time,
                    trace=traces[phi],
                    reference_times=reference.retarded_time,
                    reference_trace=reference_traces[phi],
                    window=(start, end),
                )
                estimates.append(estimate)
                row = estimate.as_dict()
                row.update(
                    {
                        "case": case,
                        "cosmological_length_over_M": LENGTHS[case],
                        "observer": observer_label,
                        "observer_radius_over_M": float(
                            result.observer_areal_radius[observer]
                        ),
                        "output_cadence_over_M": float(
                            np.median(np.diff(result.retarded_time))
                        ),
                    }
                )
                if pulse:
                    previous = estimates[pulse - 1]
                    row["delay_over_M"] = estimate.time - previous.time
                    row["amplitude_ratio"] = estimate.amplitude / previous.amplitude
                    row["flux_energy_ratio"] = (
                        estimate.integrated_flux_energy
                        / previous.integrated_flux_energy
                    )
                else:
                    row["delay_over_M"] = np.nan
                    row["amplitude_ratio"] = np.nan
                    row["flux_energy_ratio"] = np.nan
                pulse_rows.append(row)
            for first, second in zip(estimates, estimates[1:]):
                phase = local_phase_comparison(
                    result.retarded_time,
                    traces[first.phi],
                    traces[second.phi],
                    first,
                    second,
                )
                phase.update(
                    {
                        "case": case,
                        "observer": observer_label,
                        "pulse_pair": f"{first.pulse}->{second.pulse}",
                        "amplitude_ratio": second.amplitude / first.amplitude,
                        "eikonal_amplitude_ratio": np.exp(-0.5 * np.pi),
                        "phase_residual_from_minus_pi_over_2": float(
                            np.angle(
                                np.exp(
                                    1j
                                    * (phase["phase_radians"] + 0.5 * np.pi)
                                )
                            )
                        ),
                    }
                )
                phase_rows.append(phase)
    return pulse_rows, phase_rows


def _monopole_trace(result: SourcedSimulationResult, observer: int) -> np.ndarray:
    mode = int(np.flatnonzero((result.mode_ell == 0) & (result.mode_m == 0))[0])
    if result.response_signals.size:
        ell_index = int(np.flatnonzero(result.response_ell == 0)[0])
        response = result.response_signals[:, observer, ell_index]
        modal = response * result.mode_source_amplitude[mode]
    else:
        modal = result.modal_signals[:, observer, mode]
    y00 = float(real_spherical_harmonic(0, 0, np.asarray(np.pi / 2), np.asarray(0.0)))
    return modal * y00


def monopole_subtracted_rows(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for case in ("sds_L12", "sds_L20"):
        result = load_sourced_result(Path(output_dir) / "raw" / f"{case}.npz")
        observer = result.outer_index()
        monopole = _monopole_trace(result, observer)
        for pulse, (start, end, phi) in enumerate(PULSE_WINDOWS):
            times, full = direction_waveform(result, phi, observer)
            subtracted = full - monopole
            full_estimate = estimate_pulse(
                pulse=pulse,
                phi=phi,
                times=times,
                trace=full,
                reference_times=times,
                reference_trace=full,
                window=(start, end),
            )
            subtracted_estimate = estimate_pulse(
                pulse=pulse,
                phi=phi,
                times=times,
                trace=subtracted,
                reference_times=times,
                reference_trace=subtracted,
                window=(start, end),
            )
            rows.append(
                {
                    "case": case,
                    "pulse": pulse,
                    "full_time_over_M": full_estimate.time,
                    "monopole_subtracted_time_over_M": subtracted_estimate.time,
                    "timing_difference_over_M": (
                        subtracted_estimate.time - full_estimate.time
                    ),
                    "full_amplitude": full_estimate.amplitude,
                    "monopole_subtracted_amplitude": subtracted_estimate.amplitude,
                    "full_flux_energy": full_estimate.integrated_flux_energy,
                    "monopole_subtracted_flux_energy": (
                        subtracted_estimate.integrated_flux_energy
                    ),
                }
            )
    return rows


def generic_angle_rows(output_dir: Path) -> tuple[list[dict], list[dict]]:
    results = _load_cases(output_dir)
    reference = results["schwarzschild"]
    windows = {
        np.pi / 3.0: ((20.0, 42.0), (44.0, 62.0), (61.0, 77.0), (77.0, 98.0)),
        np.pi / 2.0: ((22.0, 43.0), (44.0, 61.0), (60.0, 77.0), (76.0, 96.0)),
    }
    pulse_rows: list[dict] = []
    phase_rows: list[dict] = []
    for gamma, pulse_windows in windows.items():
        reference_times, reference_trace = direction_waveform(reference, gamma)
        for case, result in results.items():
            times, trace = direction_waveform(result, gamma)
            length = LENGTHS[case]
            omega = (
                1.0 if np.isinf(length) else np.sqrt(1.0 - 27.0 / length**2)
            ) / (3.0 * np.sqrt(3.0))
            estimates = [
                estimate_pulse(
                    pulse=pulse,
                    phi=gamma,
                    times=times,
                    trace=trace,
                    reference_times=reference_times,
                    reference_trace=reference_trace,
                    window=window,
                )
                for pulse, window in enumerate(pulse_windows)
            ]
            reference_u = estimates[0].time
            for estimate in estimates:
                target = generic_target_angle(gamma, estimate.pulse)
                ray = trace_null_ray(
                    source_radius=6.0,
                    observer_radius=None,
                    target_angle=target,
                    emission_time=30.0,
                    cosmological_length=None if np.isinf(length) else length,
                    winding=estimate.pulse,
                )
                row = estimate.as_dict()
                row.update(
                    {
                        "case": case,
                        "gamma_over_pi": gamma / np.pi,
                        "Omega_ph_M": omega,
                        "rescaled_U": omega * (estimate.time - reference_u),
                        "ray_U_over_M": ray.arrival_u,
                        "simulation_minus_ray_over_M": estimate.time
                        - ray.arrival_u,
                    }
                )
                if estimate.pulse:
                    previous = estimates[estimate.pulse - 1]
                    row["amplitude_ratio"] = estimate.amplitude / previous.amplitude
                pulse_rows.append(row)
            for first, second in zip(estimates, estimates[1:]):
                phase = local_phase_comparison(times, trace, trace, first, second)
                phase.update(
                    {
                        "case": case,
                        "gamma_over_pi": gamma / np.pi,
                        "pulse_pair": f"{first.pulse}->{second.pulse}",
                        "amplitude_ratio": second.amplitude / first.amplitude,
                        "eikonal_amplitude_ratio": np.exp(-0.5 * np.pi),
                    }
                )
                phase_rows.append(phase)
    return pulse_rows, phase_rows


def scaling_fit_rows(delay_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for observer in ("r8M", "r12M", "outer"):
        selected = [row for row in delay_rows if row["observer"] == observer]
        lengths = np.asarray(
            [row["cosmological_length_over_M"] for row in selected], dtype=float
        )
        shifts = np.asarray(
            [row["delay_shift_over_M"] for row in selected], dtype=float
        )
        for fit_range, mask in (
            ("L20_through_L160", np.ones(lengths.size, dtype=bool)),
            ("L20_through_L80", lengths <= 80.0),
        ):
            x = 1.0 / lengths[mask]
            y = shifts[mask]
            if observer == "outer":
                design = np.column_stack([x, x**2])
                model = "b1_M_over_L_plus_b2_M2_over_L2"
                first_name, second_name = "b1", "b2"
            else:
                design = np.column_stack([x**2, x**4])
                model = "a2_M2_over_L2_plus_a4_M4_over_L4"
                first_name, second_name = "a2", "a4"
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficients
            slope, intercept = np.polyfit(
                np.log(lengths[mask]), np.log(np.abs(y)), 1
            )
            rows.append(
                {
                    "observer": observer,
                    "fit_range": fit_range,
                    "motivated_model": model,
                    first_name: float(coefficients[0]),
                    second_name: float(coefficients[1]),
                    "root_mean_square_residual_over_M": float(
                        np.sqrt(np.mean(residual**2))
                    ),
                    "effective_power_over_simulated_interval": float(-slope),
                    "effective_power_coefficient": float(np.exp(intercept)),
                    "effective_power_is_asymptotic_claim": False,
                }
            )
    return rows


def source_width_delay_rows(output_dir: Path) -> list[dict]:
    raw = Path(output_dir) / "pilots" / "raw"
    rows: list[dict] = []
    for length in (20, 80):
        shifts: dict[tuple[str, int], float] = {}
        narrow = load_sourced_result(raw / f"width_sds_L{length}_w0p5.npz")
        observer_lookup = {
            _observer_label(narrow, index): index
            for index in range(narrow.observer_areal_radius.size)
        }
        for width in ("1p0", "0p7", "0p5"):
            schwarzschild = load_sourced_result(raw / f"width_w{width}.npz")
            sds = load_sourced_result(raw / f"width_sds_L{length}_w{width}.npz")
            schwarzschild_pulses, _ = pulse_measurements(
                schwarzschild, schwarzschild
            )
            sds_pulses, _ = pulse_measurements(sds, schwarzschild)
            for observer in range(sds.observer_areal_radius.size):
                schwarzschild_selected = [
                    row
                    for row in schwarzschild_pulses
                    if row["observer_index"] == observer
                ]
                sds_selected = [
                    row for row in sds_pulses if row["observer_index"] == observer
                ]
                shift = (
                    sds_selected[1]["time"]
                    - sds_selected[0]["time"]
                    - schwarzschild_selected[1]["time"]
                    + schwarzschild_selected[0]["time"]
                )
                matched_shift = (
                    sds_selected[1]["matched_time"]
                    - sds_selected[0]["matched_time"]
                    - schwarzschild_selected[1]["matched_time"]
                    + schwarzschild_selected[0]["matched_time"]
                )
                shifts[(width, observer)] = shift
                rows.append(
                    {
                        "cosmological_length_over_M": length,
                        "width_scale": width.replace("p", "."),
                        "observer": _observer_label(sds, observer),
                        "delay_shift_over_M": shift,
                        "primary_estimator": "tapered_analytic_envelope",
                        "matched_template_delay_shift_over_M": matched_shift,
                        "estimator_sensitivity_over_M": abs(shift - matched_shift),
                        "difference_from_narrow_over_M": np.nan,
                    }
                )
        for row in rows:
            if row["cosmological_length_over_M"] != length:
                continue
            observer = observer_lookup[row["observer"]]
            width = row["width_scale"].replace(".", "p")
            row["difference_from_narrow_over_M"] = abs(
                shifts[(width, observer)] - shifts[("0p5", observer)]
            )
    return rows


def plot_production_collapse(path: Path, rows: list[dict]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for axis, gamma in zip(axes, (1.0 / 3.0, 0.5)):
        reference = {
            row["pulse"]: row["rescaled_U"]
            for row in rows
            if row["case"] == "schwarzschild" and row["gamma_over_pi"] == gamma
        }
        for case in CASES[1:]:
            selected = [
                row
                for row in rows
                if row["case"] == case and row["gamma_over_pi"] == gamma
            ]
            axis.plot(
                [row["pulse"] for row in selected],
                [row["rescaled_U"] - reference[row["pulse"]] for row in selected],
                marker="o",
                label=case,
            )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_title(rf"$\gamma={gamma:g}\pi$")
        axis.set_xlabel("pulse")
        axis.set_ylabel("rescaled timing residual")
    axes[-1].legend(fontsize=7)
    figure.suptitle("Normalized source clock collapse residual")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_d1_convergence(path: Path, rows: list[dict]) -> Path:
    """Plot direct four time D1 differences for each representative geometry."""

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    observers = ((0, "r8M"), (1, "r12M"), (2, "outer"))
    categories = ("spatial", "temporal", "angular")
    for axis, length in zip(axes, (20, 80)):
        for observer_index, label in observers:
            selected = {
                row["category"]: row["D1_error_over_M"]
                for row in rows
                if row["cosmological_length_over_M"] == length
                and row["observer_index"] == observer_index
            }
            axis.plot(
                categories,
                [selected[category] for category in categories],
                marker="o",
                linewidth=1.5,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_title(rf"$L/M={length}$")
        axis.set_xlabel("paired refinement comparison")
        axis.grid(axis="y", which="both", alpha=0.2)
    axes[0].set_ylabel(r"direct $D_1$ difference  $(M)$")
    axes[1].legend()
    figure.suptitle(r"Correlated four arrival time convergence of $D_1$")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def error_budget_rows(output_dir: Path, pulse_rows: list[dict]) -> list[dict]:
    """Build a D1 budget without decomposing it into independent pulse errors."""

    delay_rows = delay_shift_rows(
        [row for row in pulse_rows if row["case"] != "sds_L12"]
    )
    convergence = d1_convergence_rows(output_dir)
    width_rows = source_width_delay_rows(output_dir)
    observer_index = {"r8M": 0, "r12M": 1, "outer": 2}
    rows: list[dict] = []
    for measured in delay_rows:
        length = int(measured["cosmological_length_over_M"])
        observer = measured["observer"]
        index = observer_index[observer]
        components = {
            row["category"]: row["D1_error_over_M"]
            for row in convergence
            if row["cosmological_length_over_M"] == length
            and row["observer_index"] == index
        }
        width = next(
            (
                row["difference_from_narrow_over_M"]
                for row in width_rows
                if row["cosmological_length_over_M"] == length
                and row["observer"] == observer
                and row["width_scale"] == "0.7"
            ),
            np.nan,
        )
        complete = all(
            name in components for name in ("spatial", "temporal", "angular")
        ) and np.isfinite(width)
        values = [
            *components.values(),
            width,
        ]
        total = float(np.sqrt(np.sum(np.square(values)))) if complete else np.nan
        rows.append(
            {
                **measured,
                "spatial_D1_error_over_M": components.get("spatial", np.nan),
                "temporal_D1_error_over_M": components.get("temporal", np.nan),
                "angular_D1_error_over_M": components.get("angular", np.nan),
                "same_L_source_width_D1_sensitivity_over_M": width,
                "total_primary_D1_uncertainty_over_M": total,
                "budget_complete": complete,
                "D1_resolved": bool(
                    complete and abs(measured["delay_shift_over_M"]) > total
                ),
            }
        )
    return rows


def create_report(output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    pulse_rows, phase_rows = production_pulse_rows(output_dir)
    delay_rows = delay_shift_rows(
        [row for row in pulse_rows if row["case"] != "sds_L12"]
    )
    width_delay_rows = source_width_delay_rows(output_dir)
    ray_rows = ray_timing_rows(pulse_rows)
    monopole_rows = monopole_subtracted_rows(output_dir)
    angular_rows, angular_phase_rows = generic_angle_rows(output_dir)
    convergence = convergence_rows(output_dir)
    d1_convergence = d1_convergence_rows(output_dir)
    budget_rows = error_budget_rows(output_dir, pulse_rows)
    budget_lookup = {
        (row["cosmological_length_over_M"], row["observer"]): row
        for row in budget_rows
    }
    for row in delay_rows:
        budget = budget_lookup[(row["cosmological_length_over_M"], row["observer"])]
        row["total_uncertainty_over_M"] = budget[
            "total_primary_D1_uncertainty_over_M"
        ]
        row["budget_complete"] = budget["budget_complete"]
    fit_rows = scaling_fit_rows(delay_rows)
    written = [
        _write_csv(tables / "production_pulses.csv", pulse_rows),
        _write_csv(tables / "production_phase.csv", phase_rows),
        _write_csv(tables / "production_delay_scaling.csv", delay_rows),
        _write_csv(tables / "production_scaling_fits.csv", fit_rows),
        _write_csv(tables / "source_width_delay_sensitivity.csv", width_delay_rows),
        _write_csv(tables / "production_null_rays.csv", ray_rows),
        _write_csv(tables / "monopole_subtracted.csv", monopole_rows),
        _write_csv(tables / "production_generic_angles.csv", angular_rows),
        _write_csv(tables / "production_generic_phase.csv", angular_phase_rows),
        _write_csv(tables / "full_error_budget.csv", budget_rows),
        _write_csv(tables / "observable_convergence.csv", convergence),
        _write_csv(tables / "D1_convergence.csv", d1_convergence),
        plot_d1_convergence(output_dir / "D1_convergence.png", d1_convergence),
        plot_delay_scaling(output_dir / "production_timing_scaling.png", delay_rows),
        plot_ray_residuals(output_dir / "production_ray_residuals.png", ray_rows),
        plot_production_collapse(
            output_dir / "production_clock_collapse.png", angular_rows
        ),
        plot_damping_phase(
            output_dir / "production_damping_phase.png", angular_phase_rows
        ),
    ]
    summary = {
        "pulses": pulse_rows,
        "phase": phase_rows,
        "delay_scaling": delay_rows,
        "scaling_fits": fit_rows,
        "source_width_delay_sensitivity": width_delay_rows,
        "null_rays": ray_rows,
        "monopole_subtracted": monopole_rows,
        "generic_angles": angular_rows,
        "generic_phase": angular_phase_rows,
        "convergence": convergence,
        "D1_convergence": d1_convergence,
        "error_budget": budget_rows,
    }
    path = output_dir / "production_summary.json"
    path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    written.append(path)
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/caustic_production")
    )
    arguments = parser.parse_args()
    for path in create_report(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
