"""Observable level convergence and cross code production analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .caustic_analysis import PULSE_WINDOWS, estimate_pulse, local_phase_comparison
from .caustic_study import direction_waveform, harmonic_matrix
from .source_evolution import SourcedSimulationResult, load_sourced_result


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


def _response_signals(result: SourcedSimulationResult, observer: int) -> dict[int, np.ndarray]:
    if result.response_signals.size:
        return {
            int(ell): result.response_signals[:, observer, index]
            for index, ell in enumerate(result.response_ell)
        }
    responses: dict[int, np.ndarray] = {}
    for ell in np.unique(result.mode_ell):
        indices = np.flatnonzero(result.mode_ell == ell)
        index = int(indices[np.argmax(np.abs(result.mode_source_amplitude[indices]))])
        responses[int(ell)] = (
            result.modal_signals[:, observer, index]
            / result.mode_source_amplitude[index]
        )
    return responses


def sphere_integrated_relative_l2(
    candidate: SourcedSimulationResult,
    reference: SourcedSimulationResult,
    *,
    observer: int | None = None,
    window: tuple[float, float] = (18.0, 96.0),
    samples: int = 7801,
) -> float:
    """Return the exact sphere integrated modal waveform disagreement."""

    candidate_observer = candidate.outer_index() if observer is None else observer
    reference_observer = reference.outer_index() if observer is None else observer
    grid = np.linspace(window[0], window[1], samples)
    candidate_responses = _response_signals(candidate, candidate_observer)
    reference_responses = _response_signals(reference, reference_observer)
    candidate_modes = {
        (int(ell), int(order)): float(amplitude)
        for ell, order, amplitude in zip(
            candidate.mode_ell,
            candidate.mode_m,
            candidate.mode_source_amplitude,
        )
    }
    reference_modes = {
        (int(ell), int(order)): float(amplitude)
        for ell, order, amplitude in zip(
            reference.mode_ell,
            reference.mode_m,
            reference.mode_source_amplitude,
        )
    }
    error = np.zeros(grid.size)
    norm = np.zeros(grid.size)
    for mode in candidate_modes.keys() | reference_modes.keys():
        ell = mode[0]
        candidate_response = (
            np.interp(grid, candidate.retarded_time, candidate_responses[ell])
            if ell in candidate_responses
            else np.zeros(grid.size)
        )
        reference_response = (
            np.interp(grid, reference.retarded_time, reference_responses[ell])
            if ell in reference_responses
            else np.zeros(grid.size)
        )
        candidate_mode = candidate_modes.get(mode, 0.0) * candidate_response
        reference_mode = reference_modes.get(mode, 0.0) * reference_response
        error += (candidate_mode - reference_mode) ** 2
        norm += reference_mode**2
    return float(np.sqrt(np.trapezoid(error, grid) / np.trapezoid(norm, grid)))


def angular_focus_width(
    result: SourcedSimulationResult,
    *,
    observer: int,
    pulse_time: float,
    axis_phi: float,
    samples: int = 1441,
) -> float:
    """Return the full width at half maximum around one caustic axis."""

    offsets = np.linspace(-0.6, 0.6, samples)
    phi = np.mod(axis_phi + offsets, 2.0 * np.pi)
    basis = harmonic_matrix(result, np.full(phi.size, 0.5 * np.pi), phi)
    if result.response_signals.size:
        responses = _response_signals(result, observer)
        radial = np.asarray(
            [
                np.interp(pulse_time, result.retarded_time, responses[int(ell)])
                for ell in result.response_ell
            ]
        )
        angular_weights = result.mode_source_amplitude[:, None] * basis
        ell_weights = np.asarray(
            [
                np.sum(angular_weights[result.mode_ell == ell], axis=0)
                for ell in result.response_ell
            ]
        )
        profile = radial @ ell_weights
    else:
        modal = np.asarray(
            [
                np.interp(
                    pulse_time,
                    result.retarded_time,
                    result.modal_signals[:, observer, mode],
                )
                for mode in range(result.mode_ell.size)
            ]
        )
        profile = modal @ basis
    magnitude = np.abs(profile)
    center = samples // 2
    peak = magnitude[center]
    half = 0.5 * peak
    left_candidates = np.flatnonzero(magnitude[:center] <= half)
    right_candidates = np.flatnonzero(magnitude[center:] <= half)
    if not left_candidates.size or not right_candidates.size:
        return np.nan
    left = int(left_candidates[-1])
    right = center + int(right_candidates[0])
    return float(offsets[right] - offsets[left])


def pulse_measurements(
    result: SourcedSimulationResult,
    reference: SourcedSimulationResult,
) -> tuple[list[dict], list[dict]]:
    pulse_rows: list[dict] = []
    phase_rows: list[dict] = []
    observer_count = min(
        result.observer_areal_radius.size, reference.observer_areal_radius.size
    )
    for observer in range(observer_count):
        traces = {
            phi: direction_waveform(result, phi, observer)[1]
            for phi in (0.0, np.pi)
        }
        reference_traces = {
            phi: direction_waveform(reference, phi, observer)[1]
            for phi in (0.0, np.pi)
        }
        estimates = []
        available_windows = [
            window
            for window in PULSE_WINDOWS
            if window[1]
            <= min(float(result.retarded_time[-1]), float(reference.retarded_time[-1]))
        ]
        for pulse, (start, end, phi) in enumerate(available_windows):
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
                    "observer_index": observer,
                    "observer_radius_over_M": float(result.observer_areal_radius[observer]),
                    "angular_focus_width_radians": angular_focus_width(
                        result,
                        observer=observer,
                        pulse_time=estimate.time,
                        axis_phi=phi,
                    ),
                }
            )
            if pulse:
                previous = estimates[pulse - 1]
                row["delay_over_M"] = estimate.time - previous.time
                row["amplitude_ratio"] = estimate.amplitude / previous.amplitude
                row["flux_energy_ratio"] = (
                    estimate.integrated_flux_energy / previous.integrated_flux_energy
                )
            pulse_rows.append(row)
        for first, second in zip(estimates, estimates[1:]):
            row = local_phase_comparison(
                result.retarded_time,
                traces[first.phi],
                traces[second.phi],
                first,
                second,
            )
            row.update(
                {
                    "observer_index": observer,
                    "observer_radius_over_M": float(result.observer_areal_radius[observer]),
                    "pulse_pair": f"{first.pulse}->{second.pulse}",
                }
            )
            phase_rows.append(row)
    return pulse_rows, phase_rows


def compare_observables(
    candidate: SourcedSimulationResult,
    reference: SourcedSimulationResult,
    *,
    category: str,
    setting: str,
    reference_setting: str,
) -> list[dict]:
    candidate_pulses, candidate_phases = pulse_measurements(candidate, reference)
    reference_pulses, reference_phases = pulse_measurements(reference, reference)
    rows: list[dict] = []
    waveform_l2 = sphere_integrated_relative_l2(candidate, reference)
    for candidate_row, reference_row in zip(candidate_pulses, reference_pulses):
        rows.append(
            {
                "category": category,
                "setting": setting,
                "reference_setting": reference_setting,
                "observer_index": candidate_row["observer_index"],
                "observer_radius_over_M": candidate_row["observer_radius_over_M"],
                "pulse": candidate_row["pulse"],
                "sphere_integrated_relative_l2": waveform_l2,
                "arrival_time_error_over_M": abs(candidate_row["time"] - reference_row["time"]),
                "delay_error_over_M": abs(
                    candidate_row.get("delay_over_M", np.nan)
                    - reference_row.get("delay_over_M", np.nan)
                ),
                "relative_amplitude_error": abs(
                    candidate_row["amplitude"] / reference_row["amplitude"] - 1.0
                ),
                "relative_flux_energy_error": abs(
                    candidate_row["integrated_flux_energy"]
                    / reference_row["integrated_flux_energy"]
                    - 1.0
                ),
                "angular_focus_width_error_radians": abs(
                    candidate_row["angular_focus_width_radians"]
                    - reference_row["angular_focus_width_radians"]
                ),
                "estimator_timing_systematic_over_M": candidate_row[
                    "timing_systematic"
                ],
            }
        )
    phase_lookup = {
        (row["observer_index"], row["pulse_pair"]): row for row in reference_phases
    }
    for row, measured in zip(rows, candidate_pulses):
        if measured["pulse"] == 0:
            row["phase_error_radians"] = np.nan
            continue
        pair = f"{measured['pulse'] - 1}->{measured['pulse']}"
        candidate_phase = next(
            value
            for value in candidate_phases
            if value["observer_index"] == measured["observer_index"]
            and value["pulse_pair"] == pair
        )
        reference_phase = phase_lookup[(measured["observer_index"], pair)]
        row["phase_error_radians"] = abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        candidate_phase["phase_radians"]
                        - reference_phase["phase_radians"]
                    )
                )
            )
        )
    return rows


def convergence_rows(output_dir: Path) -> list[dict]:
    raw = Path(output_dir) / "pilots" / "raw"
    rows: list[dict] = []
    groups = {
        "spatial": ([f"radial_N{value}" for value in (768, 1024, 1536)], "radial_N2048"),
        "temporal": (["temporal_dt0.004", "temporal_dt0.002"], "temporal_dt0.001"),
    }
    for scale, cutoffs in ((1.0, (19, 23, 27)), (0.7, (27, 31, 35)), (0.5, (38, 42, 46))):
        label = str(scale).replace(".", "p")
        groups[f"angular_width_{scale:g}"] = (
            [f"angular_w{label}_lmax{cutoffs[0]}", f"angular_w{label}_lmax{cutoffs[1]}"],
            f"angular_w{label}_lmax{cutoffs[2]}",
        )
    groups["source_width"] = (["width_w1p0", "width_w0p7"], "width_w0p5")
    for category, (settings, reference_setting) in groups.items():
        reference = load_sourced_result(raw / f"{reference_setting}.npz")
        for setting in settings:
            candidate = load_sourced_result(raw / f"{setting}.npz")
            rows.extend(
                compare_observables(
                    candidate,
                    reference,
                    category=category,
                    setting=setting,
                    reference_setting=reference_setting,
                )
            )
    return rows


def cross_code_rows(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for case in ("schwarzschild", "sds_L80"):
        finite_difference = load_sourced_result(
            Path(output_dir) / "cross_code" / "finite_difference" / f"{case}.npz"
        )
        dedalus = load_sourced_result(
            Path(output_dir) / "cross_code" / "dedalus" / f"{case}.npz"
        )
        comparison = compare_observables(
            dedalus,
            finite_difference,
            category="cross_code",
            setting=f"dedalus_{case}",
            reference_setting=f"finite_difference_{case}",
        )
        for row in comparison:
            row["case"] = case
        rows.extend(comparison)
    return rows


def plot_convergence(path: Path, rows: list[dict]) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    metrics = (
        ("sphere_integrated_relative_l2", "sphere integrated relative $L^2$", 5e-4),
        ("arrival_time_error_over_M", "arrival error  $(M)$", 0.01),
        ("relative_amplitude_error", "relative amplitude error", 0.005),
        ("phase_error_radians", "phase error  (rad)", 0.05),
    )
    categories = list(dict.fromkeys(row["category"] for row in rows))
    for axis, (metric, label, target) in zip(axes.flat, metrics):
        for category in categories:
            selected = [
                row
                for row in rows
                if row["category"] == category
                and row["observer_index"] == 0
                and row["pulse"] == 1
                and np.isfinite(row[metric])
            ]
            axis.plot(
                np.arange(len(selected)),
                [row[metric] for row in selected],
                marker="o",
                label=category,
            )
        axis.axhline(target, color="#111111", linestyle=":", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.set_xlabel("successive refinement setting")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Observable level convergence against requested targets")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def create_analysis(output_dir: Path, *, include_cross_code: bool = False) -> list[Path]:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    rows = convergence_rows(output_dir)
    written = [
        _write_csv(tables / "observable_convergence.csv", rows),
        plot_convergence(output_dir / "observable_convergence.png", rows),
    ]
    summary: dict = {"convergence": rows}
    if include_cross_code:
        cross_rows = cross_code_rows(output_dir)
        written.append(_write_csv(tables / "cross_code_observables.csv", cross_rows))
        summary["cross_code"] = cross_rows
    path = output_dir / "production_analysis.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    written.append(path)
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/caustic_production")
    )
    parser.add_argument("--include-cross-code", action="store_true")
    arguments = parser.parse_args()
    for path in create_analysis(
        arguments.output_dir, include_cross_code=arguments.include_cross_code
    ):
        print(path)


if __name__ == "__main__":
    main()
