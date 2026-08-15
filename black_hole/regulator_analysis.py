"""Analysis and figures for the artificial-cosmology regulator experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.signal import hilbert
from scipy.signal.windows import tukey

from .caustic_analysis import estimate_pulse
from .caustic_study import direction_waveform, harmonic_matrix
from .regulator_suite import FLAT_LENGTHS, LEVELS, SOURCE_LENGTHS
from .sds_result import SdSSimulationResult, load_sds_result
from .source_evolution import SourcedSimulationResult, load_sourced_result


CUMULATIVE_WINDOWS = (
    ("prompt_and_early_ringdown", 0.0, 40.0),
    ("radiative_signal", 0.0, 80.0),
    ("extended_finite_time", 0.0, 160.0),
)
DISJOINT_WINDOWS = (
    ("prompt", 0.0, 40.0),
    ("early_ringdown", 40.0, 80.0),
    ("late_time", 80.0, 160.0),
)
FIXED_WINDOWS = CUMULATIVE_WINDOWS
ANALYSIS_WINDOWS = CUMULATIVE_WINDOWS + DISJOINT_WINDOWS[1:]
WINDOW_LATEX_LABELS = {
    "prompt_and_early_ringdown": r"$0\leq U/M\leq 40$",
    "radiative_signal": r"$0\leq U/M\leq 80$",
    "extended_finite_time": r"$0\leq U/M\leq 160$",
    "prompt": r"$0\leq U/M\leq 40$",
    "early_ringdown": r"$40\leq U/M\leq 80$",
    "late_time": r"$80\leq U/M\leq 160$",
}
COMPARISON_LATEX_LABELS = {
    "Winf_L80_vs_Schwarzschild": r"$W_\infty^{(80)}-W_{\rm Schw}$",
    "Winf_L160_vs_Schwarzschild": r"$W_\infty^{(160)}-W_{\rm Schw}$",
    "Winf_L80_vs_Winf_L160": r"$W_\infty^{(80)}-W_\infty^{(160)}$",
}
WINDOW_VARIANTS = {
    "primary": ((18.0, 35.0), (35.0, 53.0)),
    "inset_0p5M": ((18.5, 34.5), (35.5, 52.5)),
    "expanded_0p5M": ((17.5, 35.5), (34.5, 53.5)),
    "shift_left_0p5M": ((17.5, 34.5), (34.5, 52.5)),
    "shift_right_0p5M": ((18.5, 35.5), (35.5, 53.5)),
}
RAY_CONSISTENCY_TOLERANCE_M = 1.0
D1_ANALYSIS_CADENCE_M = 0.001
LOCALIZED_SOURCE_INTERVAL = "common_archived_interval"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return path
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _strict_json(path: Path, value: dict | list) -> Path:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [clean(entry) for entry in item]
        if isinstance(item, tuple):
            return [clean(entry) for entry in item]
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if np.isfinite(item) else None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def _l2(values: np.ndarray, times: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(np.abs(values) ** 2, x=times)))


def _flat_signal(result: SdSSimulationResult) -> np.ndarray:
    index = int(np.argmin(np.abs(result.observer_rho - 1.0)))
    return np.asarray(result.signals[:, index], dtype=float)


def _retarded_times(result: SdSSimulationResult) -> np.ndarray:
    return result.signal_times - float(result.metadata["retarded_time_offset"]["q"])


def _align_flat(result: SdSSimulationResult, times: np.ndarray) -> np.ndarray:
    source_times = _retarded_times(result)
    if times[0] < source_times[0] or times[-1] > source_times[-1]:
        raise ValueError("Requested fixed window lies outside a waveform archive.")
    return np.interp(times, source_times, _flat_signal(result))


def _window_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (times >= start) & (times <= end)
    if np.count_nonzero(mask) < 16:
        raise ValueError(f"Fixed window [{start}, {end}] has too few samples.")
    return mask


def _window_family(name: str) -> str:
    return "cumulative" if name in {row[0] for row in CUMULATIVE_WINDOWS} else "disjoint"


def waveform_metrics(
    times: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    start: float,
    end: float,
) -> dict[str, float]:
    """Return norm and zero-translation complex-overlap diagnostics."""

    mask = _window_mask(times, start, end)
    local_times = times[mask]
    candidate = np.asarray(candidate[mask], dtype=float)
    reference = np.asarray(reference[mask], dtype=float)
    difference = candidate - reference
    reference_l2 = _l2(reference, local_times)
    reference_linf = float(np.max(np.abs(reference)))
    taper = tukey(reference.size, alpha=0.1)
    candidate_analytic = hilbert(candidate * taper)
    reference_analytic = hilbert(reference * taper)
    denominator = np.trapezoid(
        np.abs(reference_analytic) ** 2, x=local_times
    )
    overlap = np.trapezoid(
        candidate_analytic * np.conjugate(reference_analytic), x=local_times
    ) / denominator
    return {
        "reference_l2": reference_l2,
        "difference_l2": _l2(difference, local_times),
        "E2": _l2(difference, local_times) / reference_l2,
        "difference_linf_absolute": float(np.max(np.abs(difference))),
        "Einf": float(np.max(np.abs(difference))) / reference_linf,
        "overlap_amplitude_ratio": float(np.abs(overlap)),
        "amplitude_difference_fraction": abs(float(np.abs(overlap)) - 1.0),
        "phase_difference_radians": float(np.angle(overlap)),
        "time_translation_fitted": False,
    }


def _sphere_modal_squared_norm(
    times: np.ndarray,
    responses: np.ndarray,
    modal_power_by_ell: np.ndarray,
    start: float,
    end: float,
) -> float:
    mask = _window_mask(times, start, end)
    density = np.sum(
        np.asarray(responses[mask], dtype=float) ** 2
        * np.asarray(modal_power_by_ell, dtype=float)[None, :],
        axis=1,
    )
    return float(np.trapezoid(density, x=np.asarray(times[mask], dtype=float)))


def sphere_modal_metrics(
    times: np.ndarray,
    candidate_responses: np.ndarray,
    reference_responses: np.ndarray,
    modal_power_by_ell: np.ndarray,
    start: float,
    end: float,
) -> dict[str, float]:
    r"""Metrics for ``\int dU dOmega |u|^2`` using modal orthogonality.

    The compact source archives store one radial response per ``ell``.  The
    factor supplied in ``modal_power_by_ell`` is ``sum_m |A_lm|^2``.  Thus
    this evaluates the sphere-integrated norm without angular quadrature or
    construction of the much larger expanded modal array.
    """

    mask = _window_mask(times, start, end)
    local_times = np.asarray(times[mask], dtype=float)
    candidate = np.asarray(candidate_responses[mask], dtype=float)
    reference = np.asarray(reference_responses[mask], dtype=float)
    weights = np.asarray(modal_power_by_ell, dtype=float)
    difference = candidate - reference

    reference_squared = _sphere_modal_squared_norm(
        local_times, reference, weights, local_times[0], local_times[-1]
    )
    difference_squared = _sphere_modal_squared_norm(
        local_times, difference, weights, local_times[0], local_times[-1]
    )
    candidate_squared = _sphere_modal_squared_norm(
        local_times, candidate, weights, local_times[0], local_times[-1]
    )
    reference_instantaneous = np.sqrt(
        np.sum(reference**2 * weights[None, :], axis=1)
    )
    difference_instantaneous = np.sqrt(
        np.sum(difference**2 * weights[None, :], axis=1)
    )
    return {
        "sphere_integrated_reference_l2": float(np.sqrt(reference_squared)),
        "sphere_integrated_difference_l2": float(np.sqrt(difference_squared)),
        "E2": float(np.sqrt(difference_squared / reference_squared)),
        "Einf": float(
            np.max(difference_instantaneous) / np.max(reference_instantaneous)
        ),
        "sphere_integrated_norm_ratio": float(
            np.sqrt(candidate_squared / reference_squared)
        ),
        "time_translation_fitted": False,
        "primary_measure": "sphere_integrated_modal_norm_Parseval",
    }


def _effective_order(
    coarse_difference: float,
    fine_difference: float,
    coarse_resolution: int,
    medium_resolution: int,
    fine_resolution: int,
) -> float:
    if coarse_difference <= 0.0 or fine_difference <= 0.0:
        return np.nan
    ratio = coarse_difference / fine_difference
    hc, hm, hf = (
        1.0 / coarse_resolution,
        1.0 / medium_resolution,
        1.0 / fine_resolution,
    )

    def residual(order: float) -> float:
        return (hc**order - hm**order) / (hm**order - hf**order) - ratio

    try:
        return float(brentq(residual, 0.05, 16.0))
    except ValueError:
        return np.nan


def load_flat_archives(output_dir: Path) -> dict[str, dict[float | None, SdSSimulationResult]]:
    root = Path(output_dir) / "raw" / "flat"
    results: dict[str, dict[float | None, SdSSimulationResult]] = {}
    for level in LEVELS:
        level_results: dict[float | None, SdSSimulationResult] = {
            None: load_sds_result(root / level / "schwarzschild.npz")
        }
        for length in FLAT_LENGTHS:
            level_results[float(length)] = load_sds_result(
                root / level / f"sds_L{length}.npz"
            )
        results[level] = level_results
    return results


def validate_contracts(archives: Iterable, study: str) -> dict:
    contracts = [archive.metadata["physical_contract"] for archive in archives]
    fingerprints = {
        archive.metadata["simulation_provenance"]["physical_contract_sha256"]
        for archive in archives
    }
    if len(fingerprints) != 1 or any(contract != contracts[0] for contract in contracts):
        raise ValueError(f"The {study} archives do not share one physical contract.")
    commits = {
        archive.metadata["simulation_provenance"]["git_commit"]
        for archive in archives
    }
    dirty = [
        archive.metadata["simulation_provenance"]["case"]
        for archive in archives
        if archive.metadata["simulation_provenance"]["git_worktree_dirty"]
    ]
    if len(commits) != 1 or dirty:
        raise ValueError(
            f"The {study} simulation provenance is not one clean commit: "
            f"commits={commits}, dirty={dirty}"
        )
    return {
        "study": study,
        "physical_contract_sha256": next(iter(fingerprints)),
        "simulation_commit": next(iter(commits)),
        "archive_count": len(contracts),
        "contract": contracts[0],
    }


def flat_analysis(output_dir: Path) -> dict:
    archives = load_flat_archives(output_dir)
    contract = validate_contracts(
        [item for values in archives.values() for item in values.values()], "flat"
    )
    fine = archives["fine"]
    reference_result = fine[None]
    reference_times = _retarded_times(reference_result)
    reference_signal = _flat_signal(reference_result)
    common = (reference_times >= 0.0) & (reference_times <= 160.0)
    times = reference_times[common]
    reference = reference_signal[common]
    fine_signals = {
        length: _align_flat(fine[float(length)], times) for length in FLAT_LENGTHS
    }

    numerical_rows: list[dict] = []
    numerical_lookup: dict[tuple[int, str], float] = {}
    estimated_fine_lookup: dict[tuple[int, str], float] = {}
    resolutions = {"coarse": 384, "medium": 512, "fine": 768}
    for length in FLAT_LENGTHS:
        paired_residuals = {}
        for level in LEVELS:
            level_results = archives[level]
            level_candidate = _align_flat(level_results[float(length)], times)
            level_reference = _align_flat(level_results[None], times)
            paired_residuals[level] = level_candidate - level_reference
        for window, start, end in ANALYSIS_WINDOWS:
            mask = _window_mask(times, start, end)
            denominator = _l2(reference[mask], times[mask])
            coarse_medium = _l2(
                paired_residuals["coarse"][mask]
                - paired_residuals["medium"][mask],
                times[mask],
            ) / denominator
            medium_fine = _l2(
                paired_residuals["medium"][mask]
                - paired_residuals["fine"][mask],
                times[mask],
            ) / denominator
            order = _effective_order(
                coarse_medium,
                medium_fine,
                resolutions["coarse"],
                resolutions["medium"],
                resolutions["fine"],
            )
            ratio = resolutions["fine"] / resolutions["medium"]
            richardson = (
                medium_fine / (ratio**order - 1.0)
                if np.isfinite(order) and order > 0.0
                else np.nan
            )
            conservative = max(
                medium_fine,
                richardson if np.isfinite(richardson) else medium_fine,
            )
            numerical_lookup[(length, window)] = conservative
            estimated_fine = richardson if np.isfinite(richardson) else medium_fine
            estimated_fine_lookup[(length, window)] = estimated_fine
            numerical_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "coarse_medium_paired_E2": coarse_medium,
                    "medium_fine_paired_E2": medium_fine,
                    "observed_coupled_order": order,
                    "richardson_fine_E2": richardson,
                    "estimated_fine_numerical_E2": estimated_fine,
                    "conservative_numerical_E2": conservative,
                    "below_0p2_percent": conservative < 0.002,
                    "below_0p1_percent": conservative < 0.001,
                    "case_specific": True,
                }
            )

    metric_rows: list[dict] = []
    for length in FLAT_LENGTHS:
        for window, start, end in ANALYSIS_WINDOWS:
            metric_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    **waveform_metrics(
                        times, fine_signals[length], reference, start, end
                    ),
                    "case_specific_numerical_E2": numerical_lookup[(length, window)],
                    "analysis_status": (
                        "quantitative"
                        if numerical_lookup[(length, window)] < 0.002
                        else "diagnostic_numerical_refinement_not_subdominant"
                    ),
                }
            )

    successive_rows: list[dict] = []
    for lower, upper in zip(FLAT_LENGTHS, FLAT_LENGTHS[1:]):
        for window, start, end in ANALYSIS_WINDOWS:
            metrics = waveform_metrics(
                times, fine_signals[upper], fine_signals[lower], start, end
            )
            mask = _window_mask(times, start, end)
            schwarzschild_norm = _l2(reference[mask], times[mask])
            successive = _l2(
                fine_signals[upper][mask] - fine_signals[lower][mask],
                times[mask],
            ) / schwarzschild_norm
            successive_linf = float(
                np.max(np.abs(fine_signals[upper][mask] - fine_signals[lower][mask]))
                / np.max(np.abs(reference[mask]))
            )
            successive_rows.append(
                {
                    "lower_L_over_M": lower,
                    "upper_L_over_M": upper,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "successive_difference_E2_over_schwarzschild": successive,
                    "successive_difference_Einf_over_schwarzschild": successive_linf,
                    "upper_to_lower_overlap_amplitude_ratio": metrics[
                        "overlap_amplitude_ratio"
                    ],
                    "upper_to_lower_amplitude_difference_fraction": metrics[
                        "amplitude_difference_fraction"
                    ],
                    "upper_to_lower_phase_difference_radians": metrics[
                        "phase_difference_radians"
                    ],
                    "time_translation_fitted": False,
                }
            )

    extrapolants = {
        80: (fine_signals[80] - 6.0 * fine_signals[160] + 8.0 * fine_signals[320]) / 3.0,
        160: (fine_signals[160] - 6.0 * fine_signals[320] + 8.0 * fine_signals[640]) / 3.0,
    }
    medium_times = times
    medium_signals = {
        length: _align_flat(archives["medium"][float(length)], medium_times)
        for length in (80, 160, 320, 640)
    }
    medium_reference = _align_flat(archives["medium"][None], medium_times)
    medium_extrapolants = {
        80: (
            medium_signals[80]
            - 6.0 * medium_signals[160]
            + 8.0 * medium_signals[320]
        ) / 3.0,
        160: (
            medium_signals[160]
            - 6.0 * medium_signals[320]
            + 8.0 * medium_signals[640]
        ) / 3.0,
    }
    extrapolation_rows: list[dict] = []
    extrapolation_coefficients = {
        "Winf_L80_vs_Schwarzschild": {80: 1.0 / 3.0, 160: -2.0, 320: 8.0 / 3.0},
        "Winf_L160_vs_Schwarzschild": {160: 1.0 / 3.0, 320: -2.0, 640: 8.0 / 3.0},
        "Winf_L80_vs_Winf_L160": {
            80: 1.0 / 3.0,
            160: -7.0 / 3.0,
            320: 14.0 / 3.0,
            640: -8.0 / 3.0,
        },
    }
    comparisons = (
        ("Winf_L80_vs_Schwarzschild", extrapolants[80], reference),
        ("Winf_L160_vs_Schwarzschild", extrapolants[160], reference),
        ("Winf_L80_vs_Winf_L160", extrapolants[80], extrapolants[160]),
    )
    for comparison, candidate, target in comparisons:
        for window, start, end in ANALYSIS_WINDOWS:
            metrics = waveform_metrics(times, candidate, target, start, end)
            mask = _window_mask(times, start, end)
            denominator = _l2(reference[mask], times[mask])
            if comparison == "Winf_L80_vs_Schwarzschild":
                fine_residual = extrapolants[80] - reference
                medium_residual = medium_extrapolants[80] - medium_reference
            elif comparison == "Winf_L160_vs_Schwarzschild":
                fine_residual = extrapolants[160] - reference
                medium_residual = medium_extrapolants[160] - medium_reference
            else:
                fine_residual = extrapolants[80] - extrapolants[160]
                medium_residual = medium_extrapolants[80] - medium_extrapolants[160]
            numerical = _l2(
                fine_residual[mask] - medium_residual[mask], times[mask]
            ) / denominator
            propagated_fine_error = sum(
                abs(coefficient) * estimated_fine_lookup[(length, window)]
                for length, coefficient in extrapolation_coefficients[comparison].items()
            )
            extrapolation_rows.append(
                {
                    "comparison": comparison,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    **metrics,
                    "paired_medium_fine_numerical_E2": numerical,
                    "propagated_estimated_fine_numerical_E2": propagated_fine_error,
                    "directly_observed_medium_fine_change_E2": numerical,
                    "central_residual_is_resolved_accuracy": False,
                    "central_residual_interpretation": (
                        "central extrapolant residual; not a resolved accuracy estimate"
                    ),
                    "agreement_within_1_percent": metrics["E2"] <= 0.01,
                    "numerical_below_0p2_percent": propagated_fine_error < 0.002,
                    "regulator_success": metrics["E2"] <= 0.01
                    and propagated_fine_error < 0.002,
                    "analysis_status": (
                        "quantitative"
                        if propagated_fine_error < 0.002
                        else "diagnostic_numerical_refinement_not_subdominant"
                    ),
                }
            )

    threshold_rows: list[dict] = []
    for window, _, _ in ANALYSIS_WINDOWS:
        selected = [row for row in metric_rows if row["window"] == window]
        for threshold in (0.05, 0.02, 0.01):
            qualified = [
                row
                for row in selected
                if row["E2"] + row["case_specific_numerical_E2"] <= threshold
            ]
            threshold_rows.append(
                {
                    "waveform_family": "pure_ell2_fixed_data",
                    "threshold_scope": "pure_ell2_direct_error_only",
                    "window": window,
                    "window_family": _window_family(window),
                    "threshold_fraction": threshold,
                    "smallest_L_over_M": (
                        min(row["cosmological_length_over_M"] for row in qualified)
                        if qualified
                        else ""
                    ),
                    "status": (
                        "attained_with_numerical_margin"
                        if qualified
                        else "not_attained_through_L640"
                    ),
                    "criterion": "E2 + conservative numerical E2 <= threshold",
                }
            )

    aligned_columns = [times, reference]
    aligned_headers = ["U_over_M", "schwarzschild"]
    for length in FLAT_LENGTHS:
        aligned_columns.append(fine_signals[length])
        aligned_headers.append(f"sds_L{length}")
    aligned_columns.extend((extrapolants[80], extrapolants[160]))
    aligned_headers.extend(("Winf_from_L80", "Winf_from_L160"))
    aligned_path = Path(output_dir) / "tables" / "flat_aligned_waveforms.csv"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    with aligned_path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(aligned_headers)
        writer.writerows(np.column_stack(aligned_columns).tolist())

    return {
        "contract": contract,
        "times": times,
        "reference": reference,
        "fine_signals": fine_signals,
        "extrapolants": extrapolants,
        "metrics": metric_rows,
        "numerical": numerical_rows,
        "successive": successive_rows,
        "extrapolation": extrapolation_rows,
        "thresholds": threshold_rows,
    }


def load_source_archives(
    output_dir: Path,
) -> dict[str, dict[float | None, SourcedSimulationResult]]:
    root = Path(output_dir) / "raw" / "source"
    results: dict[str, dict[float | None, SourcedSimulationResult]] = {}
    for level in LEVELS:
        level_results: dict[float | None, SourcedSimulationResult] = {
            None: load_sourced_result(root / level / "schwarzschild.npz")
        }
        for length in SOURCE_LENGTHS:
            level_results[float(length)] = load_sourced_result(
                root / level / f"sds_L{length}.npz"
            )
        results[level] = level_results
    return results


def _source_modal_power_by_ell(
    result: SourcedSimulationResult, maximum_ell: int
) -> np.ndarray:
    weights = np.zeros(maximum_ell + 1, dtype=float)
    for ell in range(maximum_ell + 1):
        weights[ell] = float(
            np.sum(result.mode_source_amplitude[result.mode_ell == ell] ** 2)
        )
    return weights


def _align_source_responses(
    result: SourcedSimulationResult,
    times: np.ndarray,
    maximum_ell: int,
) -> np.ndarray:
    """Align compact outer-boundary responses onto one fixed ``U`` grid."""

    aligned = np.zeros((times.size, maximum_ell + 1), dtype=np.float32)
    observer = result.outer_index()
    for source_index, ell in enumerate(result.response_ell):
        ell = int(ell)
        if ell <= maximum_ell:
            aligned[:, ell] = np.interp(
                times,
                result.retarded_time,
                result.response_signals[:, observer, source_index],
            )
    return aligned


def _source_direction_coefficients(
    result: SourcedSimulationResult, maximum_ell: int, phi: float
) -> np.ndarray:
    basis = harmonic_matrix(
        result,
        np.asarray([0.5 * np.pi]),
        np.asarray([phi]),
    )[:, 0]
    coefficients = np.zeros(maximum_ell + 1, dtype=float)
    for ell in range(maximum_ell + 1):
        selected = result.mode_ell == ell
        coefficients[ell] = float(
            np.sum(result.mode_source_amplitude[selected] * basis[selected])
        )
    return coefficients


def localized_waveform_analysis(
    archives: dict[str, dict[float | None, SourcedSimulationResult]],
) -> dict:
    """Full fixed-source waveform extrapolation on existing raw archives."""

    fine = archives["fine"]
    reference_result = fine[None]
    maximum_ell = int(np.max(reference_result.response_ell))
    common_end = min(
        float(result.retarded_time[-1])
        for level in LEVELS
        for result in archives[level].values()
    )
    reference_times = reference_result.retarded_time
    inside = (reference_times >= 0.0) & (reference_times <= common_end)
    times = reference_times[inside]
    start, end = float(times[0]), float(times[-1])
    weights = _source_modal_power_by_ell(reference_result, maximum_ell)
    fine_responses = {
        length: _align_source_responses(result, times, maximum_ell)
        for length, result in fine.items()
    }
    reference = fine_responses[None]
    reference_squared = _sphere_modal_squared_norm(
        times, reference, weights, start, end
    )

    numerical_rows: list[dict] = []
    numerical_lookup: dict[int, float] = {}
    estimated_fine_lookup: dict[int, float] = {}
    resolution = {"coarse": 1024, "medium": 1536, "fine": 2048}
    for length in SOURCE_LENGTHS:
        paired_residuals = {}
        for level in LEVELS:
            candidate = _align_source_responses(
                archives[level][float(length)], times, maximum_ell
            )
            level_reference = _align_source_responses(
                archives[level][None], times, maximum_ell
            )
            paired_residuals[level] = candidate - level_reference
        coarse_medium = np.sqrt(
            _sphere_modal_squared_norm(
                times,
                paired_residuals["coarse"] - paired_residuals["medium"],
                weights,
                start,
                end,
            )
            / reference_squared
        )
        medium_fine = np.sqrt(
            _sphere_modal_squared_norm(
                times,
                paired_residuals["medium"] - paired_residuals["fine"],
                weights,
                start,
                end,
            )
            / reference_squared
        )
        order = _effective_order(
            coarse_medium,
            medium_fine,
            resolution["coarse"],
            resolution["medium"],
            resolution["fine"],
        )
        richardson = (
            medium_fine / ((2048.0 / 1536.0) ** order - 1.0)
            if np.isfinite(order) and order > 0.0
            else np.nan
        )
        estimated = richardson if np.isfinite(richardson) else medium_fine
        conservative = max(medium_fine, estimated)
        numerical_lookup[length] = conservative
        estimated_fine_lookup[length] = estimated
        numerical_rows.append(
            {
                "cosmological_length_over_M": length,
                "observer": "outer",
                "window": LOCALIZED_SOURCE_INTERVAL,
                "interval_classification": "common archived interval across all source archives",
                "complete_late_time_signal": False,
                "window_start_U_over_M": start,
                "window_end_U_over_M": end,
                "coarse_medium_paired_sphere_E2": coarse_medium,
                "medium_fine_paired_sphere_E2": medium_fine,
                "directly_observed_refinement_change_E2": medium_fine,
                "observed_coupled_order": order,
                "richardson_fine_sphere_E2": richardson,
                "estimated_fine_numerical_E2": estimated,
                "conservative_numerical_E2": conservative,
                "case_specific": True,
            }
        )

    metric_rows = []
    for length in SOURCE_LENGTHS:
        metric_rows.append(
            {
                "cosmological_length_over_M": length,
                "observer": "outer",
                "window": LOCALIZED_SOURCE_INTERVAL,
                "interval_classification": "common archived interval across all source archives",
                "complete_late_time_signal": False,
                "window_start_U_over_M": start,
                "window_end_U_over_M": end,
                **sphere_modal_metrics(
                    times,
                    fine_responses[float(length)],
                    reference,
                    weights,
                    start,
                    end,
                ),
                "case_specific_numerical_E2": numerical_lookup[length],
                "analysis_status": "quantitative_case_specific_refinement",
            }
        )

    extrapolants = {
        80: (
            fine_responses[80.0]
            - 6.0 * fine_responses[160.0]
            + 8.0 * fine_responses[320.0]
        )
        / 3.0,
        160: (
            fine_responses[160.0]
            - 6.0 * fine_responses[320.0]
            + 8.0 * fine_responses[640.0]
        )
        / 3.0,
    }
    medium_responses = {
        length: _align_source_responses(
            archives["medium"][length], times, maximum_ell
        )
        for length in (None, 80.0, 160.0, 320.0, 640.0)
    }
    medium_extrapolants = {
        80: (
            medium_responses[80.0]
            - 6.0 * medium_responses[160.0]
            + 8.0 * medium_responses[320.0]
        )
        / 3.0,
        160: (
            medium_responses[160.0]
            - 6.0 * medium_responses[320.0]
            + 8.0 * medium_responses[640.0]
        )
        / 3.0,
    }
    coefficients = {
        "Winf_L80_vs_Schwarzschild": {80: 1.0 / 3.0, 160: -2.0, 320: 8.0 / 3.0},
        "Winf_L160_vs_Schwarzschild": {160: 1.0 / 3.0, 320: -2.0, 640: 8.0 / 3.0},
        "Winf_L80_vs_Winf_L160": {
            80: 1.0 / 3.0,
            160: -7.0 / 3.0,
            320: 14.0 / 3.0,
            640: -8.0 / 3.0,
        },
    }
    comparisons = (
        (
            "Winf_L80_vs_Schwarzschild",
            extrapolants[80],
            reference,
            medium_extrapolants[80],
            medium_responses[None],
        ),
        (
            "Winf_L160_vs_Schwarzschild",
            extrapolants[160],
            reference,
            medium_extrapolants[160],
            medium_responses[None],
        ),
        (
            "Winf_L80_vs_Winf_L160",
            extrapolants[80],
            extrapolants[160],
            medium_extrapolants[80],
            medium_extrapolants[160],
        ),
    )
    extrapolation_rows = []
    for comparison, candidate, target, medium_candidate, medium_target in comparisons:
        observed_refinement = np.sqrt(
            _sphere_modal_squared_norm(
                times,
                (candidate - target) - (medium_candidate - medium_target),
                weights,
                start,
                end,
            )
            / reference_squared
        )
        propagated = sum(
            abs(coefficient) * estimated_fine_lookup[length]
            for length, coefficient in coefficients[comparison].items()
        )
        extrapolation_rows.append(
            {
                "comparison": comparison,
                "observer": "outer",
                "window": LOCALIZED_SOURCE_INTERVAL,
                "interval_classification": "common archived interval across all source archives",
                "complete_late_time_signal": False,
                "window_start_U_over_M": start,
                "window_end_U_over_M": end,
                **sphere_modal_metrics(
                    times, candidate, target, weights, start, end
                ),
                "directly_observed_medium_fine_change_E2": observed_refinement,
                "propagated_estimated_fine_numerical_E2": propagated,
                "central_residual_is_resolved_accuracy": False,
                "central_residual_interpretation": (
                    "central extrapolant residual; not a resolved accuracy estimate"
                ),
                "agreement_within_1_percent": bool(
                    sphere_modal_metrics(
                        times, candidate, target, weights, start, end
                    )["E2"]
                    <= 0.01
                ),
            }
        )

    direction_rows = []
    direction_plot = {}
    for phi, label in ((0.0, "gamma_0"), (np.pi, "gamma_pi")):
        direction_weights = _source_direction_coefficients(
            reference_result, maximum_ell, phi
        )
        reference_trace = reference @ direction_weights
        traces = {
            length: responses @ direction_weights
            for length, responses in fine_responses.items()
        }
        extrapolated_traces = {
            length: response @ direction_weights
            for length, response in extrapolants.items()
        }
        for length in SOURCE_LENGTHS:
            direction_rows.append(
                {
                    "diagnostic": "direct",
                    "direction": label,
                    "window": LOCALIZED_SOURCE_INTERVAL,
                    "complete_late_time_signal": False,
                    "gamma_over_pi": phi / np.pi,
                    "cosmological_length_over_M": length,
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    **waveform_metrics(
                        times,
                        traces[float(length)],
                        reference_trace,
                        start,
                        end,
                    ),
                    "analysis_status": "secondary_direction_diagnostic",
                }
            )
        for length in (80, 160):
            direction_rows.append(
                {
                    "diagnostic": "extrapolant_vs_schwarzschild",
                    "direction": label,
                    "window": LOCALIZED_SOURCE_INTERVAL,
                    "complete_late_time_signal": False,
                    "gamma_over_pi": phi / np.pi,
                    "base_L_over_M": length,
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    **waveform_metrics(
                        times,
                        extrapolated_traces[length],
                        reference_trace,
                        start,
                        end,
                    ),
                    "analysis_status": "secondary_direction_diagnostic",
                }
            )
        direction_rows.append(
            {
                "diagnostic": "extrapolant_difference",
                "direction": label,
                "window": LOCALIZED_SOURCE_INTERVAL,
                "complete_late_time_signal": False,
                "gamma_over_pi": phi / np.pi,
                "window_start_U_over_M": start,
                "window_end_U_over_M": end,
                **waveform_metrics(
                    times,
                    extrapolated_traces[80],
                    extrapolated_traces[160],
                    start,
                    end,
                ),
                "analysis_status": "secondary_direction_diagnostic",
            }
        )
        direction_plot[label] = {
            "reference": reference_trace,
            "L320": traces[320.0],
            "L640": traces[640.0],
            "Winf80": extrapolated_traces[80],
            "Winf160": extrapolated_traces[160],
        }

    instantaneous_reference = np.sqrt(
        np.sum(reference.astype(float) ** 2 * weights[None, :], axis=1)
    )
    sphere_plot = {
        "reference": instantaneous_reference,
        "L320_residual": np.sqrt(
            np.sum(
                (fine_responses[320.0] - reference).astype(float) ** 2
                * weights[None, :],
                axis=1,
            )
        ),
        "L640_residual": np.sqrt(
            np.sum(
                (fine_responses[640.0] - reference).astype(float) ** 2
                * weights[None, :],
                axis=1,
            )
        ),
        "extrapolant_difference": np.sqrt(
            np.sum(
                (extrapolants[80] - extrapolants[160]).astype(float) ** 2
                * weights[None, :],
                axis=1,
            )
        ),
    }
    return {
        "times": times,
        "window": (start, end),
        "metrics": metric_rows,
        "numerical": numerical_rows,
        "extrapolation": extrapolation_rows,
        "directions": direction_rows,
        "sphere_plot": sphere_plot,
        "direction_plot": direction_plot,
    }


def _d1_variant_rows(
    sds: SourcedSimulationResult,
    schwarzschild: SourcedSimulationResult,
    windows: tuple[tuple[float, float], tuple[float, float]],
    analysis_cadence: float = D1_ANALYSIS_CADENCE_M,
) -> list[dict]:
    """Measure correlated D1 values on one resolution-independent U grid."""

    analysis_times = np.arange(
        min(window[0] for window in windows) - 0.5,
        max(window[1] for window in windows) + 0.5 + 0.5 * analysis_cadence,
        analysis_cadence,
    )
    rows: list[dict] = []
    for observer in range(sds.observer_areal_radius.size):
        arrivals = {}
        for background, result in (("sds", sds), ("schwarzschild", schwarzschild)):
            measured = []
            for pulse, (window, phi) in enumerate(zip(windows, (0.0, np.pi))):
                times, trace = direction_waveform(result, phi, observer)
                reference_times, reference_trace = direction_waveform(
                    schwarzschild, phi, observer
                )
                estimate = estimate_pulse(
                    pulse=pulse,
                    phi=phi,
                    times=analysis_times,
                    trace=np.interp(analysis_times, times, trace),
                    reference_times=analysis_times,
                    reference_trace=np.interp(
                        analysis_times, reference_times, reference_trace
                    ),
                    window=window,
                )
                measured.append(estimate)
            arrivals[background] = measured
        analytic = (
            arrivals["sds"][1].analytic_time
            - arrivals["sds"][0].analytic_time
            - arrivals["schwarzschild"][1].analytic_time
            + arrivals["schwarzschild"][0].analytic_time
        )
        matched = (
            arrivals["sds"][1].matched_time
            - arrivals["sds"][0].matched_time
            - arrivals["schwarzschild"][1].matched_time
            + arrivals["schwarzschild"][0].matched_time
        )
        rows.append(
            {
                "observer_index": observer,
                "observer_radius_over_M": float(sds.observer_areal_radius[observer]),
                # The matched-template lag is the primary arrival estimator:
                # unlike the local envelope maximum it is stable across the
                # three PDE refinements.  The full difference between the two
                # estimators remains a separately reported timing systematic.
                "primary_estimator": "matched_template_lag",
                "analysis_cadence_over_M": analysis_cadence,
                "D1_over_M": float(matched),
                "analytic_envelope_D1_over_M": float(analytic),
                "matched_template_D1_over_M": float(matched),
                "estimator_sensitivity_over_M": abs(float(analytic - matched)),
                "matched_template_arrivals_resolved": all(
                    estimate.matched_time_resolved
                    for estimates in arrivals.values()
                    for estimate in estimates
                ),
            }
        )
    return rows


def _d1_window_variant(
    sds: SourcedSimulationResult,
    schwarzschild: SourcedSimulationResult,
    windows: tuple[tuple[float, float], tuple[float, float]],
    analysis_cadence: float = D1_ANALYSIS_CADENCE_M,
) -> list[float]:
    return [
        row["D1_over_M"]
        for row in _d1_variant_rows(
            sds, schwarzschild, windows, analysis_cadence
        )
    ]


def _observer_label(result: SourcedSimulationResult, index: int) -> str:
    if index == result.outer_index():
        return "outer"
    return f"r{result.observer_areal_radius[index]:g}M"


def source_analysis(output_dir: Path, repository_root: Path | None = None) -> dict:
    archives = load_source_archives(output_dir)
    contract = validate_contracts(
        [item for values in archives.values() for item in values.values()], "source"
    )
    waveform = localized_waveform_analysis(archives)
    source_width_lookup = _source_width_sensitivity_lookup(
        Path(repository_root or Path.cwd())
    )
    measurements: dict[tuple[int, str], list[dict]] = {}
    for level in LEVELS:
        reference = archives[level][None]
        for length in SOURCE_LENGTHS:
            measurements[(length, level)] = _d1_variant_rows(
                archives[level][float(length)],
                reference,
                WINDOW_VARIANTS["primary"],
            )

    measurement_rows: list[dict] = []
    numerical_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    fit_rows_by_key: dict[tuple[int, str], dict] = {}
    for length in SOURCE_LENGTHS:
        fine_result = archives["fine"][float(length)]
        window_values = {
            name: _d1_window_variant(
                fine_result, archives["fine"][None], windows
            )
            for name, windows in WINDOW_VARIANTS.items()
        }
        cadence_values = {
            cadence: _d1_window_variant(
                fine_result,
                archives["fine"][None],
                WINDOW_VARIANTS["primary"],
                cadence,
            )
            for cadence in (0.0005, 0.002)
        }
        for observer in range(fine_result.observer_areal_radius.size):
            label = _observer_label(fine_result, observer)
            level_rows = {
                level: measurements[(length, level)][observer] for level in LEVELS
            }
            primary = level_rows["fine"]
            coarse_medium = abs(
                level_rows["coarse"]["D1_over_M"]
                - level_rows["medium"]["D1_over_M"]
            )
            medium_fine = abs(
                level_rows["medium"]["D1_over_M"]
                - level_rows["fine"]["D1_over_M"]
            )
            order = _effective_order(coarse_medium, medium_fine, 1024, 1536, 2048)
            richardson = (
                medium_fine / ((2048.0 / 1536.0) ** order - 1.0)
                if np.isfinite(order) and order > 0.0
                else np.nan
            )
            numerical = max(
                medium_fine,
                richardson if np.isfinite(richardson) else medium_fine,
            )
            primary_window = window_values["primary"][observer]
            window_sensitivity = max(
                abs(values[observer] - primary_window)
                for name, values in window_values.items()
                if name != "primary"
            )
            estimator = primary["estimator_sensitivity_over_M"]
            cadence_sensitivity = max(
                abs(values[observer] - primary_window)
                for values in cadence_values.values()
            )
            combined = float(
                numerical + estimator + window_sensitivity + cadence_sensitivity
            )
            source_width_sensitivity = source_width_lookup.get((length, label), np.nan)
            signal_to_uncertainty = (
                abs(primary["D1_over_M"]) / combined if combined > 0.0 else np.inf
            )
            resolved = bool(
                primary["matched_template_arrivals_resolved"]
                and signal_to_uncertainty >= 3.0
            )
            local = label != "outer"
            numerical_target = 1e-4 if length == 320 and local else 5e-4
            target_applies = (length == 320 and local) or (
                length in (320, 640) and not local
            )
            diagnostic = bool(not resolved)
            measurement_row = {
                "cosmological_length_over_M": length,
                "observer": label,
                **primary,
                "discretization_D1_uncertainty_over_M": numerical,
                "estimator_sensitivity_over_M": estimator,
                "window_sensitivity_over_M": window_sensitivity,
                "cadence_sensitivity_over_M": cadence_sensitivity,
                "source_width_sensitivity_over_M": source_width_sensitivity,
                "source_width_sensitivity_available": bool(
                    np.isfinite(source_width_sensitivity)
                ),
                "source_width_sensitivity_classification": (
                    "physical_source_dependence_not_numerical_error"
                ),
                "combined_fixed_source_timing_sensitivity_over_M": combined,
                "combined_timing_uncertainty_over_M": combined,
                "combination_rule": (
                    "linear sum of absolute discretization, estimator, window, "
                    "and cadence sensitivities; deterministic bound, not a statistical error"
                ),
                "deterministic_sensitivities_are_statistical_errors": False,
                "signal_to_combined_uncertainty": signal_to_uncertainty,
                "source_width_dependence_included_as_error": False,
                "genuinely_resolved": resolved,
                "genuinely_resolved_criterion": (
                    "all matched-template arrivals interior and "
                    "abs(D1)/combined fixed-source deterministic sensitivity >= 3"
                ),
                "diagnostic_only": diagnostic,
                "target_applies": target_applies,
                "target_uncertainty_over_M": numerical_target if target_applies else np.nan,
                "meets_target": combined < numerical_target if target_applies else np.nan,
                "meets_combined_timing_sensitivity_target": (
                    combined < numerical_target if target_applies else np.nan
                ),
                "target_test_basis": (
                    "combined fixed-source deterministic timing sensitivity, not PDE discretization alone"
                ),
                "case_specific_error": True,
            }
            measurement_rows.append(measurement_row)
            fit_rows_by_key[(length, label)] = measurement_row
            numerical_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "observer": label,
                    "coarse_D1_over_M": level_rows["coarse"]["D1_over_M"],
                    "medium_D1_over_M": level_rows["medium"]["D1_over_M"],
                    "fine_D1_over_M": level_rows["fine"]["D1_over_M"],
                    "coarse_medium_difference_over_M": coarse_medium,
                    "medium_fine_difference_over_M": medium_fine,
                    "observed_coupled_order": order,
                    "richardson_fine_error_over_M": richardson,
                    "conservative_discretization_error_over_M": numerical,
                }
            )
            for name, values in window_values.items():
                sensitivity_rows.append(
                    {
                        "cosmological_length_over_M": length,
                        "observer": label,
                        "category": "window",
                        "setting": name,
                        "D1_over_M": values[observer],
                        "difference_from_primary_over_M": abs(
                            values[observer] - primary_window
                        ),
                    }
                )
            sensitivity_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "observer": label,
                    "category": "estimator",
                    "setting": "matched_template_primary",
                    "D1_over_M": primary["matched_template_D1_over_M"],
                    "difference_from_primary_over_M": 0.0,
                }
            )
            sensitivity_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "observer": label,
                    "category": "estimator",
                    "setting": "analytic_envelope_alternate",
                    "D1_over_M": primary["analytic_envelope_D1_over_M"],
                    "difference_from_primary_over_M": estimator,
                }
            )
            for cadence, values in cadence_values.items():
                sensitivity_rows.append(
                    {
                        "cosmological_length_over_M": length,
                        "observer": label,
                        "category": "cadence",
                        "setting": f"analysis_cadence_{cadence:g}M",
                        "D1_over_M": values[observer],
                        "difference_from_primary_over_M": abs(
                            values[observer] - primary_window
                        ),
                    }
                )
            if np.isfinite(source_width_sensitivity):
                sensitivity_rows.append(
                    {
                        "cosmological_length_over_M": length,
                        "observer": label,
                        "category": "source_width",
                        "setting": "historical_fixed_background_width_variation",
                        "D1_over_M": np.nan,
                        "difference_from_primary_over_M": source_width_sensitivity,
                        "classification": "physical dependence; excluded from fixed-source timing target",
                    }
                )

    fit_rows: list[dict] = []
    for observer in ("r8M", "r12M", "outer"):
        selected = [
            fit_rows_by_key[(length, observer)] for length in SOURCE_LENGTHS
        ]
        eligible = [
            row
            for row in selected
            if row["case_specific_error"] and not row["diagnostic_only"]
        ]
        lengths = np.asarray(
            [row["cosmological_length_over_M"] for row in eligible], dtype=float
        )
        values = np.asarray([row["D1_over_M"] for row in eligible], dtype=float)
        sigma = np.asarray(
            [row["combined_fixed_source_timing_sensitivity_over_M"] for row in eligible],
            dtype=float,
        )
        sigma = np.maximum(sigma, np.finfo(float).eps)
        x = 1.0 / lengths
        if observer == "outer":
            design = np.column_stack((x, x**2))
            model = "b1_M_over_L_plus_b2_M2_over_L2"
            names = ("b1", "b2")
        else:
            design = np.column_stack((x**2, x**4))
            model = "a2_M2_over_L2_plus_a4_M4_over_L4"
            names = ("a2", "a4")
        weighted_design = design / sigma[:, None]
        weighted_values = values / sigma
        coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        residual = values - design @ coefficients
        scaled_residual_sum = float(np.sum((residual / sigma) ** 2))
        fit_rows.append(
            {
                "observer": observer,
                "model": model,
                "fit_status": "consistency_evidence_only",
                "scientific_interpretation": (
                    "scaling consistency evidence; not a precision coefficient measurement "
                    "or independent regulator claim"
                ),
                "input_lengths_over_M": ";".join(f"{value:g}" for value in lengths),
                "excluded_diagnostic_lengths_over_M": ";".join(
                    f"{row['cosmological_length_over_M']:g}"
                    for row in selected
                    if row["diagnostic_only"]
                ),
                names[0]: float(coefficients[0]),
                names[1]: float(coefficients[1]),
                "rms_residual_over_M": float(np.sqrt(np.mean(residual**2))),
                "scaled_residual_sum_squares": scaled_residual_sum,
                "chi_squared_interpretation": False,
                "degrees_of_freedom": int(len(values) - len(coefficients)),
                "fit_weights": (
                    "inverse squared case-specific deterministic sensitivity scale; "
                    "not inverse statistical variance"
                ),
            }
        )

    return {
        "contract": contract,
        "archives": archives,
        "measurements": measurement_rows,
        "numerical": numerical_rows,
        "sensitivity": sensitivity_rows,
        "fits": fit_rows,
        "waveform": waveform,
    }


def l12_phase_cleanup(repository_root: Path) -> list[dict]:
    tables = Path(repository_root) / "results" / "caustic_production_v2" / "tables"
    with (tables / "production_generic_angles.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        pulses = [row for row in csv.DictReader(stream) if row["case"] == "sds_L12"]
    with (tables / "production_generic_phase.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        phases = [row for row in csv.DictReader(stream) if row["case"] == "sds_L12"]
    pulse_lookup = {
        (float(row["gamma_over_pi"]), int(row["pulse"])): row for row in pulses
    }
    rows: list[dict] = []
    for phase in phases:
        gamma = float(phase["gamma_over_pi"])
        first, second = (int(value) for value in phase["pulse_pair"].split("->"))
        first_residual = abs(
            float(pulse_lookup[(gamma, first)]["simulation_minus_ray_over_M"])
        )
        second_residual = abs(
            float(pulse_lookup[(gamma, second)]["simulation_minus_ray_over_M"])
        )
        consistent = max(first_residual, second_residual) <= RAY_CONSISTENCY_TOLERANCE_M
        rows.append(
            {
                **phase,
                "first_pulse_abs_null_ray_residual_over_M": first_residual,
                "second_pulse_abs_null_ray_residual_over_M": second_residual,
                "null_ray_consistency_tolerance_over_M": RAY_CONSISTENCY_TOLERANCE_M,
                "pulse_arrivals_consistent": consistent,
                "included_in_quantitative_phase_analysis": False,
                "corrected_phase_radians": np.nan,
                "exclusion_reason": (
                    "L12_extracted_pulse_inconsistent_with_null_ray_arrival"
                    if not consistent
                    else "L12_phase_retained_as_diagnostic_only"
                ),
            }
        )
    return rows


def _source_width_classification(repository_root: Path) -> list[dict]:
    path = (
        Path(repository_root)
        / "results"
        / "caustic_production_v2"
        / "tables"
        / "source_width_delay_sensitivity.csv"
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            **row,
            "classification": "physical_source_width_dependence",
            "included_in_numerical_uncertainty": False,
            "included_in_fixed_source_regulator_comparison": False,
        }
        for row in rows
    ]


def _source_width_sensitivity_lookup(
    repository_root: Path,
) -> dict[tuple[int, str], float]:
    """Historical width variation, retained only as physical dependence."""

    rows = _source_width_classification(repository_root)
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        length = int(float(row["cosmological_length_over_M"]))
        observer = str(row["observer"])
        grouped.setdefault((length, observer), []).append(
            abs(float(row["difference_from_narrow_over_M"]))
        )
    return {key: max(values) for key, values in grouped.items()}


def _save_publication_figure(
    figure: plt.Figure, output_dir: Path, stem: str
) -> list[Path]:
    png = Path(output_dir) / f"{stem}.png"
    pdf = Path(output_dir) / f"{stem}.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: str,
    header: tuple[str, ...],
    rows: list[tuple[str, ...]],
    wide: bool = False,
) -> Path:
    environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{environment}}}",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend((r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_paper_tables(output_dir: Path, flat: dict, source: dict) -> list[Path]:
    tables = Path(output_dir) / "tables"
    flat_numerical = {
        (row["cosmological_length_over_M"], row["window"]): row
        for row in flat["numerical"]
    }
    direct_rows = []
    for row in flat["metrics"]:
        if row["cosmological_length_over_M"] not in (320, 640):
            continue
        direct_rows.append(
            (
                f"{row['cosmological_length_over_M']:g}",
                WINDOW_LATEX_LABELS[row["window"]],
                f"{100.0 * row['E2']:.4f}",
                f"{100.0 * row['Einf']:.4f}",
                f"{100.0 * flat_numerical[(row['cosmological_length_over_M'], row['window'])]['medium_fine_paired_E2']:.4f}",
                (
                    "quantitative"
                    if row["analysis_status"] == "quantitative"
                    else "diagnostic"
                ),
            )
        )
    flat_table = _write_latex_table(
        tables / "paper_flat_windows.tex",
        caption=(
            r"Direct regulator errors for the pure $\ell=2$ fixed-data sequence. "
            "These thresholds do not apply to the localized-source calculation."
        ),
        label="tab:flat-regulator-windows",
        columns="rlrrrl",
        header=(
            r"$L/M$",
            r"interval",
            r"$E_2$ (\%)",
            r"$E_\infty$ (\%)",
            r"observed $N_{\rm med}-N_{\rm fine}$ (\%)",
            "status",
        ),
        rows=direct_rows,
        wide=True,
    )

    extrapolation_rows = [
        (
            COMPARISON_LATEX_LABELS[row["comparison"]],
            WINDOW_LATEX_LABELS[row["window"]],
            f"{100.0 * row['E2']:.4f}",
            f"{100.0 * row['directly_observed_medium_fine_change_E2']:.4f}",
            f"{100.0 * row['propagated_estimated_fine_numerical_E2']:.4f}",
            r"1\% test" if row["regulator_success"] else "diagnostic",
        )
        for row in flat["extrapolation"]
    ]
    extrapolation_table = _write_latex_table(
        tables / "paper_flat_extrapolants.tex",
        caption=(
            r"Nested extrapolants for the pure $\ell=2$ fixed-data sequence. "
            "The central residual, directly observed medium-to-fine change, and "
            "propagated Richardson estimate are distinct quantities."
        ),
        label="tab:flat-extrapolants",
        columns="llrrrl",
        header=(
            "comparison",
            "interval",
            r"central residual (\%)",
            r"observed refinement (\%)",
            r"Richardson estimate (\%)",
            "status",
        ),
        rows=extrapolation_rows,
        wide=True,
    )

    localized = source["waveform"]
    localized_start, localized_end = localized["window"]
    localized_numerical = {
        row["cosmological_length_over_M"]: row for row in localized["numerical"]
    }
    localized_rows = [
        (
            r"$W_L-W_{\rm Schw}$",
            f"{row['cosmological_length_over_M']:g}",
            f"{100.0 * row['E2']:.5f}",
            f"{100.0 * localized_numerical[row['cosmological_length_over_M']]['medium_fine_paired_sphere_E2']:.5f}",
        )
        for row in localized["metrics"]
    ]
    localized_rows.extend(
        (
            COMPARISON_LATEX_LABELS[row["comparison"]],
            "--",
            f"{100.0 * row['E2']:.5f}",
            f"{100.0 * row['directly_observed_medium_fine_change_E2']:.5f}",
        )
        for row in localized["extrapolation"]
    )
    localized_table = _write_latex_table(
        tables / "paper_localized_source.tex",
        caption=(
            "Localized-source comparison on the common archived interval "
            rf"${localized_start:.6g}\leq U/M\leq {localized_end:.4f}$, "
            "using the sphere-integrated modal norm. "
            "The interval is not a complete late-time signal."
        ),
        label="tab:localized-source-regulator",
        columns="llrr",
        header=("comparison", r"$L/M$", r"$E_2$ (\%)", r"observed refinement (\%)"),
        rows=localized_rows,
    )

    target_rows = []
    for row in source["measurements"]:
        if not row["target_applies"]:
            continue
        target_rows.append(
            (
                f"{row['cosmological_length_over_M']:g}",
                row["observer"],
                f"{row['discretization_D1_uncertainty_over_M']:.2e}",
                f"{row['estimator_sensitivity_over_M']:.2e}",
                f"{row['window_sensitivity_over_M']:.2e}",
                f"{row['cadence_sensitivity_over_M']:.2e}",
                f"{row['combined_fixed_source_timing_sensitivity_over_M']:.2e}",
                f"{row['target_uncertainty_over_M']:.1e}",
                "yes" if row["meets_target"] else "no",
            )
        )
    timing_table = _write_latex_table(
        tables / "paper_timing_sensitivities.tex",
        caption=(
            "Deterministic timing sensitivities in units of M. The combined "
            "column is a conservative linear sum, not a statistical error. "
            "The associated $D_1$ scaling is used only as consistency evidence."
        ),
        label="tab:timing-sensitivities",
        columns="llrrrrrrl",
        header=(
            r"$L/M$",
            "observer",
            "PDE",
            "estimator",
            "window",
            "cadence",
            "combined",
            "target",
            "met?",
        ),
        rows=target_rows,
        wide=True,
    )
    return [flat_table, extrapolation_table, localized_table, timing_table]


def create_plots(output_dir: Path, flat: dict, source: dict, l12_rows: list[dict]) -> list[Path]:
    output_dir = Path(output_dir)
    paths: list[Path] = []
    times = flat["times"]
    reference = flat["reference"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(times, reference, color="black", linewidth=2.0, label="Schwarzschild")
    for length in FLAT_LENGTHS:
        axes[0].plot(times, flat["fine_signals"][length], linewidth=1.1, label=f"L/M={length}")
        axes[1].plot(
            times,
            flat["fine_signals"][length] - reference,
            linewidth=1.0,
            label=f"L/M={length}",
        )
    axes[0].set_ylabel("boundary waveform")
    axes[1].set(xlabel=r"fixed retarded time $U/M$", ylabel=r"$W_L-W_{\rm Schw}$")
    axes[0].legend(ncol=4, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 80)
    fig.suptitle(r"Pure $\ell=2$ fixed-data regulator sequence")
    fig.tight_layout()
    paths.extend(_save_publication_figure(fig, output_dir, "flat_waveform_sequence"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for axis, family, windows in (
        (axes[0], "Cumulative", CUMULATIVE_WINDOWS),
        (axes[1], "Disjoint", DISJOINT_WINDOWS),
    ):
        for window, start, end in windows:
            selected = [row for row in flat["metrics"] if row["window"] == window]
            (line,) = axis.loglog(
                [row["cosmological_length_over_M"] for row in selected],
                [row["E2"] for row in selected],
                "o-",
                label=rf"${start:g}\leq U/M\leq {end:g}$",
            )
            axis.loglog(
                [row["cosmological_length_over_M"] for row in selected],
                [row["case_specific_numerical_E2"] for row in selected],
                "x--",
                color=line.get_color(),
                linewidth=0.9,
                alpha=0.75,
            )
        for value, label in ((0.05, "5%"), (0.02, "2%"), (0.01, "1%")):
            axis.axhline(value, color="0.5", linestyle="--", linewidth=0.8)
            axis.text(21, value * 1.05, label, color="0.35", fontsize=8)
        axis.set(
            xscale="log",
            yscale="log",
            xlabel=r"cosmological length $L/M$",
            title=f"{family} windows",
            xlim=(18.0, 760.0),
            xticks=(20, 40, 80, 160, 320, 640),
        )
        axis.set_xticklabels(("20", "40", "80", "160", "320", "640"))
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.tick_params(axis="y", which="minor", labelleft=False)
        axis.grid(which="both", alpha=0.2)
        axis.legend(fontsize=8)
    # The dashed curves carry the conservative case-specific numerical scale,
    # which is the larger of the observed refinement change and the Richardson
    # estimate; the legend must not name only one of the two.
    axes[1].plot([], [], "kx--", linewidth=0.9, label="case-specific numerical scale")
    axes[1].legend(fontsize=8)
    axes[0].set_ylabel(r"direct $E_2(L)$")
    fig.suptitle(r"Pure $\ell=2$ direct errors: cumulative and disjoint intervals")
    fig.tight_layout()
    paths.extend(_save_publication_figure(fig, output_dir, "flat_window_errors"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    axes[0].plot(times, reference, color="black", linewidth=2, label="Schwarzschild")
    for length, style in ((80, "--"), (160, "-.")):
        axes[0].plot(times, flat["extrapolants"][length], style, linewidth=1.5, label=rf"$W_\infty^{{({length})}}$")
        axes[1].plot(times, flat["extrapolants"][length] - reference, style, linewidth=1.3, label=rf"$W_\infty^{{({length})}}-W_{{\rm Schw}}$")
    axes[1].plot(times, flat["extrapolants"][80] - flat["extrapolants"][160], ":", linewidth=1.4, label="extrapolant difference")
    axes[0].set_ylabel("waveform")
    axes[1].set(xlabel=r"fixed retarded time $U/M$", ylabel="difference")
    for axis in axes:
        axis.set_xlim(0, 80)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(r"Pure $\ell=2$ nested extrapolants and Schwarzschild reference")
    fig.tight_layout()
    paths.extend(_save_publication_figure(fig, output_dir, "nested_extrapolants"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for axis, observer in zip(axes, ("r8M", "r12M", "outer")):
        selected = [row for row in source["measurements"] if row["observer"] == observer]
        lengths = np.asarray([row["cosmological_length_over_M"] for row in selected])
        axis.loglog(lengths, [abs(row["D1_over_M"]) for row in selected], "o-", label=r"$|D_1|$")
        axis.loglog(lengths, [row["discretization_D1_uncertainty_over_M"] for row in selected], "s--", label="discretization")
        axis.loglog(lengths, [row["estimator_sensitivity_over_M"] for row in selected], "^--", label="estimator")
        axis.loglog(lengths, [row["window_sensitivity_over_M"] for row in selected], "d--", label="window")
        axis.loglog(lengths, [row["cadence_sensitivity_over_M"] for row in selected], "v--", label="cadence")
        axis.loglog(lengths, [row["combined_fixed_source_timing_sensitivity_over_M"] for row in selected], "k-", label="combined fixed source")
        axis.set_title(observer)
        axis.grid(which="both", alpha=0.2)
        axis.set_xlabel(r"$L/M$")
        axis.set(xlim=(70.0, 720.0), xticks=(80, 160, 320, 640))
        axis.set_xticklabels(("80", "160", "320", "640"))
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.tick_params(axis="y", which="minor", labelleft=False)
    axes[0].set_ylabel(r"timing scale $(M)$")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=6,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(r"Deterministic $D_1$ sensitivity audit")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    paths.extend(_save_publication_figure(fig, output_dir, "D1_error_separation"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    fit_lookup = {row["observer"]: row for row in source["fits"]}
    fit_lengths = np.geomspace(75.0, 700.0, 300)
    for axis, observer in zip(axes, ("r8M", "r12M", "outer")):
        selected = [row for row in source["measurements"] if row["observer"] == observer]
        quantitative = [row for row in selected if not row["diagnostic_only"]]
        diagnostic = [row for row in selected if row["diagnostic_only"]]
        axis.errorbar(
            [row["cosmological_length_over_M"] for row in quantitative],
            [abs(row["D1_over_M"]) for row in quantitative],
            yerr=[row["combined_timing_uncertainty_over_M"] for row in quantitative],
            fmt="o",
            capsize=3,
            label="retained measurement",
        )
        if diagnostic:
            axis.plot(
                [row["cosmological_length_over_M"] for row in diagnostic],
                [abs(row["D1_over_M"]) for row in diagnostic],
                "o",
                markerfacecolor="none",
                markersize=8,
                label="diagnostic",
            )
        fit = fit_lookup[observer]
        if observer == "outer":
            fitted = fit["b1"] / fit_lengths + fit["b2"] / fit_lengths**2
        else:
            fitted = fit["a2"] / fit_lengths**2 + fit["a4"] / fit_lengths**4
        axis.loglog(
            fit_lengths,
            np.abs(fitted),
            "--",
            linewidth=1.2,
            label="scaling consistency guide",
        )
        axis.set(
            title=observer,
            xlabel=r"$L/M$",
            xlim=(70.0, 720.0),
            xticks=(80, 160, 320, 640),
        )
        axis.set_xticklabels(("80", "160", "320", "640"))
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.grid(which="both", alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    outer_handles, outer_labels = axes[-1].get_legend_handles_labels()
    for handle, label in zip(outer_handles, outer_labels):
        if label not in legend_labels:
            handles.append(handle)
            legend_labels.append(label)
    axes[0].set_ylabel(r"$|D_1|/M$")
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(r"$D_1$ scaling as consistency evidence only")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    paths.extend(_save_publication_figure(fig, output_dir, "D1_scaling"))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(l12_rows))
    first = [row["first_pulse_abs_null_ray_residual_over_M"] for row in l12_rows]
    second = [row["second_pulse_abs_null_ray_residual_over_M"] for row in l12_rows]
    axis.bar(x - 0.18, first, width=0.36, label="first pulse")
    axis.bar(x + 0.18, second, width=0.36, label="second pulse")
    axis.axhline(RAY_CONSISTENCY_TOLERANCE_M, color="red", linestyle="--", label="consistency tolerance")
    axis.set_xticks(
        x,
        [
            rf"${float(row['gamma_over_pi']):.3g}\pi$" + "\n" + row["pulse_pair"]
            for row in l12_rows
        ],
    )
    axis.set(
        ylabel=r"absolute simulation-minus-ray arrival $(M)$",
        xlabel=r"angle and pulse pair",
        title=r"Why all $L/M=12$ phase pairs are excluded",
        yscale="log",
    )
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    paths.extend(_save_publication_figure(fig, output_dir, "L12_phase_exclusion"))
    plt.close(fig)

    waveform = source["waveform"]
    source_times = waveform["times"]
    source_metrics = waveform["metrics"]
    source_extrapolation = waveform["extrapolation"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    axes[0, 0].loglog(
        [row["cosmological_length_over_M"] for row in source_metrics],
        [row["E2"] for row in source_metrics],
        "o-",
        color="#16697a",
        label="localized source direct error",
    )
    axes[0, 0].axhline(0.01, color="0.45", linestyle="--", linewidth=0.9, label="1%")
    axes[0, 0].set(
        xlabel=r"$L/M$",
        ylabel=r"sphere-integrated $E_2$",
        title="Direct error on common archived interval",
    )
    axes[0, 0].legend(fontsize=8)
    labels = [
        r"$W_\infty^{(80)}-W_{\rm Schw}$",
        r"$W_\infty^{(160)}-W_{\rm Schw}$",
        r"$W_\infty^{(80)}-W_\infty^{(160)}$",
    ]
    axes[0, 1].bar(
        np.arange(3),
        [row["E2"] for row in source_extrapolation],
        color=("#4c78a8", "#f58518", "#54a24b"),
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(0.01, color="0.45", linestyle="--", linewidth=0.9)
    axes[0, 1].set_xticks(np.arange(3), labels, rotation=12, ha="right")
    axes[0, 1].set(ylabel=r"sphere-integrated $E_2$", title="Nested extrapolants")
    sphere_plot = waveform["sphere_plot"]
    axes[1, 0].semilogy(source_times, sphere_plot["reference"], color="black", label="Schwarzschild norm")
    axes[1, 0].semilogy(source_times, sphere_plot["L320_residual"], label=r"$L/M=320$ residual")
    axes[1, 0].semilogy(source_times, sphere_plot["L640_residual"], label=r"$L/M=640$ residual")
    axes[1, 0].semilogy(source_times, sphere_plot["extrapolant_difference"], label="extrapolant difference")
    axes[1, 0].set(
        xlabel=r"$U/M$",
        ylabel=r"instantaneous $L^2(S^2)$ norm",
        title="Archived interval modal residuals",
    )
    axes[1, 0].set_xlim(23.0, source_times[-1])
    axes[1, 0].set_ylim(1e-7, None)
    axes[1, 0].legend(fontsize=7)
    direction = waveform["direction_plot"]["gamma_pi"]
    axes[1, 1].plot(source_times, direction["reference"], color="black", linewidth=1.5, label="Schwarzschild")
    axes[1, 1].plot(source_times, direction["L320"], linewidth=1.0, label=r"$L/M=320$")
    axes[1, 1].plot(source_times, direction["L640"], linewidth=1.0, label=r"$L/M=640$")
    axes[1, 1].plot(source_times, direction["Winf160"], linestyle="--", linewidth=1.0, label=r"$W_\infty^{(160)}$")
    axes[1, 1].set(xlabel=r"$U/M$", ylabel="directional waveform", title=r"Secondary caustic diagnostic ($\gamma=\pi$)")
    axes[1, 1].set_xlim(35.0, source_times[-1])
    axes[1, 1].legend(fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Localized source on the common archived interval: primary modal norm and supporting caustic diagnostic"
    )
    fig.tight_layout()
    paths.extend(_save_publication_figure(fig, output_dir, "localized_source_regulator"))
    plt.close(fig)
    return paths


def create_analysis(output_dir: Path, repository_root: Path | None = None) -> list[Path]:
    output_dir = Path(output_dir)
    repository_root = Path(repository_root or Path.cwd())
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    flat = flat_analysis(output_dir)
    source = source_analysis(output_dir, repository_root)
    l12_rows = l12_phase_cleanup(repository_root)
    width_rows = _source_width_classification(repository_root)
    written = [
        _write_csv(tables / "flat_waveform_metrics.csv", flat["metrics"]),
        _write_csv(tables / "flat_numerical_errors.csv", flat["numerical"]),
        _write_csv(tables / "flat_successive_L.csv", flat["successive"]),
        _write_csv(tables / "flat_extrapolant_comparisons.csv", flat["extrapolation"]),
        _write_csv(tables / "flat_direct_thresholds.csv", flat["thresholds"]),
        _write_csv(
            tables / "localized_source_waveform_metrics.csv",
            source["waveform"]["metrics"],
        ),
        _write_csv(
            tables / "localized_source_numerical_errors.csv",
            source["waveform"]["numerical"],
        ),
        _write_csv(
            tables / "localized_source_extrapolant_comparisons.csv",
            source["waveform"]["extrapolation"],
        ),
        _write_csv(
            tables / "localized_source_direction_diagnostics.csv",
            source["waveform"]["directions"],
        ),
        _write_csv(tables / "D1_measurements.csv", source["measurements"]),
        _write_csv(tables / "D1_numerical_errors.csv", source["numerical"]),
        _write_csv(tables / "D1_estimator_window_sensitivity.csv", source["sensitivity"]),
        _write_csv(tables / "D1_scaling_fits.csv", source["fits"]),
        _write_csv(tables / "L12_phase_cleanup.csv", l12_rows),
        _write_csv(tables / "source_width_dependence.csv", width_rows),
    ]
    written.extend(write_paper_tables(output_dir, flat, source))
    written.extend(create_plots(output_dir, flat, source, l12_rows))
    summary = {
        "purpose": (
            "Primary result: quantitative artificial-cosmology regulator test and "
            "controlled Schwarzschild waveform recovery. Supporting result: SdS "
            "wave propagation and caustic echoes."
        ),
        "paper_target": "Physical Review D",
        "production_archives_modified": False,
        "cumulative_windows": CUMULATIVE_WINDOWS,
        "disjoint_windows": DISJOINT_WINDOWS,
        "flat_contract": flat["contract"],
        "source_contract": source["contract"],
        "flat_metrics": flat["metrics"],
        "flat_numerical_errors": flat["numerical"],
        "nested_extrapolants": flat["extrapolation"],
        "localized_source_waveform_metrics": source["waveform"]["metrics"],
        "localized_source_numerical_errors": source["waveform"]["numerical"],
        "localized_source_nested_extrapolants": source["waveform"]["extrapolation"],
        "localized_source_direction_diagnostics": source["waveform"]["directions"],
        "direct_thresholds": flat["thresholds"],
        "direct_threshold_scope": "pure ell=2 fixed-data sequence only",
        "localized_source_interval": {
            "classification": "common archived interval",
            "complete_late_time_signal": False,
            "start_U_over_M": source["waveform"]["window"][0],
            "end_U_over_M": source["waveform"]["window"][1],
        },
        "D1_measurements": source["measurements"],
        "D1_scaling_fits": source["fits"],
        "D1_scaling_interpretation": (
            "consistency evidence only; not a precision coefficient measurement "
            "or independent regulator claim"
        ),
        "L12_phase_cleanup": l12_rows,
        "source_width_classification": (
            "physical dependence, excluded from numerical uncertainty and regulator comparison"
        ),
        "L1280_run": False,
        "supported_claim": (
            "agreement within 1% for the cumulative prompt-dominated flat-waveform "
            "norms and the sphere-integrated localized-source extrapolation on the "
            "common archived interval; "
            "disjoint flat windows are diagnostic and do not support a uniform "
            "late-time 1% claim"
        ),
        "central_extrapolant_residual_is_resolved_accuracy": False,
        "deterministic_timing_sensitivities_are_statistical_errors": False,
    }
    written.append(_strict_json(output_dir / "analysis_summary.json", summary))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/regulator_production_v3")
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    for path in create_analysis(arguments.output_dir, arguments.repository_root):
        print(path)


if __name__ == "__main__":
    main()
