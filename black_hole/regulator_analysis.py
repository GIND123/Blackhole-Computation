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
from .caustic_study import direction_waveform
from .regulator_suite import FLAT_LENGTHS, LEVELS, SOURCE_LENGTHS
from .sds_result import SdSSimulationResult, load_sds_result
from .source_evolution import SourcedSimulationResult, load_sourced_result


FIXED_WINDOWS = (
    ("prompt_and_early_ringdown", 0.0, 40.0),
    ("radiative_signal", 0.0, 80.0),
    ("extended_finite_time", 0.0, 160.0),
)
WINDOW_VARIANTS = {
    "primary": ((18.0, 35.0), (35.0, 53.0)),
    "inset_0p5M": ((18.5, 34.5), (35.5, 52.5)),
    "expanded_0p5M": ((17.5, 35.5), (34.5, 53.5)),
    "shift_left_0p5M": ((17.5, 34.5), (34.5, 52.5)),
    "shift_right_0p5M": ((18.5, 35.5), (35.5, 53.5)),
}
RAY_CONSISTENCY_TOLERANCE_M = 1.0
D1_ANALYSIS_CADENCE_M = 0.001


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
        for window, start, end in FIXED_WINDOWS:
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
        for window, start, end in FIXED_WINDOWS:
            metric_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    **waveform_metrics(
                        times, fine_signals[length], reference, start, end
                    ),
                    "case_specific_numerical_E2": numerical_lookup[(length, window)],
                }
            )

    successive_rows: list[dict] = []
    for lower, upper in zip(FLAT_LENGTHS, FLAT_LENGTHS[1:]):
        for window, start, end in FIXED_WINDOWS:
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
        for window, start, end in FIXED_WINDOWS:
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
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    **metrics,
                    "paired_medium_fine_numerical_E2": numerical,
                    "propagated_estimated_fine_numerical_E2": propagated_fine_error,
                    "agreement_within_1_percent": metrics["E2"] <= 0.01,
                    "numerical_below_0p2_percent": propagated_fine_error < 0.002,
                    "regulator_success": metrics["E2"] <= 0.01
                    and propagated_fine_error < 0.002,
                }
            )

    threshold_rows: list[dict] = []
    for window, _, _ in FIXED_WINDOWS:
        selected = [row for row in metric_rows if row["window"] == window]
        for threshold in (0.05, 0.02, 0.01):
            qualified = [
                row
                for row in selected
                if row["E2"] + row["case_specific_numerical_E2"] <= threshold
            ]
            threshold_rows.append(
                {
                    "window": window,
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


def _d1_variant_rows(
    sds: SourcedSimulationResult,
    schwarzschild: SourcedSimulationResult,
    windows: tuple[tuple[float, float], tuple[float, float]],
) -> list[dict]:
    """Measure correlated D1 values on one resolution-independent U grid."""

    analysis_times = np.arange(
        min(window[0] for window in windows) - 0.5,
        max(window[1] for window in windows) + 0.5 + 0.5 * D1_ANALYSIS_CADENCE_M,
        D1_ANALYSIS_CADENCE_M,
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
                "analysis_cadence_over_M": D1_ANALYSIS_CADENCE_M,
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
) -> list[float]:
    return [row["D1_over_M"] for row in _d1_variant_rows(sds, schwarzschild, windows)]


def _observer_label(result: SourcedSimulationResult, index: int) -> str:
    if index == result.outer_index():
        return "outer"
    return f"r{result.observer_areal_radius[index]:g}M"


def source_analysis(output_dir: Path) -> dict:
    archives = load_source_archives(output_dir)
    contract = validate_contracts(
        [item for values in archives.values() for item in values.values()], "source"
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
            combined = float(
                np.sqrt(numerical**2 + estimator**2 + window_sensitivity**2)
            )
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
                "combined_timing_uncertainty_over_M": combined,
                "signal_to_combined_uncertainty": signal_to_uncertainty,
                "source_width_dependence_included_as_error": False,
                "genuinely_resolved": resolved,
                "genuinely_resolved_criterion": (
                    "all matched-template arrivals interior and "
                    "abs(D1)/quadrature(discretization,estimator,window) >= 3"
                ),
                "diagnostic_only": diagnostic,
                "target_applies": target_applies,
                "target_uncertainty_over_M": numerical_target if target_applies else np.nan,
                "meets_target": numerical < numerical_target if target_applies else np.nan,
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
                    "setting": "matched_template_vs_analytic_envelope",
                    "D1_over_M": primary["matched_template_D1_over_M"],
                    "difference_from_primary_over_M": estimator,
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
            [row["combined_timing_uncertainty_over_M"] for row in eligible],
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
        chi2 = float(np.sum((residual / sigma) ** 2))
        fit_rows.append(
            {
                "observer": observer,
                "model": model,
                "fit_status": "quantitative_case_specific_errors",
                "input_lengths_over_M": ";".join(f"{value:g}" for value in lengths),
                "excluded_diagnostic_lengths_over_M": ";".join(
                    f"{row['cosmological_length_over_M']:g}"
                    for row in selected
                    if row["diagnostic_only"]
                ),
                names[0]: float(coefficients[0]),
                names[1]: float(coefficients[1]),
                "rms_residual_over_M": float(np.sqrt(np.mean(residual**2))),
                "chi_squared": chi2,
                "degrees_of_freedom": int(len(values) - len(coefficients)),
                "fit_weights": (
                    "case-specific quadrature of separately tabulated "
                    "discretization, estimator, and window timing terms"
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
    fig.suptitle("Fixed-data artificial-cosmology waveform sequence")
    fig.tight_layout()
    path = output_dir / "flat_waveform_sequence.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for window, _, _ in FIXED_WINDOWS:
        selected = [row for row in flat["metrics"] if row["window"] == window]
        axis.loglog(
            [row["cosmological_length_over_M"] for row in selected],
            [row["E2"] for row in selected],
            "o-",
            label=window.replace("_", " "),
        )
    for value, label in ((0.05, "5%"), (0.02, "2%"), (0.01, "1%")):
        axis.axhline(value, color="0.5", linestyle="--", linewidth=0.8)
        axis.text(21, value * 1.05, label, color="0.35", fontsize=8)
    axis.set(
        xlabel=r"cosmological length $L/M$",
        ylabel=r"direct $E_2(L)$",
        title="Direct agreement on predeclared fixed windows",
    )
    axis.grid(which="both", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "flat_window_errors.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths.append(path)

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
    fig.suptitle("Nested regulator extrapolants and direct Schwarzschild waveform")
    fig.tight_layout()
    path = output_dir / "nested_extrapolants.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for axis, observer in zip(axes, ("r8M", "r12M", "outer")):
        selected = [row for row in source["measurements"] if row["observer"] == observer]
        lengths = np.asarray([row["cosmological_length_over_M"] for row in selected])
        axis.loglog(lengths, [abs(row["D1_over_M"]) for row in selected], "o-", label=r"$|D_1|$")
        axis.loglog(lengths, [row["discretization_D1_uncertainty_over_M"] for row in selected], "s--", label="discretization")
        axis.loglog(lengths, [row["estimator_sensitivity_over_M"] for row in selected], "^--", label="estimator")
        axis.loglog(lengths, [row["window_sensitivity_over_M"] for row in selected], "d--", label="window")
        axis.set_title(observer)
        axis.grid(which="both", alpha=0.2)
        axis.set_xlabel(r"$L/M$")
    axes[0].set_ylabel(r"timing scale $(M)$")
    axes[-1].legend(fontsize=8)
    fig.suptitle(r"Separated $D_1$ discretization and analysis sensitivities")
    fig.tight_layout()
    path = output_dir / "D1_error_separation.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths.append(path)

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
            label="quantitative",
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
            label="weighted fit",
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
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"$|D_1|/M$")
    fig.suptitle(r"Caustic timing sequence with case-specific uncertainties")
    fig.tight_layout()
    path = output_dir / "D1_scaling.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(l12_rows))
    first = [row["first_pulse_abs_null_ray_residual_over_M"] for row in l12_rows]
    second = [row["second_pulse_abs_null_ray_residual_over_M"] for row in l12_rows]
    axis.bar(x - 0.18, first, width=0.36, label="first pulse")
    axis.bar(x + 0.18, second, width=0.36, label="second pulse")
    axis.axhline(RAY_CONSISTENCY_TOLERANCE_M, color="red", linestyle="--", label="consistency tolerance")
    axis.set_xticks(x, [f"{float(row['gamma_over_pi']):.3g}π\n{row['pulse_pair']}" for row in l12_rows])
    axis.set(
        ylabel=r"absolute simulation-minus-ray arrival $(M)$",
        xlabel=r"angle and pulse pair",
        title=r"Why all $L/M=12$ phase pairs are excluded",
        yscale="log",
    )
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "L12_phase_exclusion.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths.append(path)
    return paths


def create_analysis(output_dir: Path, repository_root: Path | None = None) -> list[Path]:
    output_dir = Path(output_dir)
    repository_root = Path(repository_root or Path.cwd())
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    flat = flat_analysis(output_dir)
    source = source_analysis(output_dir)
    l12_rows = l12_phase_cleanup(repository_root)
    width_rows = _source_width_classification(repository_root)
    written = [
        _write_csv(tables / "flat_waveform_metrics.csv", flat["metrics"]),
        _write_csv(tables / "flat_numerical_errors.csv", flat["numerical"]),
        _write_csv(tables / "flat_successive_L.csv", flat["successive"]),
        _write_csv(tables / "flat_extrapolant_comparisons.csv", flat["extrapolation"]),
        _write_csv(tables / "flat_direct_thresholds.csv", flat["thresholds"]),
        _write_csv(tables / "D1_measurements.csv", source["measurements"]),
        _write_csv(tables / "D1_numerical_errors.csv", source["numerical"]),
        _write_csv(tables / "D1_estimator_window_sensitivity.csv", source["sensitivity"]),
        _write_csv(tables / "D1_scaling_fits.csv", source["fits"]),
        _write_csv(tables / "L12_phase_cleanup.csv", l12_rows),
        _write_csv(tables / "source_width_dependence.csv", width_rows),
    ]
    written.extend(create_plots(output_dir, flat, source, l12_rows))
    summary = {
        "purpose": "Test artificial cosmology as a regulator for asymptotically flat waveforms.",
        "fixed_windows": FIXED_WINDOWS,
        "flat_contract": flat["contract"],
        "source_contract": source["contract"],
        "flat_metrics": flat["metrics"],
        "flat_numerical_errors": flat["numerical"],
        "nested_extrapolants": flat["extrapolation"],
        "direct_thresholds": flat["thresholds"],
        "D1_measurements": source["measurements"],
        "D1_scaling_fits": source["fits"],
        "L12_phase_cleanup": l12_rows,
        "source_width_classification": (
            "physical dependence, excluded from numerical uncertainty and regulator comparison"
        ),
        "L1280_run": False,
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
