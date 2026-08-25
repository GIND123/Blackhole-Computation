"""Raw waveform analysis for the far-transition production sequence.

This module deliberately contains no fitted alignment or background-dependent
waveform correction.  Every exterior-supported waveform is evaluated on the
retarded-time grid of the frozen fine Schwarzschild archive.  Candidate
discretization errors are obtained only from the candidate's own, generally
unequally spaced, resolution ladder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from .far_regulator_production import (
    CONTROL_ROOT,
    END_TIME,
    LENGTHS,
    OUTPUT_ROOT,
    RESOLUTIONS,
    TIMESTEPS,
    archive_path,
    contract_sha256,
    physical_contract,
)
from .regulator_analysis import (
    ANALYSIS_WINDOWS,
    CUMULATIVE_WINDOWS,
    WINDOW_LATEX_LABELS,
    _align_flat,
    _effective_order,
    _flat_signal,
    _l2,
    _retarded_times,
    _window_family,
    _window_mask,
)
from .regulator_suite import LEVELS, flat_initial_data
from .sds_result import SdSSimulationResult, load_sds_result


HEADLINE_WINDOW = "radiative_signal"
SUBSTANTIAL_REDUCTION_FACTOR = 0.75
DIRECT_THRESHOLDS = (0.05, 0.02, 0.01)
CONTROL_NUMERICAL_TABLE = Path("tables/flat_numerical_errors.csv")

ANALYSIS_CONTRACT = {
    "headline_observable": "raw unshifted outer waveform E2 on 0<=U/M<=80",
    "comparison_grid": "frozen fine Schwarzschild retarded-time grid",
    "candidate_convergence": "candidate-only unequal-resolution ladder",
    "time_translation_fitted": False,
    "amplitude_rescaling_fitted": False,
    "time_dilation_fitted": False,
    "background_transfer_correction_used": False,
    "thresholds_include_schwarzschild_reference_floor": True,
}


def analysis_contract_sha256() -> str:
    """Return a stable fingerprint for the post-processing contract."""

    encoded = json.dumps(
        ANALYSIS_CONTRACT, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if np.isfinite(item) else None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def _control_archive_path(
    control_dir: Path, level: str, length: int | None
) -> Path:
    label = "schwarzschild" if length is None else f"sds_L{length}"
    return Path(control_dir) / "raw" / "flat" / level / f"{label}.npz"


def load_controls(
    control_dir: Path,
    lengths: Iterable[int],
) -> dict[str, dict[str, SdSSimulationResult | dict[int, SdSSimulationResult]]]:
    """Load the frozen Schwarzschild and uniform-SdS archives read-only."""

    selected = tuple(int(length) for length in lengths)
    return {
        level: {
            "schwarzschild": load_sds_result(
                _control_archive_path(control_dir, level, None)
            ),
            "uniform_sds": {
                length: load_sds_result(
                    _control_archive_path(control_dir, level, length)
                )
                for length in selected
            },
        }
        for level in LEVELS
    }


def load_candidates(
    output_dir: Path,
    lengths: Iterable[int],
) -> dict[str, dict[int, SdSSimulationResult]]:
    """Load the isolated exterior-supported production archives."""

    selected = tuple(int(length) for length in lengths)
    return {
        level: {
            length: load_sds_result(archive_path(output_dir, length, level))
            for length in selected
        }
        for level in LEVELS
    }


def _require_waveform_archive(
    result: SdSSimulationResult,
    *,
    label: str,
    maximum_U: float = 160.0,
) -> None:
    """Reject malformed or incomplete outer-waveform archives."""

    times = np.asarray(result.signal_times, dtype=float)
    signals = np.asarray(result.signals, dtype=float)
    observers = np.asarray(result.observer_rho, dtype=float)
    if times.ndim != 1 or times.size < 16 or not np.all(np.isfinite(times)):
        raise ValueError(f"{label} has an invalid signal-time array.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{label} signal times are not strictly increasing.")
    if signals.shape != (times.size, observers.size):
        raise ValueError(f"{label} has inconsistent signal-array dimensions.")
    if not np.all(np.isfinite(signals)):
        raise ValueError(f"{label} contains a nonfinite waveform sample.")
    if result.constraint_linf.size == 0 or result.constraint_l2.size == 0:
        raise ValueError(f"{label} has no constraint diagnostic.")
    if not np.all(np.isfinite(result.constraint_linf)) or not np.all(
        np.isfinite(result.constraint_l2)
    ):
        raise ValueError(f"{label} contains a nonfinite constraint diagnostic.")
    if observers.size == 0 or not np.isclose(observers[-1], 1.0, atol=1.0e-14):
        raise ValueError(f"{label} does not contain the outer-boundary observer.")
    offset = result.metadata.get("retarded_time_offset", {}).get("q")
    if offset is None or not np.isfinite(float(offset)):
        raise ValueError(f"{label} has no finite retarded-time offset.")
    retarded = _retarded_times(result)
    if retarded[0] > 0.0 or retarded[-1] < maximum_U:
        raise ValueError(f"{label} does not cover 0<=U/M<={maximum_U:g}.")


def validate_archives(
    controls: dict,
    candidates: dict,
    lengths: Iterable[int],
) -> None:
    """Enforce the frozen-control and production-candidate contracts."""

    selected = tuple(int(length) for length in lengths)
    expected_initial = flat_initial_data().as_dict()
    expected_contract = json.loads(json.dumps(physical_contract()))
    required_background_checks = (
        "finite_coefficients",
        "positive_interior_lapse",
        "spacelike_bridge_interior",
        "nonnegative_scalar_potential",
    )

    for level in LEVELS:
        schwarzschild = controls[level]["schwarzschild"]
        _require_waveform_archive(schwarzschild, label=f"Schwarzschild {level}")
        if schwarzschild.metadata.get("initial_data") != expected_initial:
            raise ValueError(f"Schwarzschild {level} initial-data mismatch.")
        control_numerical = schwarzschild.metadata.get("numerical", {})
        if float(control_numerical.get("end_time", np.nan)) != END_TIME:
            raise ValueError(f"Schwarzschild {level} endpoint mismatch.")

        for length in selected:
            uniform = controls[level]["uniform_sds"][length]
            candidate = candidates[level][length]
            _require_waveform_archive(
                uniform, label=f"uniform SdS L{length} {level}"
            )
            _require_waveform_archive(
                candidate, label=f"exterior SdS L{length} {level}"
            )
            if uniform.metadata.get("initial_data") != expected_initial:
                raise ValueError(
                    f"Uniform SdS L/M={length} {level} initial-data mismatch."
                )
            if uniform.metadata.get("numerical") != control_numerical:
                raise ValueError(
                    f"Uniform SdS L/M={length} {level} control-grid mismatch."
                )
            if candidate.metadata.get("initial_data") != expected_initial:
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} initial-data mismatch."
                )

            numerical = candidate.metadata.get("numerical", {})
            expected_resolution = RESOLUTIONS[length][level]
            expected_timestep = TIMESTEPS[level]
            required_numerical = {
                "resolution": expected_resolution,
                "timestep": expected_timestep,
                "end_time": END_TIME,
                "signal_dt": 0.03,
                "snapshot_dt": END_TIME,
                "timestepper": "RK222",
                "bridge": "minimal",
                "dealias": 1.5,
            }
            for key, expected in required_numerical.items():
                actual = numerical.get(key)
                if isinstance(expected, float):
                    matches = actual is not None and np.isclose(
                        float(actual), expected, rtol=1.0e-13, atol=1.0e-15
                    )
                else:
                    matches = actual == expected
                if not matches:
                    raise ValueError(
                        f"Exterior SdS L/M={length} {level} has {key}={actual!r}; "
                        f"expected {expected!r}."
                    )

            model = candidate.metadata.get("model", {})
            if float(model.get("cosmological_length", np.nan)) != float(length):
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} model mismatch."
                )
            provenance = candidate.metadata.get("simulation_provenance", {})
            if provenance.get("physical_contract_sha256") != contract_sha256():
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} contract hash mismatch."
                )
            if candidate.metadata.get("physical_contract") != expected_contract:
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} physical contract mismatch."
                )
            preflight = candidate.metadata.get("spectral_preflight", {})
            if not preflight.get("passed", False):
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} lacks a passing preflight."
                )
            if (
                int(preflight.get("resolution", -1)) != expected_resolution
                or int(preflight.get("length_over_M", -1)) != length
                or preflight.get("level") != level
            ):
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} preflight mismatch."
                )
            audit = candidate.metadata.get("background_audit", {})
            if not all(bool(audit.get(key, False)) for key in required_background_checks):
                raise ValueError(
                    f"Exterior SdS L/M={length} {level} background audit failed."
                )


def conservative_refinement_scale(
    coarse_medium: float,
    medium_fine: float,
    resolutions: tuple[int, int, int],
) -> dict[str, float | bool | str]:
    """Return a conservative scale for an arbitrary three-grid ladder.

    A Richardson estimate is accepted only when successive changes decrease
    and a positive effective order can be inferred with the actual grid
    spacings.  A monotone sequence always retains at least the observed
    medium-to-fine change.  A nonmonotone sequence is marked unresolved and
    retains the larger of the two observed changes.
    """

    coarse_medium = float(coarse_medium)
    medium_fine = float(medium_fine)
    if coarse_medium < 0.0 or medium_fine < 0.0:
        raise ValueError("Successive-grid waveform changes must be nonnegative.")
    nc, nm, nf = (int(value) for value in resolutions)
    if not (0 < nc < nm < nf):
        raise ValueError("Resolutions must be strictly increasing.")

    decreasing = coarse_medium > medium_fine > 0.0
    order = (
        _effective_order(coarse_medium, medium_fine, nc, nm, nf)
        if decreasing
        else np.nan
    )
    richardson = np.nan
    if np.isfinite(order) and order > 0.0:
        ratio = nf / nm
        denominator = ratio**order - 1.0
        if denominator > 0.0:
            richardson = medium_fine / denominator

    if decreasing:
        estimated = float(richardson) if np.isfinite(richardson) else medium_fine
        conservative = max(medium_fine, estimated)
        status = (
            "monotone_three_grid_sequence"
            if np.isfinite(richardson)
            else "monotone_sequence_order_unresolved"
        )
    else:
        conservative = max(coarse_medium, medium_fine)
        estimated = conservative
        status = "unresolved_three_grid_sequence"
    return {
        "successive_changes_decrease": decreasing,
        "observed_coupled_order": float(order),
        "richardson_fine_E2": float(richardson),
        "estimated_fine_numerical_E2": float(estimated),
        "conservative_numerical_E2": float(conservative),
        "refinement_status": status,
    }


def raw_waveform_metrics(
    times: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    start: float,
    end: float,
) -> dict[str, float]:
    """Return direct real-waveform norms without alignment or fitted factors."""

    mask = _window_mask(times, start, end)
    local_times = np.asarray(times[mask], dtype=float)
    candidate = np.asarray(candidate[mask], dtype=float)
    reference = np.asarray(reference[mask], dtype=float)
    difference = candidate - reference
    reference_l2 = _l2(reference, local_times)
    reference_linf = float(np.max(np.abs(reference)))
    difference_l2 = _l2(difference, local_times)
    return {
        "reference_l2": reference_l2,
        "difference_l2": difference_l2,
        "E2": difference_l2 / reference_l2,
        "difference_linf_absolute": float(np.max(np.abs(difference))),
        "Einf": float(np.max(np.abs(difference))) / reference_linf,
    }


def improvement_with_margins(
    candidate_error: float,
    candidate_scale: float,
    uniform_error: float,
    uniform_scale: float,
    *,
    substantial_factor: float = SUBSTANTIAL_REDUCTION_FACTOR,
) -> dict[str, float | bool]:
    """Evaluate improvement without using the common reference-grid floor."""

    candidate_upper = float(candidate_error + candidate_scale)
    uniform_lower = float(uniform_error - uniform_scale)
    resolved = uniform_lower > 0.0 and candidate_upper < uniform_lower
    ratio = candidate_upper / uniform_lower if uniform_lower > 0.0 else np.nan
    return {
        "exterior_upper_E2": candidate_upper,
        "uniform_lower_E2": uniform_lower,
        "upper_to_lower_ratio": float(ratio),
        "resolved_improvement_with_numerical_margins": bool(resolved),
        "resolved_reduction_at_least_25_percent": bool(
            resolved and ratio <= substantial_factor
        ),
    }


def threshold_with_reference_floor(
    direct_error: float,
    numerical_scale: float,
    schwarzschild_reference_floor: float,
    threshold: float,
) -> dict[str, float | bool]:
    """Test a direct-error target including candidate and reference scales."""

    upper = float(direct_error + numerical_scale + schwarzschild_reference_floor)
    return {
        "conservative_direct_upper_E2": upper,
        "attained_with_numerical_margin": bool(upper <= threshold),
    }


def load_uniform_numerical_scales(
    control_dir: Path,
    lengths: Iterable[int],
) -> dict[tuple[int, str], float]:
    """Read the frozen uniform-SdS numerical scales from the v3 audit table."""

    selected = {int(length) for length in lengths}
    table = Path(control_dir) / CONTROL_NUMERICAL_TABLE
    rows: dict[tuple[int, str], float] = {}
    with table.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            length = int(float(row["cosmological_length_over_M"]))
            if length not in selected:
                continue
            key = (length, row["window"])
            if key in rows:
                raise ValueError(f"Duplicate frozen numerical row for {key}.")
            value = float(row["conservative_numerical_E2"])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid frozen numerical scale for {key}.")
            rows[key] = value
    expected = {
        (length, window)
        for length in selected
        for window, _, _ in ANALYSIS_WINDOWS
    }
    missing = sorted(expected - rows.keys())
    if missing:
        raise ValueError(f"Frozen numerical table lacks rows: {missing}.")
    return rows


def analyze(
    output_dir: Path,
    control_dir: Path = CONTROL_ROOT,
    lengths: Iterable[int] = LENGTHS,
) -> dict:
    """Analyze a completed subset of the far-transition production sequence."""

    selected = tuple(sorted({int(length) for length in lengths}))
    if not selected:
        raise ValueError("At least one cosmological length is required.")
    if any(length not in LENGTHS for length in selected):
        raise ValueError(f"Lengths must be selected from {LENGTHS}.")

    controls = load_controls(control_dir, selected)
    candidates = load_candidates(output_dir, selected)
    validate_archives(controls, candidates, selected)
    uniform_scales = load_uniform_numerical_scales(control_dir, selected)

    fine_schwarzschild = controls["fine"]["schwarzschild"]
    reference_times = _retarded_times(fine_schwarzschild)
    common = (reference_times >= 0.0) & (reference_times <= 160.0)
    times = reference_times[common]
    reference = _flat_signal(fine_schwarzschild)[common]

    candidate_signals = {
        level: {
            length: _align_flat(candidates[level][length], times)
            for length in selected
        }
        for level in LEVELS
    }
    uniform_signals = {
        level: {
            length: _align_flat(
                controls[level]["uniform_sds"][length], times
            )
            for length in selected
        }
        for level in LEVELS
    }
    schwarzschild_signals = {
        level: _align_flat(controls[level]["schwarzschild"], times)
        for level in LEVELS
    }

    reference_floor_rows: list[dict] = []
    reference_floor: dict[str, float] = {}
    for window, start, end in ANALYSIS_WINDOWS:
        mask = _window_mask(times, start, end)
        denominator = _l2(reference[mask], times[mask])
        floor = _l2(
            schwarzschild_signals["medium"][mask]
            - schwarzschild_signals["fine"][mask],
            times[mask],
        ) / denominator
        reference_floor[window] = floor
        reference_floor_rows.append(
            {
                "window": window,
                "window_family": _window_family(window),
                "window_start_U_over_M": start,
                "window_end_U_over_M": end,
                "medium_fine_schwarzschild_E2": floor,
                "role": "common_reference_grid_floor_for_direct_thresholds",
            }
        )

    direct_rows: list[dict] = []
    for length in selected:
        for level in LEVELS:
            for window, start, end in ANALYSIS_WINDOWS:
                metrics = raw_waveform_metrics(
                    times,
                    candidate_signals[level][length],
                    reference,
                    start,
                    end,
                )
                direct_rows.append(
                    {
                        "cosmological_length_over_M": length,
                        "refinement_level": level,
                        "resolution": RESOLUTIONS[length][level],
                        "window": window,
                        "window_family": _window_family(window),
                        "window_start_U_over_M": start,
                        "window_end_U_over_M": end,
                        **metrics,
                        "reference": "frozen_fine_Schwarzschild",
                        "raw_unshifted": True,
                        "amplitude_rescaling_fitted": False,
                        "time_dilation_fitted": False,
                        "background_transfer_correction_used": False,
                    }
                )

    numerical_rows: list[dict] = []
    candidate_scales: dict[tuple[int, str], float] = {}
    for length in selected:
        resolutions = tuple(RESOLUTIONS[length][level] for level in LEVELS)
        for window, start, end in ANALYSIS_WINDOWS:
            mask = _window_mask(times, start, end)
            denominator = _l2(reference[mask], times[mask])
            coarse_medium = _l2(
                candidate_signals["coarse"][length][mask]
                - candidate_signals["medium"][length][mask],
                times[mask],
            ) / denominator
            medium_fine = _l2(
                candidate_signals["medium"][length][mask]
                - candidate_signals["fine"][length][mask],
                times[mask],
            ) / denominator
            refinement = conservative_refinement_scale(
                coarse_medium, medium_fine, resolutions
            )
            candidate_scales[(length, window)] = float(
                refinement["conservative_numerical_E2"]
            )
            numerical_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "coarse_resolution": resolutions[0],
                    "medium_resolution": resolutions[1],
                    "fine_resolution": resolutions[2],
                    "coarse_medium_candidate_E2": coarse_medium,
                    "medium_fine_candidate_E2": medium_fine,
                    **refinement,
                    "difference_definition": (
                        "candidate waveforms only; frozen controls are not paired "
                        "to candidate refinement levels"
                    ),
                }
            )

    comparison_rows: list[dict] = []
    for length in selected:
        for window, start, end in ANALYSIS_WINDOWS:
            exterior_metrics = raw_waveform_metrics(
                times,
                candidate_signals["fine"][length],
                reference,
                start,
                end,
            )
            uniform_metrics = raw_waveform_metrics(
                times,
                uniform_signals["fine"][length],
                reference,
                start,
                end,
            )
            exterior_scale = candidate_scales[(length, window)]
            uniform_scale = uniform_scales[(length, window)]
            margins = improvement_with_margins(
                exterior_metrics["E2"],
                exterior_scale,
                uniform_metrics["E2"],
                uniform_scale,
            )
            uniform_upper = threshold_with_reference_floor(
                uniform_metrics["E2"],
                uniform_scale,
                reference_floor[window],
                np.inf,
            )["conservative_direct_upper_E2"]
            exterior_upper = threshold_with_reference_floor(
                exterior_metrics["E2"],
                exterior_scale,
                reference_floor[window],
                np.inf,
            )["conservative_direct_upper_E2"]
            comparison_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "uniform_sds_E2": uniform_metrics["E2"],
                    "exterior_sds_E2": exterior_metrics["E2"],
                    "uniform_sds_Einf": uniform_metrics["Einf"],
                    "exterior_sds_Einf": exterior_metrics["Einf"],
                    "uniform_conservative_numerical_E2": uniform_scale,
                    "exterior_conservative_numerical_E2": exterior_scale,
                    "schwarzschild_reference_floor_E2": reference_floor[window],
                    "uniform_conservative_direct_upper_E2": uniform_upper,
                    "exterior_conservative_direct_upper_E2": exterior_upper,
                    "uniform_within_5_percent_with_margin": uniform_upper <= 0.05,
                    "uniform_within_2_percent_with_margin": uniform_upper <= 0.02,
                    "uniform_within_1_percent_with_margin": uniform_upper <= 0.01,
                    "exterior_within_5_percent_with_margin": exterior_upper <= 0.05,
                    "exterior_within_2_percent_with_margin": exterior_upper <= 0.02,
                    "exterior_within_1_percent_with_margin": exterior_upper <= 0.01,
                    "central_improvement_factor": (
                        uniform_metrics["E2"] / exterior_metrics["E2"]
                        if exterior_metrics["E2"] > 0.0
                        else np.inf
                    ),
                    "central_relative_reduction_fraction": (
                        1.0 - exterior_metrics["E2"] / uniform_metrics["E2"]
                        if uniform_metrics["E2"] > 0.0
                        else np.nan
                    ),
                    **margins,
                    "substantial_reduction_definition": (
                        "exterior E2 plus its numerical scale <= 0.75 times "
                        "uniform E2 minus its numerical scale"
                    ),
                    "reference_floor_not_double_counted_in_improvement": True,
                    "raw_unshifted": True,
                    "time_translation_fitted": False,
                    "amplitude_rescaling_fitted": False,
                    "time_dilation_fitted": False,
                    "background_transfer_correction_used": False,
                }
            )

    threshold_rows: list[dict] = []
    for window, _, _ in ANALYSIS_WINDOWS:
        window_rows = [row for row in comparison_rows if row["window"] == window]
        for family in ("uniform_sds", "exterior_sds"):
            scale_field = f"{family.split('_')[0]}_conservative_numerical_E2"
            for threshold in DIRECT_THRESHOLDS:
                qualified = []
                for row in window_rows:
                    result = threshold_with_reference_floor(
                        row[f"{family}_E2"],
                        row[scale_field],
                        row["schwarzschild_reference_floor_E2"],
                        threshold,
                    )
                    if result["attained_with_numerical_margin"]:
                        qualified.append(row["cosmological_length_over_M"])
                threshold_rows.append(
                    {
                        "background_family": family,
                        "window": window,
                        "window_family": _window_family(window),
                        "threshold_fraction": threshold,
                        "smallest_tested_L_over_M": min(qualified) if qualified else "",
                        "status": (
                            "attained_with_numerical_margin"
                            if qualified
                            else f"not_attained_through_L{max(selected)}"
                        ),
                        "criterion": (
                            "raw E2 + family numerical scale + Schwarzschild "
                            "medium-fine reference floor <= threshold"
                        ),
                    }
                )

    archive_audit_rows: list[dict] = []
    for length in selected:
        for level in LEVELS:
            candidate = candidates[level][length]
            preflight = candidate.metadata["spectral_preflight"]
            archive_audit_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "refinement_level": level,
                    "resolution": RESOLUTIONS[length][level],
                    "maximum_constraint_linf": float(
                        np.max(np.abs(candidate.constraint_linf))
                    ),
                    "maximum_constraint_l2": float(
                        np.max(np.abs(candidate.constraint_l2))
                    ),
                    "spectral_preflight_passed": bool(preflight["passed"]),
                    "transition_nodes": int(preflight["transition_nodes"]),
                    "outer_cap_nodes": int(preflight["outer_cap_nodes"]),
                    "maximum_error_over_analytic_minimum_Q": float(
                        preflight["maximum_error_over_analytic_minimum"]
                    ),
                    "simulation_case": candidate.metadata[
                        "simulation_provenance"
                    ]["case"],
                    "simulation_contract_sha256": candidate.metadata[
                        "simulation_provenance"
                    ]["physical_contract_sha256"],
                }
            )

    return {
        "lengths": selected,
        "times": times,
        "reference": reference,
        "candidate_signals": candidate_signals,
        "uniform_signals": uniform_signals,
        "reference_floor": reference_floor_rows,
        "direct": direct_rows,
        "numerical": numerical_rows,
        "comparisons": comparison_rows,
        "thresholds": threshold_rows,
        "archive_audit": archive_audit_rows,
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png = Path(output_dir) / f"{stem}.png"
    pdf = Path(output_dir) / f"{stem}.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def create_plots(output_dir: Path, analysis: dict) -> list[Path]:
    """Write raw waveform and direct-error figures without fitted alignment."""

    output_dir = Path(output_dir)
    times = analysis["times"]
    reference = analysis["reference"]
    lengths = analysis["lengths"]
    exterior = analysis["candidate_signals"]["fine"]
    uniform = analysis["uniform_signals"]["fine"]
    written: list[Path] = []

    figure, axes = plt.subplots(
        len(lengths), 2, figsize=(10.8, 2.55 * len(lengths)), sharex=True
    )
    axes = np.atleast_2d(axes)
    mask = (times >= 0.0) & (times <= 80.0)
    for row, length in enumerate(lengths):
        axes[row, 0].plot(times[mask], reference[mask], color="black", label="Schwarzschild")
        axes[row, 0].plot(
            times[mask], uniform[length][mask], color="#d95f02", label="uniform SdS"
        )
        axes[row, 0].plot(
            times[mask], exterior[length][mask], color="#1b9e77", linestyle="--",
            label="exterior-supported SdS",
        )
        axes[row, 1].plot(
            times[mask], uniform[length][mask] - reference[mask], color="#d95f02",
            label="uniform minus Schwarzschild",
        )
        axes[row, 1].plot(
            times[mask], exterior[length][mask] - reference[mask], color="#1b9e77",
            label="exterior-supported minus Schwarzschild",
        )
        axes[row, 0].set_ylabel(rf"$W$ ($L/M={length}$)")
        axes[row, 1].set_ylabel(r"$\Delta W$")
        for axis in axes[row]:
            axis.grid(alpha=0.2)
        if row == 0:
            axes[row, 0].legend(fontsize=7, ncol=3)
            axes[row, 1].legend(fontsize=7)
    axes[-1, 0].set_xlabel(r"$U/M$")
    axes[-1, 1].set_xlabel(r"$U/M$")
    axes[0, 0].set_title("Raw outer-boundary waveforms")
    axes[0, 1].set_title("Unshifted residuals")
    figure.tight_layout()
    written.extend(_save_figure(figure, output_dir, "far_raw_waveforms"))
    plt.close(figure)

    comparisons = analysis["comparisons"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for color, (window, _, _) in zip(colors, CUMULATIVE_WINDOWS):
        rows = [row for row in comparisons if row["window"] == window]
        x = np.asarray([row["cosmological_length_over_M"] for row in rows])
        uniform_error = np.asarray([row["uniform_sds_E2"] for row in rows])
        exterior_error = np.asarray([row["exterior_sds_E2"] for row in rows])
        uniform_scale = np.asarray(
            [row["uniform_conservative_numerical_E2"] for row in rows]
        )
        exterior_scale = np.asarray(
            [row["exterior_conservative_numerical_E2"] for row in rows]
        )
        label = WINDOW_LATEX_LABELS[window]
        axes[0].errorbar(
            x, uniform_error, yerr=uniform_scale, marker="o", linestyle="--",
            color=color, capsize=2, label=f"uniform, {label}",
        )
        axes[0].errorbar(
            x, exterior_error, yerr=exterior_scale, marker="s", linestyle="-",
            color=color, capsize=2, label=f"exterior, {label}",
        )
        axes[1].plot(
            x, [row["central_improvement_factor"] for row in rows], marker="o",
            color=color, label=label,
        )
    axes[0].set(
        xscale="log", yscale="log", xlabel=r"$L/M$", ylabel=r"$E_2$",
        title="Raw disagreement with Schwarzschild",
    )
    axes[1].axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1].set(
        xscale="log", xlabel=r"$L/M$",
        ylabel=r"$E_2^{\rm uniform}/E_2^{\rm exterior}$",
        title="Central improvement factor",
    )
    for axis in axes:
        axis.set_xticks(lengths, [str(length) for length in lengths])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=7)
    figure.tight_layout()
    written.extend(_save_figure(figure, output_dir, "far_raw_error_reduction"))
    plt.close(figure)
    return written


def create_analysis(
    output_dir: Path,
    control_dir: Path = CONTROL_ROOT,
    lengths: Iterable[int] = LENGTHS,
) -> list[Path]:
    """Write the complete raw production analysis package."""

    output_dir = Path(output_dir)
    result = analyze(output_dir, control_dir, lengths)
    tables = output_dir / "tables"
    written = [
        _write_csv(tables / "far_direct_errors_by_level.csv", result["direct"]),
        _write_csv(
            tables / "far_candidate_numerical_errors.csv", result["numerical"]
        ),
        _write_csv(
            tables / "far_schwarzschild_reference_floor.csv",
            result["reference_floor"],
        ),
        _write_csv(tables / "far_vs_uniform_raw.csv", result["comparisons"]),
        _write_csv(tables / "far_direct_thresholds.csv", result["thresholds"]),
        _write_csv(tables / "far_archive_audit.csv", result["archive_audit"]),
    ]

    aligned = tables / "far_aligned_waveforms.csv"
    columns = [result["times"], result["reference"]]
    headers = ["U_over_M", "schwarzschild_fine"]
    for length in result["lengths"]:
        columns.extend(
            (
                result["uniform_signals"]["fine"][length],
                result["candidate_signals"]["fine"][length],
            )
        )
        headers.extend((f"uniform_sds_L{length}", f"exterior_sds_L{length}"))
    aligned.parent.mkdir(parents=True, exist_ok=True)
    with aligned.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(np.column_stack(columns).tolist())
    written.append(aligned)
    written.extend(create_plots(output_dir, result))

    headline = [
        row for row in result["comparisons"] if row["window"] == HEADLINE_WINDOW
    ]
    summary = {
        "study": "far_horizon_supported_artificial_cosmology_production_analysis",
        "analysis_contract": ANALYSIS_CONTRACT,
        "analysis_contract_sha256": analysis_contract_sha256(),
        "simulation_contract_sha256": contract_sha256(),
        "lengths": result["lengths"],
        "headline_window": HEADLINE_WINDOW,
        "headline_raw_comparisons": headline,
        "schwarzschild_reference_floor": result["reference_floor"],
        "direct_thresholds": result["thresholds"],
        "archive_audit": result["archive_audit"],
        "substantial_reduction_factor": SUBSTANTIAL_REDUCTION_FACTOR,
    }
    written.append(_strict_json(output_dir / "analysis_summary.json", summary))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--control-dir", type=Path, default=CONTROL_ROOT)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=list(LENGTHS), choices=LENGTHS
    )
    arguments = parser.parse_args()
    for path in create_analysis(
        arguments.output_dir, arguments.control_dir, arguments.lengths
    ):
        print(path)


if __name__ == "__main__":
    main()
