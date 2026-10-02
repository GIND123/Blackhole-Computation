"""Frozen-rule, tail-normalized analysis of the isolated operator-cutoff audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .large_l_tail import (LocalFitSettings, _load_final_set, _interpolate,
                          _longest_true_interval, ladder_envelope_floor,
                          _anchored_long_interval_end,
                          final_cases, archive_path,
                          measure_transition, retarded_series, effective_rates,
                          cosmological_rate, PRICE_DURATION, PRICE_TOLERANCE,
                          PRICE_TARGET_INFINITY, SCREEN_PRICE_ANCHOR_U)
from .sds_result import load_sds_result


PUBLISHED_INTERVAL = (231.88, 381.88)
COMPOSED_PADDING = 25.0


def validate_pair(reference, candidate):
    """Reject incomplete or physically unmatched archives before comparison."""
    for name, result in (("reference", reference), ("candidate", candidate)):
        if (not np.all(np.isfinite(result.signals))
                or not np.all(np.isfinite(result.u_snapshots))
                or not np.all(np.diff(result.signal_times) > 0)
                or result.signal_times[0] != 0):
            raise ValueError(f"Invalid or non-finite {name} trajectory")
    for key in ("model", "initial_data", "equations", "imex_split"):
        if reference.metadata[key] != candidate.metadata[key]:
            raise ValueError(f"Unmatched {key}")
    for key in ("resolution", "timestep", "signal_dt", "snapshot_dt",
                "observers", "bridge", "dealias", "timestepper"):
        if reference.metadata["numerical"][key] != candidate.metadata["numerical"][key]:
            raise ValueError(f"Unmatched numerical setting {key}")
    for key in ("rho", "observer_rho"):
        np.testing.assert_array_equal(getattr(reference, key), getattr(candidate, key))
    q = candidate.metadata["retarded_time_offset"]["q"]
    if reference.metadata["retarded_time_offset"]["q"] != q:
        raise ValueError("Unmatched geometric clock")
    end = candidate.metadata["numerical"]["end_time"]
    if (abs(candidate.signal_times[-1] - end) > 1e-8
            or abs(candidate.metadata["final_time"] - end) > 1e-8):
        raise ValueError("Candidate did not finish normally")
    if candidate.signal_times[-1]-q < 450.0-1e-8:
        raise ValueError("Candidate record is shorter than the planned U450")
    if candidate.signal_times[-1]-q < PUBLISHED_INTERVAL[1]+COMPOSED_PADDING:
        raise ValueError("Insufficient composed centered-estimator padding")
    old_cutoffs = reference.metadata["dedalus_matrix_assembly"]
    new_cutoffs = candidate.metadata["dedalus_matrix_assembly"]
    if old_cutoffs != {"ncc_cutoff": 1e-6, "entry_cutoff": 1e-12}:
        raise ValueError("Unexpected production cutoffs")
    if new_cutoffs != {"ncc_cutoff": 1e-10, "entry_cutoff": 0.0}:
        raise ValueError("Unexpected tightened cutoffs")
    audit = candidate.metadata["uniform_cutoff_audit"]
    if not audit["fixed_velocity_scale"] > 0 or not audit["history"]:
        raise ValueError("Missing valid velocity-normalized constraint history")
    if any(not np.isfinite(row["constraint_linf"]) for row in audit["history"]):
        raise ValueError("Non-finite constraint history")
    return {
        "matching_physical_configuration": True,
        "normal_completion": True,
        "end_U": float(candidate.signal_times[-1]-q),
        "padding_beyond_fixed_interval": float(candidate.signal_times[-1]-q-PUBLISHED_INTERVAL[1]),
        "required_composed_padding": COMPOSED_PADDING,
        "same_timestep_and_output_convention": True,
        "production_Dedalus_version": reference.metadata.get("reproducibility", {}).get("dedalus"),
        "candidate_Dedalus_version": candidate.metadata.get("reproducibility", {}).get("dedalus"),
        "comparison_caveat": "Radial assembly paths are compatible, but platform/dependency arithmetic changes are not independently isolated from cutoff changes.",
    }


def compare(reference, candidate, floor_times, floor):
    times, signal = retarded_series(reference, 2)
    other_times, other_signal = retarded_series(candidate, 2)
    # Work entirely inside both records; never extrapolate beyond the audit.
    keep = times <= other_times[-1]
    times, signal = times[keep], signal[keep]
    other = _interpolate(other_times, other_signal, times)
    measured_floor = _interpolate(floor_times, floor, times)
    kappa = cosmological_rate(3072.0)
    amplitude, power, _ = effective_rates(times, signal, LocalFitSettings(),
                                        kappa=kappa, measured_floor=measured_floor)
    other_amplitude, other_power, _ = effective_rates(
        times, other, LocalFitSettings(), kappa=kappa, measured_floor=measured_floor)
    window = (times >= PUBLISHED_INTERVAL[0]) & (times <= PUBLISHED_INTERVAL[1])
    valid = window & np.isfinite(amplitude) & np.isfinite(other_amplitude) & np.isfinite(power) & np.isfinite(other_power)
    if np.count_nonzero(valid) != np.count_nonzero(window):
        raise ValueError("An unchanged fixed-window mask rejected audit samples")
    return {
        "window_lower": PUBLISHED_INTERVAL[0], "window_upper": PUBLISHED_INTERVAL[1],
        "sample_count": int(np.count_nonzero(valid)),
        "maximum_relative_envelope_change": float(np.max(np.abs(other_amplitude[valid]-amplitude[valid])/amplitude[valid])),
        "relative_tail_waveform_l2": float(np.sqrt(np.trapezoid((other[window]-signal[window])**2, times[window])/np.trapezoid(signal[window]**2, times[window]))),
        "maximum_absolute_local_index_change": float(np.max(np.abs(other_power[valid]-power[valid]))),
    }


def classify(sds, schwarzschild, floors):
    times, _ = retarded_series(sds, 2)
    measured_floor = _interpolate(floors["sds"]["times"], floors["sds"]["floor"], times)
    result = measure_transition(sds, schwarzschild, 2, LocalFitSettings(), measured_floor=measured_floor)
    rt, rs = retarded_series(schwarzschild, 2)
    reference_floor = _interpolate(floors["schwarzschild"]["times"], floors["schwarzschild"]["floor"], rt)
    _, rp, _ = effective_rates(rt, rs, LocalFitSettings(), kappa=cosmological_rate(3072.0), measured_floor=reference_floor)
    rp = _interpolate(rt, rp, times)
    refinement = {}
    for bg in ("sds", "schwarzschild"):
        record = floors[bg]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = record["spatial_fine"] / record["amplitude"]
        refinement[bg] = _interpolate(record["times"], relative, times)
    tolerance = PRICE_TARGET_INFINITY*PRICE_TOLERANCE
    good = ((times >= 0) & (np.abs(result["power"]-3) <= tolerance)
            & (np.abs(rp-3) <= tolerance)
            & (np.abs(result["power"]-rp) <= tolerance)
            & (refinement["sds"] <= .01)
            & (refinement["schwarzschild"] <= .01))
    longest = _longest_true_interval(times, good)
    departure = _anchored_long_interval_end(times, good, PRICE_DURATION, SCREEN_PRICE_ANCHOR_U)
    accepted = (times >= departure-PRICE_DURATION) & (times <= departure) if departure is not None else np.zeros_like(times, dtype=bool)
    return {
        "departure_U_over_M": departure,
        "rule_selected_interval": None if departure is None else [departure-PRICE_DURATION, departure],
        "classification": "passes" if departure is not None else "fails",
        "longest_continuous_interval": longest,
        "longest_duration": None if longest is None else longest[1]-longest[0],
        "anchor": SCREEN_PRICE_ANCHOR_U, "required_duration": PRICE_DURATION,
        "p_target": PRICE_TARGET_INFINITY, "relative_band": PRICE_TOLERANCE,
        "mutual_index_band": tolerance,
        "maximum_original_fine_envelope_refinement_in_selected_interval": {
            bg: float(np.max(values[accepted])) if np.any(accepted) else None
            for bg, values in refinement.items()},
        "envelope_refinement_condition": "both original finest-grid envelope differences <=1%; not a new refined ladder for tightened cutoffs",
        "measured_floor": "unchanged original finest spatial/temporal envelope ladder",
        "record_end_U": float(times[-1]),
        "composed_price_estimator_padding_M": COMPOSED_PADDING,
    }


def analyze(root):
    root = Path(root)
    original = _load_final_set(Path("results/large_l_tail"), 3072.0)
    settings = LocalFitSettings()
    floors = {bg: ladder_envelope_floor(original, 2, settings, background=bg)
              for bg in ("sds", "schwarzschild")}
    runs = {}
    input_paths = [archive_path(Path("results/large_l_tail"), case)
                   for case in final_cases(3072.0)]
    validation = {}
    for bg in ("sds", "schwarzschild"):
        path = root/"raw"/f"{bg}_N2048_tight.npz"
        runs["tight", bg] = load_sds_result(path)
        runs["production", bg] = original[bg, 2048, .0025]
        input_paths.append(path)
        validation[bg] = validate_pair(runs["production", bg], runs["tight", bg])
    metrics = []
    for bg in ("sds", "schwarzschild"):
        metrics.append({"background": bg, "comparison": "compatible_archived_production_to_tightened",
                        **compare(runs["production", bg], runs["tight", bg], floors[bg]["times"], floors[bg]["floor"])})
    classifications = {
        label: classify(runs[label, "sds"], runs[label, "schwarzschild"], floors)
        for label in ("production", "tight")}
    classifications["cutoff_departure_change"] = (
        classifications["tight"]["departure_U_over_M"] - classifications["production"]["departure_U_over_M"]
        if all(classifications[x]["departure_U_over_M"] is not None for x in ("production", "tight")) else None)
    audits = {f"{label}_{bg}": run.metadata["uniform_cutoff_audit"]
              for (label, bg), run in runs.items() if label == "tight"}
    sensitivity_path = Path("results/large_l_tail/tables/final_L3072_numerical_sensitivities.csv")
    input_paths.append(sensitivity_path)
    with sensitivity_path.open(newline="") as handle:
        old_sensitivities = list(csv.DictReader(handle))
    scale_comparisons = []
    for metric in metrics:
        matching = [row for row in old_sensitivities if row["background"] == metric["background"]]
        for row in matching:
            old_scale = float(row["maximum_relative_envelope_difference_price"])
            scale_comparisons.append({
                "background": metric["background"], "original_refinement": row["sensitivity"],
                "original_maximum_relative_envelope_change": old_scale,
                "measured_configuration_change_divided_by_original_scale": metric["maximum_relative_envelope_change"]/old_scale})
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out/"fixed_interval_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (out/"summary.json").write_text(json.dumps({"metrics": metrics,
        "classifications": classifications, "diagnostics": audits,
        "validation": validation,
        "comparison_to_original_refinement_scales": scale_comparisons,
        "scope": "N2048 cutoff regression through U450; not long-record cutoff validation",
        "scientific_interpretation": {
            "rate_interval_classification_stable": all(row["classification"] == "passes" for key, row in classifications.items() if isinstance(row, dict)),
            "amplitude_sensitivity_subdominant_to_finest_spatial_change": False,
            "original_finest_grid_interval_replaced": False,
            "new_refined_production_calculation": False,
            "second_tighter_trajectory_performed": False,
            "cutoff_plateau_established": False,
            "restriction": "The measured combined cutoff/platform sensitivity leaves the original rate criterion satisfied but does not demonstrate waveform-cutoff insensitivity at finest-refinement precision, a continuum error bound, or the long record near U1984. The unchanged archived spatial floor is support from the original ladder, not a new tightened-cutoff ladder."},
        "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths},
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(__file__).with_name("uniform_cutoff_audit.py"),
                                       Path(__file__).with_name("large_l_tail.py"))}}, indent=2))
    print(json.dumps({"metrics": metrics, "classifications": classifications}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/uniform_cutoff_revision_v1"))
    analyze(parser.parse_args().root)


if __name__ == "__main__":
    main()
