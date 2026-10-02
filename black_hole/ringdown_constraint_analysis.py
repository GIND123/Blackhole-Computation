"""Waveform-vector and constraint audit of the targeted shared-flux campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .sds_result import load_sds_result
from .sbp_ringdown_validation import read_run, norm, check_archive_contract


ROOT = Path("results/ringdown_constraint_revision_v1")
SBP_ROOT = Path("results/revision_sbp_ringdown_v1")
WINDOWS = ((15., 45.), (10., 40.), (20., 50.), (0., 80.))
SETTINGS = ((512, .01), (768, .01), (1024, .01), (768, .005))


def load_curve(path):
    result = load_sds_result(path)
    times = result.signal_times - float(result.metadata["retarded_time_offset"]["q"])
    index = int(np.argmin(abs(result.observer_rho - 1)))
    if (not np.all(np.isfinite(result.signals))
            or not np.all(np.isfinite(result.constraint_linf))
            or not np.all(np.diff(times) > 0)):
        raise ValueError(f"Nonfinite or invalid archived evolution: {path}")
    return times, result.signals[:, index], result


def interpolate(times, values, target):
    if target[0] < times[0] or target[-1] > times[-1] + 1e-10:
        raise ValueError("Refuse waveform extrapolation")
    return CubicSpline(times, values)(target)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root=ROOT):
    times = np.arange(8001) * .01
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    input_paths = [root / "campaign_plan.json", Path(__file__),
                   Path(__file__).with_name("ringdown_constraint_diagnostic.py"),
                   Path(__file__).with_name("run_ringdown_constraint_revision.py"),
                   Path(__file__).with_name("sds_solver.py"),
                   Path(__file__).with_name("exterior_sds_model.py"),
                   Path(__file__).with_name("regulator_suite.py"),
                   Path("tests/test_ringdown_constraint_diagnostic.py")]
    input_paths.extend(sorted((root / "diagnostic").glob("*.npz")))
    sbp, sbp_configurations = {}, {}
    for case in ("schwarzschild", "uniform320", "uniform640", "exterior320", "exterior640"):
        for level in ("fine", "medium", "coarse", "half_dt"):
            result = read_run(SBP_ROOT / "runs" / f"{case}_{level}")
            if result is None:
                raise FileNotFoundError(f"Missing complete SBP control {case}_{level}")
            sbp[case, level] = interpolate(result.times, result.u, times)
            sbp_configurations[case, level] = result.configuration
            input_paths.extend(result.path / name for name in (
                "history.csv", "configuration.json", "backend.json"))
    reference = sbp["schwarzschild", "fine"]
    records, constraints, rhs_rows, stage_rows, histories, orderings = [], [], [], [], [], []
    curves = {"U": times, "schwarzschild_sbp_fine": reference}
    for length in (320, 640):
        members = {}
        for n, dt in SETTINGS:
            path = root / "raw" / "shared_flux" / f"L{length}" / f"N{n}_dt{str(dt).replace('.', 'p')}.npz"
            input_paths.append(path)
            tau, values, result = load_curve(path)
            check_archive_contract(result, sbp_configurations[f"exterior{length}", "fine"])
            audit = result.metadata["quadrupole_constraint_audit"]
            if not (np.isfinite(audit["initial_constraint_scale"])
                    and audit["initial_constraint_scale"] > 0
                    and audit["history"] and audit["stages"]):
                raise ValueError(f"Incomplete or invalid constraint scale/history: {path}")
            for row in audit["history"] + audit["stages"]:
                if not all(np.isfinite(row[key]) for key in (
                        "tau", "constraint_linf", "constraint_fixed_scale")):
                    raise ValueError(f"Nonfinite constraint history or stage: {path}")
            wave = interpolate(tau, values, times)
            indices = np.unique(np.r_[np.arange(0, len(tau), 2), len(tau)-1])
            cadence_change = interpolate(tau[indices], values[indices], times) - wave
            members[n, dt] = wave, cadence_change
            curves[f"exterior{length}_N{n}_dt{dt}"] = wave
            history = audit["history"]
            constraints.append(dict(
                L=length, N=n, dt=dt, S0=audit["initial_constraint_scale"],
                maximum_sampled_C=max(row["constraint_linf"] for row in history),
                maximum_sampled_C_hat=max(row["constraint_fixed_scale"] for row in history),
                initial_C=history[0]["constraint_linf"], final_C=history[-1]["constraint_linf"],
                cadence=audit["history_dt"], sample_count=len(history),
                maximum_early_stage_C=max(row["constraint_linf"] for row in audit["stages"]),
                wall_seconds=audit["wall_seconds_with_audit"],
            ))
            for target, source in ((rhs_rows, audit["rhs"]), (stage_rows, audit["stages"]), (histories, history)):
                target.extend(dict(L=length, N=n, dt=dt, **row) for row in source)
        old_path = Path(f"results/exterior_regulator_width_floor_qnm_v5/raw/exterior/L{length}/fine/sds_L{length}.npz")
        input_paths.append(old_path)
        old_t, old_u, old_result = load_curve(old_path)
        check_archive_contract(old_result, sbp_configurations[f"exterior{length}", "fine"])
        old = interpolate(old_t, old_u, times)
        curves[f"exterior{length}_old_fine"] = old
        fine, fine_cadence = members[1024, .01]
        for lo, hi in WINDOWS:
            mask = (times >= lo) & (times <= hi)
            t = times[mask]
            ref = reference[mask]
            denominator = norm(ref, t)
            relative = lambda vector: norm(vector[mask], t) / denominator
            uniform = sbp[f"uniform{length}", "fine"]
            E_uniform = relative(uniform-reference)
            E_shared = relative(fine-reference)
            for (n, dt), (wave, _) in members.items():
                level_error = relative(wave-reference)
                orderings.append(dict(
                    L=length, U_min=lo, U_max=hi, N=n, dt=dt,
                    E_uniform_SBP=E_uniform, E_shared=level_error,
                    Delta_E=E_uniform-level_error,
                    relative_reduction_percent=100*(1-level_error/E_uniform),
                    reference="one common fine SBP Schwarzschild control",
                ))
            # Change the single shared Schwarzschild reference jointly in
            # both errors; it is not two independent stochastic uncertainties.
            paired_reference_sensitivities = []
            for level in ("coarse", "medium", "half_dt"):
                other_ref = sbp["schwarzschild", level]
                other_denominator = norm(other_ref[mask], t)
                other_delta = (
                    norm((uniform-other_ref)[mask], t)
                    - norm((fine-other_ref)[mask], t)
                ) / other_denominator
                paired_reference_sensitivities.append(abs(other_delta-(E_uniform-E_shared)))
            row = dict(
                L=length, U_min=lo, U_max=hi,
                E_uniform_SBP=E_uniform, E_shared_fine=E_shared,
                Delta_E=E_uniform-E_shared,
                relative_reduction_percent=100*(1-E_shared/E_uniform),
                shared_coarse_medium_waveform_change=relative(members[512,.01][0]-members[768,.01][0]),
                shared_medium_fine_waveform_change=relative(members[768,.01][0]-fine),
                shared_medium_half_dt_waveform_change=relative(members[768,.005][0]-members[768,.01][0]),
                shared_fine_half_cadence_change=relative(fine_cadence),
                shared_vs_old_waveform_change=relative(fine-old),
                shared_vs_SBP_waveform_change=relative(fine-sbp[f"exterior{length}","fine"]),
                reference_medium_fine_waveform_change=relative(sbp["schwarzschild","medium"]-reference),
                joint_Delta_E_reference_sensitivity=max(paired_reference_sensitivities),
            )
            records.append(row)
        curves[f"exterior{length}_shared_residual"] = fine-reference
        curves[f"uniform{length}_SBP_residual"] = sbp[f"uniform{length}","fine"]-reference
        curves[f"exterior{length}_shared_minus_old"] = fine-old
        curves[f"exterior{length}_shared_minus_SBP"] = fine-sbp[f"exterior{length}","fine"]
    write_csv(output / "waveform_validation.csv", records)
    write_csv(output / "constraint_summary.csv", constraints)
    write_csv(output / "constraint_histories.csv", histories)
    write_csv(output / "split_rhs_validation.csv", rhs_rows)
    write_csv(output / "initial_stage_validation.csv", stage_rows)
    write_csv(output / "all_level_ordering.csv", orderings)
    np.savez_compressed(output / "actual_waveforms_and_residuals.npz", **curves)
    # Private visual audit, not an additional manuscript figure.  These are
    # signed vectors on the same geometric U grid, without fitted alignment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, constrained_layout=True)
    for column, length in enumerate((320, 640)):
        axes[0, column].plot(times, curves[f"uniform{length}_SBP_residual"], label="Uniform SBP")
        axes[0, column].plot(times, curves[f"exterior{length}_shared_residual"], label="Exterior shared flux")
        axes[0, column].set_title(f"L/M={length}")
        axes[0, column].set_ylabel("Signed waveform residual")
        axes[1, column].plot(times, curves[f"exterior{length}_shared_minus_SBP"], label="Shared minus SBP")
        axes[1, column].plot(times, curves[f"exterior{length}_shared_minus_old"], label="Shared minus old")
        axes[1, column].set_ylabel("Signed formulation change")
        axes[1, column].set_xlabel("U/M")
        for axis in axes[:, column]:
            axis.axvspan(15, 45, color="grey", alpha=.1)
            axis.set_xlim(0, 80)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axis.legend(fontsize=8)
    fig.savefig(output / "private_residual_waveform_audit.pdf")
    fig.savefig(output / "private_residual_waveform_audit.png", dpi=160)
    plt.close(fig)
    output_paths = [output / name for name in (
        "waveform_validation.csv", "constraint_summary.csv", "constraint_histories.csv",
        "split_rhs_validation.csv", "initial_stage_validation.csv", "actual_waveforms_and_residuals.npz",
        "all_level_ordering.csv",
        "private_residual_waveform_audit.pdf", "private_residual_waveform_audit.png")]
    manifest = {
        "complete_runs": 8,
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(set(input_paths))},
        "outputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in output_paths},
        "constraint_sampling": "0.1M full-step history plus endpoint; initial3steps/6RKstages; no continuous-time bound",
        "interpretation": "Observed waveform-vector changes and independent-formulation comparison, not rigorous error bounds or statistical uncertainties",
        "interrupted_runs_excluded": ["L320_N2048_dt0p01", "L640_N2048_dt0p01"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"windows": records, "constraints": constraints}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    analyze(parser.parse_args().root)
