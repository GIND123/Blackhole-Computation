"""Independent u,v quadrupole check; not a reduction-constraint experiment.

Only the new revision package is written.  The frozen production files and
manuscript remain untouched.  ``run`` writes the prespecified experiment plan
before preparation or evolution; ``analyze`` compares actual residual vectors.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline

from .sbp_hermite import ROOT, run


DEFAULT = ROOT / "results/revision_sbp_ringdown_v1"
CASES = (("schwarzschild", None), ("uniform", 320), ("uniform", 640),
         ("exterior", 320), ("exterior", 640))
LEVELS = (("coarse", 24, "0.05"), ("medium", 32, "0.05"),
          ("fine", 40, "0.05"), ("half_dt", 32, "0.025"))
WINDOWS = ((0., 80.), (15., 45.), (10., 40.), (20., 50.))


def stem(background, length):
    return background if length is None else f"{background}{length}"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    """Read complete numerical output without importing plotting machinery."""
    configuration_path, backend_path = path/"configuration.json", path/"backend.json"
    if not configuration_path.exists() or not backend_path.exists():
        return None
    config = json.loads(configuration_path.read_text())
    backend = json.loads(backend_path.read_text())
    if not backend.get("completed") or config.get("returncode",0):
        return None
    values = np.genfromtxt(path/"history.csv", delimiter=",", names=True)
    times = values["tau"]-float(config["q"])
    if not np.isclose(times[-1],float(config["end_u"]),atol=1e-10):
        raise ValueError("Incomplete history")
    return SimpleNamespace(path=path, configuration=config, backend=backend,
                           times=times, u=values["u_right"])


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan():
    return dict(
        purpose="Independent second-order u,v check of matched quadrupole waveforms; not shared-flux constraint validation",
        physics=dict(mass=1, ell=2, xi=0, initial_data="u=b((r-4M)/(1.5M)); u_tau=0",
                     bump="b(x)=exp(1-1/(1-x^2)) for |x|<1, otherwise zero",
                     geometries=CASES, end_U=80, reference_radius=4,
                     exterior_profile="frozen width-floor switch and exact endpoint coefficients",
                     clock="U=tau-q, exact geometric q; no fitted alignment or amplitude"),
        numerics=dict(levels=LEVELS, layout="wave-resolved for every background",
                      arithmetic="double-double", coefficient_digits=50, integrator="Radau IIA order 3",
                      output_every=1, windows=WINDOWS, analysis_step=.025,
                      interpolation="cubic with half-output-cadence check and linear sensitivity",
                      reference="same-level Schwarzschild for paired residual refinement; fine Schwarzschild for headline errors"),
        interpretation="Observed changes diagnose resolution; no Richardson extrapolation or claimed rigorous bound.",
    )


def run_all(root, jobs=2):
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    expected = plan()
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(expected)):
            raise ValueError("Existing plan differs")
    else:
        with plan_path.open("x") as stream:
            json.dump(expected, stream, indent=2)
            stream.write("\n")
    backend = ROOT / "black_hole/high_precision/_build/hermite_banded_dd"

    def case(background, length):
        name = stem(background, length)
        for degree in (24, 32, 40):
            prepared = root / "operators" / f"{name}_p{degree}"
            if prepared.exists():
                if not (prepared / "configuration.json").exists():
                    raise ValueError(f"Incomplete preparation: {prepared}")
                continue
            command = [sys.executable, "-m", "black_hole.sbp_hermite", "prepare",
                       "--background", background, "--length", str(length or 640),
                       "--ell", "2", "--coupling", "0", "--initial-data", "displacement",
                       "--degree", str(degree), "--layout", "wave-resolved", "--destination", str(prepared)]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            print(f"Prepared {name} p{degree}", flush=True)
        for level, degree, dt in LEVELS:
            destination = root / "runs" / f"{name}_{level}"
            if destination.exists():
                if read_run(destination) is None:
                    raise ValueError(f"Incomplete run: {destination}")
                continue
            run(root / "operators" / f"{name}_p{degree}", destination,
                backend, dt, "80", 1, 600, "radau3")
            print(f"Completed {name} {level}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(case, background, length) for background, length in CASES]
        for future in futures:
            future.result()


def norm(values, times):
    return float(np.sqrt(np.trapezoid(np.abs(values)**2, x=times)))


def interpolate(times, values, target):
    if target[0] < times[0] or target[-1] > times[-1] + 1e-10:
        raise ValueError("Refuse waveform extrapolation")
    return CubicSpline(times, values)(target)


def old_path(background, length):
    if background == "exterior":
        return ROOT / f"results/exterior_regulator_width_floor_qnm_v5/raw/exterior/L{length}/fine/sds_L{length}.npz"
    filename = "schwarzschild" if background == "schwarzschild" else f"sds_L{length}"
    return ROOT / f"results/regulator_production_v3/raw/flat/fine/{filename}.npz"


def check_archive_contract(result, reference_config):
    model, initial = result.metadata["model"], result.metadata["initial_data"]
    if (model["ell"] != 2 or model["mass"] != 1
            or float(model.get("curvature_coupling", 0)) != 0
            or initial["center_radius"] != 4 or initial["support_half_width"] != 1.5
            or initial["time_symmetric"] is not True):
        raise ValueError("Archive is not the minimally coupled displacement quadrupole")
    if reference_config["background"] != "schwarzschild":
        if float(model["cosmological_length"]) != float(reference_config["L_over_M"]):
            raise ValueError("Archive has a different cosmological length")
    if not np.isclose(float(result.metadata["retarded_time_offset"]["q"]),
                      float(reference_config["q"]),rtol=0,atol=1e-9):
        raise ValueError("Archive has a different geometric clock")
    if reference_config["background"] == "exterior":
        geometry = reference_config["geometry"]
        for old_key,new_key in (("transition_inner_rho","transition_inner_rho"),
                                ("transition_outer_rho","transition_outer_rho")):
            if not np.isclose(float(model[old_key]),float(geometry[new_key]),rtol=0,atol=1e-14):
                raise ValueError("Archive has a different exterior transition")


def analyze(root):
    from .sds_result import load_sds_result
    times = np.arange(3201) * .025
    members, waves, sensitivity, archive_inputs = {}, {}, {}, []
    for background, length in CASES:
        name = stem(background, length)
        members[name], waves[name], sensitivity[name] = {}, {}, {}
        for level, degree, dt in LEVELS:
            result = read_run(root / "runs" / f"{name}_{level}")
            if result is None:
                raise FileNotFoundError(f"Incomplete: {name}_{level}")
            if (result.configuration.get("initial_data_kind") != "displacement"
                    or result.configuration["ell"] != 2):
                raise ValueError("Incompatible physical initial data")
            config = result.configuration
            if (config["background"] != background or float(config["xi"]) != 0
                    or config["degree"] != degree or float(config["dt"]) != float(dt)
                    or config["layout"] != "wave-resolved" or config["integrator"] != "radau3"
                    or result.backend["precision"] != "double-double"
                    or (length is not None and float(config["L_over_M"]) != length)):
                raise ValueError("Run does not match predeclared case and numerical level")
            members[name][level] = result
            waves[name][level] = interpolate(result.times, result.u, times)
            indices = np.unique(np.r_[np.arange(0, len(result.times), 2), len(result.times)-1])
            half_cadence = interpolate(result.times[indices], result.u[indices], times)
            linear = np.interp(times, result.times, result.u)
            sensitivity[name][level] = dict(half=half_cadence-waves[name][level],
                                            linear=linear-waves[name][level])
        path = old_path(background, length)
        old = load_sds_result(path)
        check_archive_contract(old,members[name]["fine"].configuration)
        old_times = old.signal_times-float(old.metadata["retarded_time_offset"]["q"])
        observer = int(np.argmin(abs(old.observer_rho-1)))
        old_signal = np.asarray(old.signals[:,observer],float)
        waves[name]["old"] = interpolate(old_times, old_signal, times)
        archive_inputs.append(path)
    reference = waves["schwarzschild"]["fine"]
    rows, errors, cadences, differences, formulation_rows = [], [], [], [], []
    curves = {"U": times, "Schwarzschild": reference}
    for background, length in CASES:
        name = stem(background, length)
        curves[name] = waves[name]["fine"]
        curves[f"{name}_residual"] = waves[name]["fine"]-reference
        for lo, hi in WINDOWS:
            mask = (times >= lo) & (times <= hi)
            local_times = times[mask]
            denominator = norm(reference[mask], local_times)
            for kind, first, second in (("space_coarse_medium", "coarse", "medium"),
                                        ("space_medium_fine", "medium", "fine"),
                                        ("time_medium_half", "medium", "half_dt"),
                                        ("frozen_vs_new_fine", "old", "fine")):
                delta = waves[name][first]-waves[name][second]
                ref_delta = waves["schwarzschild"][first]-waves["schwarzschild"][second]
                # This is the difference of whole residual waveforms, not
                # the difference of scalar error magnitudes.
                paired_delta = delta-ref_delta
                rows.append(dict(case=name, lo=lo, hi=hi, diagnostic=kind,
                                 waveform_change=norm(delta[mask], local_times)/denominator,
                                 reference_change=norm(ref_delta[mask], local_times)/denominator,
                                 paired_residual_change=norm(paired_delta[mask], local_times)/denominator))
            for level, _, _ in LEVELS:
                cadences.append(dict(case=name, lo=lo, hi=hi, level=level,
                                     half_cadence_change=norm(sensitivity[name][level]["half"][mask],local_times)/denominator,
                                     linear_vs_cubic_change=norm(sensitivity[name][level]["linear"][mask],local_times)/denominator))
                own_ref = waves["schwarzschild"][level]
                residual = waves[name][level]-own_ref
                errors.append(dict(case=name, lo=lo, hi=hi, level=level,
                                   E=norm(residual[mask],local_times)/norm(own_ref[mask],local_times)))
    for length in (320,640):
        for lo, hi in WINDOWS:
            mask = (times>=lo)&(times<=hi)
            for level,_,_ in LEVELS:
                ref = waves["schwarzschild"][level]
                denominator = norm(ref[mask],times[mask])
                uniform = waves[f"uniform{length}"][level]-ref
                exterior = waves[f"exterior{length}"][level]-ref
                eu, ex = norm(uniform[mask],times[mask])/denominator, norm(exterior[mask],times[mask])/denominator
                # Hold both candidates fixed and substitute the same medium
                # reference jointly, including the common norm denominator.
                medium_ref = waves["schwarzschild"]["medium"]
                medium_den = norm(medium_ref[mask],times[mask])
                eu_medium_ref = norm((waves[f"uniform{length}"][level]-medium_ref)[mask],times[mask])/medium_den
                ex_medium_ref = norm((waves[f"exterior{length}"][level]-medium_ref)[mask],times[mask])/medium_den
                half_ref = waves["schwarzschild"]["half_dt"]
                half_den = norm(half_ref[mask],times[mask])
                eu_half_ref = norm((waves[f"uniform{length}"][level]-half_ref)[mask],times[mask])/half_den
                ex_half_ref = norm((waves[f"exterior{length}"][level]-half_ref)[mask],times[mask])/half_den
                interpolation_checks = {}
                for interpolation_kind in ("half","linear"):
                    alt_ref = ref+sensitivity["schwarzschild"][level][interpolation_kind]
                    alt_uniform = waves[f"uniform{length}"][level]+sensitivity[f"uniform{length}"][level][interpolation_kind]
                    alt_exterior = waves[f"exterior{length}"][level]+sensitivity[f"exterior{length}"][level][interpolation_kind]
                    alt_den = norm(alt_ref[mask],times[mask])
                    alt_difference = (norm((alt_uniform-alt_ref)[mask],times[mask])
                                      -norm((alt_exterior-alt_ref)[mask],times[mask]))/alt_den
                    interpolation_checks[f"{interpolation_kind}_interpolation_Delta_E_sensitivity"] = abs(alt_difference-(eu-ex))
                differences.append(dict(L=length,lo=lo,hi=hi,level=level,E_uniform=eu,E_exterior=ex,
                                        Delta_E=eu-ex,reduction_percent=100*(1-ex/eu),
                                        Delta_E_medium_reference=eu_medium_ref-ex_medium_ref,
                                        common_reference_Delta_E_sensitivity=abs((eu_medium_ref-ex_medium_ref)-(eu-ex)),
                                        Delta_E_half_dt_reference=eu_half_ref-ex_half_ref,
                                        common_half_dt_reference_Delta_E_sensitivity=abs((eu_half_ref-ex_half_ref)-(eu-ex)),
                                        **interpolation_checks))
        shared_path = ROOT / f"results/ringdown_constraint_revision_v1/raw/shared_flux/L{length}/N1024_dt0p01.npz"
        if shared_path.exists():
            shared = load_sds_result(shared_path)
            check_archive_contract(shared,members[f"exterior{length}"]["fine"].configuration)
            shared_times = shared.signal_times-float(shared.metadata["retarded_time_offset"]["q"])
            shared_index = int(np.argmin(abs(shared.observer_rho-1)))
            signal = np.asarray(shared.signals[:,shared_index],float)
            if shared_times[-1] >= 80-1e-10:
                shared_signal = interpolate(shared_times,signal,times)
                archive_inputs.append(shared_path)
                curves[f"shared_exterior{length}"] = shared_signal
                for lo,hi in WINDOWS:
                    mask = (times>=lo)&(times<=hi)
                    denominator = norm(reference[mask],times[mask])
                    difference = shared_signal-waves[f"exterior{length}"]["fine"]
                    formulation_rows.append(dict(L=length,lo=lo,hi=hi,
                                                shared_vs_sbp_waveform_change=norm(difference[mask],times[mask])/denominator,
                                                shared_E=norm((shared_signal-reference)[mask],times[mask])/denominator,
                                                reference="SBP fine Schwarzschild common to both residuals"))
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    csv_write(output/"waveform_refinement.csv", rows)
    csv_write(output/"errors.csv", errors)
    csv_write(output/"improvement.csv", differences)
    csv_write(output/"cadence.csv", cadences)
    if formulation_rows:
        csv_write(output/"cross_formulation.csv", formulation_rows)
    lookup = {(r["case"],r["lo"],r["hi"],r["diagnostic"]):r for r in rows}
    shared_lookup = {(r["L"],r["lo"],r["hi"]):r for r in formulation_rows}
    compact = []
    for improvement in differences:
        if improvement["level"] != "fine":
            continue
        length,lo,hi = (improvement[key] for key in ("L","lo","hi"))
        row = dict(improvement)
        for case in ("schwarzschild",f"uniform{length}",f"exterior{length}"):
            label = "reference" if case == "schwarzschild" else case.rstrip("0123456789")
            for column in ("waveform_change","paired_residual_change"):
                row[f"{label}_largest_adjacent_{column}"] = max(
                    lookup[(case,lo,hi,kind)][column]
                    for kind in ("space_coarse_medium","space_medium_fine"))
                row[f"{label}_half_timestep_{column}"] = lookup[(case,lo,hi,"time_medium_half")][column]
                row[f"{label}_frozen_vs_new_{column}"] = lookup[(case,lo,hi,"frozen_vs_new_fine")][column]
        shared = shared_lookup.get((length,lo,hi))
        row["shared_flux_vs_sbp_waveform_change"] = (None if shared is None else shared["shared_vs_sbp_waveform_change"])
        compact.append(row)
    csv_write(output/"compact_summary.csv",compact)
    np.savez_compressed(output/"waveforms.npz", **curves)
    inputs = [path for family in members.values() for member in family.values()
              for path in (member.path/"history.csv",member.path/"configuration.json",member.path/"backend.json")]
    inputs += archive_inputs + [root/"plan.json",Path(__file__)]
    inputs += sorted((root/"operators").glob("*/operator.txt"))
    manifest = dict(plan=json.loads((root/"plan.json").read_text()),
                    files={str(path.relative_to(ROOT)):digest(path) for path in inputs},
                    outputs={str(path.relative_to(ROOT)):digest(path) for path in sorted(output.iterdir())
                             if path.is_file() and path.name != "manifest.json"},
                    notes=["No reduction variable: this check cannot establish shared-flux constraint control.",
                           "Reference refinement is included through paired residual vectors, not independent statistical errors.",
                           "Level-to-level changes and cadence checks are diagnostic scales, not rigorous bounds."])
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for row in differences:
        if row["level"] == "fine" and row["lo"] == 15:
            print(row)


def legacy_audit(root):
    """Read-only L80/160 reanalysis on the exact frozen comparison grid.

    Both fixed-reference and level-paired uniform diagnostics are retained
    explicitly. Exterior levels were not paired to the control resolutions.
    Shared-reference sensitivity substitutes one reference jointly in both
    errors and in their common denominator, never adds independent errors.
    """
    from .sds_result import load_sds_result
    levels = ("coarse","medium","fine")
    inputs = []

    def load(path):
        inputs.append(path)
        result = load_sds_result(path)
        index = int(np.argmin(abs(result.observer_rho-1)))
        times = result.signal_times-float(result.metadata["retarded_time_offset"]["q"])
        return times,np.asarray(result.signals[:,index],float),result.metadata

    controls = {level:load(ROOT/f"results/regulator_production_v3/raw/flat/{level}/schwarzschild.npz")
                for level in levels}
    times, raw_reference, _ = controls["fine"]
    common = (times>=0)&(times<=80)
    times, reference = times[common],raw_reference[common]
    references = {level:np.interp(times,control[0],control[1]) for level,control in controls.items()}
    rows = []
    for length in (80,160):
        candidates, grids = {}, {}
        for background in ("uniform","exterior"):
            candidates[background],grids[background] = {},{}
            for level in levels:
                path = (ROOT/f"results/regulator_production_v3/raw/flat/{level}/sds_L{length}.npz"
                        if background == "uniform" else
                        ROOT/f"results/exterior_regulator_far_production_v1/raw/exterior/L{length}/{level}/sds_L{length}.npz")
                t,w,metadata = load(path)
                if t[0] > times[0] or t[-1] < times[-1]:
                    raise ValueError("Legacy archive does not cover frozen comparison grid")
                candidates[background][level] = np.interp(times,t,w)
                grids[background][level] = metadata["numerical"]
        for lo,hi in ((15.,45.),(10.,40.),(20.,50.),(0.,80.)):
            mask = (times>=lo)&(times<=hi)
            t = times[mask]
            den = norm(reference[mask],t)
            changes = {}
            for background in ("uniform","exterior"):
                for first,second in (("coarse","medium"),("medium","fine")):
                    delta = candidates[background][first]-candidates[background][second]
                    changes[f"{background}_{first}_{second}"] = norm(delta[mask],t)/den
                    if background == "uniform":
                        paired = delta-(references[first]-references[second])
                        changes[f"uniform_paired_{first}_{second}"] = norm(paired[mask],t)/den
            eu = norm((candidates["uniform"]["fine"]-reference)[mask],t)/den
            ex = norm((candidates["exterior"]["fine"]-reference)[mask],t)/den
            medium_ref = references["medium"]
            medium_den = norm(medium_ref[mask],t)
            eu_med = norm((candidates["uniform"]["fine"]-medium_ref)[mask],t)/medium_den
            ex_med = norm((candidates["exterior"]["fine"]-medium_ref)[mask],t)/medium_den
            rows.append(dict(L=length,lo=lo,hi=hi,E_uniform=eu,E_exterior=ex,
                             Delta_E=eu-ex,reduction_percent=100*(1-ex/eu),
                             **changes,
                             uniform_max_fixed_reference_change=max(changes[f"uniform_{a}_{b}"] for a,b in (("coarse","medium"),("medium","fine"))),
                             uniform_max_paired_change=max(changes[f"uniform_paired_{a}_{b}"] for a,b in (("coarse","medium"),("medium","fine"))),
                             exterior_max_change=max(changes[f"exterior_{a}_{b}"] for a,b in (("coarse","medium"),("medium","fine"))),
                             common_reference_medium_fine=norm((medium_ref-reference)[mask],t)/den,
                             common_reference_coarse_medium=norm((references["coarse"]-medium_ref)[mask],t)/den,
                             Delta_E_medium_reference=eu_med-ex_med,
                             common_reference_Delta_E_sensitivity=abs((eu_med-ex_med)-(eu-ex))))
    output=root/"legacy_analysis"
    output.mkdir(parents=True,exist_ok=True)
    csv_write(output/"legacy_ringdown_audit.csv",rows)
    (output/"manifest.json").write_text(json.dumps(dict(
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        analysis_source_sha256=digest(Path(__file__)),
        convention="Original fine-Schwarzschild grid, original linear interpolation, unchanged windows and clock",
        diagnostics="Largest observed adjacent coupled-level vector change; no Richardson model or proven bound",
        uniform_alternatives="Fixed-reference and level-paired residual changes are separate named columns",
        reference_sensitivity="Replace Schwarzschild fine by medium jointly in both candidate norms and denominator",
    ),indent=2)+"\n")
    for row in rows:
        if row["lo"] == 15:
            print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "analyze", "legacy"))
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--jobs",type=int,default=2)
    args = parser.parse_args()
    if args.operation == "run":
        run_all(args.root,args.jobs)
    elif args.operation == "analyze":
        analyze(args.root)
    else:
        legacy_audit(args.root)


if __name__ == "__main__":
    main()
