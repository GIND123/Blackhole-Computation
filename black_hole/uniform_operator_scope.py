"""Static, assembled-operator cutoff audit; deliberately takes no time steps.

This hooks the actual scalar IVP construction, evaluates its split RHS on
specified constraint-satisfying states, and raises before evolution begins.
It does not change production defaults or validate finite-time waveforms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import splu


class AssemblyComplete(BaseException):
    """Stop the production call before it creates outputs or takes a step."""


def measure(length, resolution, profile, ncc, entry):
    import dedalus
    import dedalus.public as d3
    from .sds_solver import (SdSNumericalParameters, run_sds_simulation,
                             run_schwarzschild_scalar_simulation)
    from .sds_model import SdSParameters, ArealVelocityBumpInitialData
    from .schwarzschild_scalar import SchwarzschildScalarParameters
    from .regulator_suite import flat_initial_data

    saved_build = d3.IVP.build_solver
    report, arrays = {}, {}
    started = time.monotonic()

    def build(problem, *args, **kwargs):
        kwargs.update(ncc_cutoff=ncc, entry_cutoff=entry)
        solver = saved_build(problem, *args, **kwargs)
        ns, state = problem.namespace, tuple(solver.state)
        assert tuple(f.name for f in state) == ("u", "psi", "pi")
        u, psi, pi = state
        derivative = ns["drho"]
        grid = np.asarray(ns["rho"]).ravel().copy()
        nonempty = [sp for sp in solver.subproblems if sp.size]
        if len(nonempty) != 1:
            raise RuntimeError("Serial radial audit requires one subproblem")
        sp = nonempty[0]
        mass_solver = splu(sp.M_min.tocsc())
        report.update(
            background="schwarzschild" if length is None else "uniform",
            L_over_M=length, resolution=resolution, profile=profile,
            dedalus=dedalus.__version__, ncc_cutoff=solver.ncc_cutoff,
            entry_cutoff=solver.entry_cutoff, max_ncc_terms=solver.max_ncc_terms,
            split="implicit transport and potential; zero explicit RHS",
            timestepper="RK222", dealias=1.5, steps_taken=solver.iteration,
            L_nnz=sp.L_min.nnz, M_nnz=sp.M_min.nnz, states=[],
            endpoint_treatment="natural/outflow; no boundary or tau rows",
        )

        def coeff_space():
            solver.evaluator.require_coeff_space(state)

        def put(vector):
            for field in state:
                field.preset_layout("c")
            sp.scatter_inputs(vector, state)

        def values():
            result = []
            for field in state:
                field.change_scales(1)
                result.append(np.asarray(field["g"]).ravel().copy())
            coeff_space()
            return np.asarray(result)

        def constraint():
            c = (psi - derivative(u)).evaluate()
            c.change_scales(1)
            norm = float(np.max(np.abs(c["g"])))
            endpoints = [float(c(rho=x).evaluate()["g"].ravel()[0]) for x in (0., 1.)]
            coeff_space()
            return dict(grid_linf=norm, endpoints=endpoints)

        def audit(label):
            coeff_space()
            x = sp.gather_inputs(state).copy()
            initial_constraint = constraint()
            implicit = mass_solver.solve(-sp.L_min @ x)
            put(x)
            solver.evaluator.evaluate_group("F")
            explicit = mass_solver.solve(sp.gather_outputs(solver.F).copy())
            row = dict(label=label, state_constraint=initial_constraint)
            arrays[label + "_state"] = values()
            for name, vector in (("implicit", implicit), ("explicit", explicit),
                                 ("complete", implicit + explicit)):
                put(vector)
                row[name + "_constraint_rhs"] = constraint()
                arrays[label + "_" + name] = values()
            report["states"].append(row)
            put(x)

        audit("actual_initial_data")
        for f in state:
            f.change_scales(1)
        u["g"] = grid**2 * (1-grid)**2
        du = derivative(u).evaluate()
        du.change_scales(1)
        psi["g"] = du["g"]
        pi["g"] = np.sin(2*np.pi*grid) + .4*(1-grid)
        audit("constraint_satisfying_polynomial_and_momentum")
        arrays["rho"] = grid
        report["seconds"] = time.monotonic() - started
        assert solver.iteration == 0
        raise AssemblyComplete()

    numerical = SdSNumericalParameters(
        resolution=resolution, timestep=.0025, end_time=.0025,
        signal_dt=.05, snapshot_dt=20., observers=(0.,1.),
        timestepper="RK222", bridge="minimal", dealias=1.5)
    initial = flat_initial_data() if profile == "flat" else ArealVelocityBumpInitialData(
        center_radius=6., support_half_width=3., amplitude=1.)
    ell = 2 if profile == "flat" else 1
    d3.IVP.build_solver = build
    try:
        if length is None:
            run_schwarzschild_scalar_simulation(
                SchwarzschildScalarParameters(mass=1., ell=ell), initial, numerical)
        else:
            run_sds_simulation(SdSParameters(mass=1., cosmological_length=length,
                                            ell=ell), initial, numerical)
    except AssemblyComplete:
        pass
    finally:
        d3.IVP.build_solver = saved_build
    if not report or report["steps_taken"] != 0:
        raise RuntimeError("Static assembly audit did not complete")
    return report, arrays


def run(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing existing output {out}")
    out.mkdir(parents=True)
    length = None if args.length == "schwarzschild" else float(args.length)
    cutoffs = [("production",1e-6,1e-12),("tight",1e-10,0.),
               ("roundoff",1e-14,0.)]
    if args.production_tight_only:
        cutoffs = cutoffs[:2]
    if args.zero:
        cutoffs.append(("zero",0.,0.))
    reports, all_arrays = [], {}
    base_arrays = None
    for label,ncc,entry in cutoffs:
        r,a = measure(length,args.resolution,args.profile,ncc,entry)
        r["label"] = label
        if base_arrays is None:
            base_arrays = a
        else:
            r["production_action_changes"] = {}
            for k,v in a.items():
                if k == "rho" or k.endswith("_state"):
                    continue
                baseline = base_arrays[k]
                r["production_action_changes"][k] = {
                    "variable_linf": np.max(np.abs(v-baseline),axis=1).tolist(),
                    "relative_l2": (float(np.linalg.norm(v-baseline)/np.linalg.norm(v))
                                    if np.linalg.norm(v)>0 else 0.),
                }
        reports.append(r)
        all_arrays.update({label+"_"+k:v for k,v in a.items()})
        print(json.dumps(r),flush=True)
    np.savez_compressed(out/"actions.npz",**all_arrays)
    source = Path(__file__)
    payload = dict(reports=reports, archive_sha256=hashlib.sha256(
        (out/"actions.npz").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        simulation_source_sha256=hashlib.sha256(
            source.with_name("sds_solver.py").read_bytes()).hexdigest(),
        limitation="Static operator audit, not finite-time waveform validation")
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length",required=True)
    p.add_argument("--resolution",type=int,required=True)
    p.add_argument("--profile",choices=("flat","velocity"),default="flat")
    p.add_argument("--zero",action="store_true")
    p.add_argument("--production-tight-only",action="store_true",
                   help="Avoid potentially dense roundoff-level NCC fill-in")
    p.add_argument("--output",required=True)
    run(p.parse_args())
