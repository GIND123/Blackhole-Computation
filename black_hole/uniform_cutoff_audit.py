"""Isolated, non-invasive cutoff regression for the archived dipole benchmark.

Only Dedalus's coefficient and assembled-entry cutoffs are overridden.  The
production PDE, initial data, RK split, outputs, and geometric clock are reused.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import splu


@contextmanager
def instrument(*, ncc_cutoff, entry_cutoff, history_dt=1.0,
               rhs_times=(100.0, 250.0, 375.0, 450.0)):
    import dedalus.public as d3

    original_build = d3.IVP.build_solver
    report = {"rhs": [], "history": [], "ncc_cutoff": ncc_cutoff,
              "entry_cutoff": entry_cutoff, "history_dt": history_dt,
              "sampled_not_continuous_maxima": True}

    def build(problem, *args, **kwargs):
        kwargs.update(ncc_cutoff=ncc_cutoff, entry_cutoff=entry_cutoff)
        started = time.perf_counter()
        solver = original_build(problem, *args, **kwargs)
        report["assembly_seconds"] = time.perf_counter() - started
        state = tuple(solver.state)
        if tuple(f.name for f in state) != ("u", "psi", "pi"):
            raise ValueError("This audit is for the unchanged standard formulation")
        ns = problem.namespace
        derivative = ns["drho"]
        constraint = state[1] - derivative(state[0])
        subproblems = [sp for sp in solver.subproblems if sp.size]
        if len(subproblems) != 1:
            raise ValueError("Expected one serial radial subproblem")
        sp = subproblems[0]
        mass = splu(sp.M_min.tocsc())
        report.update(matrix_shape=list(sp.L_min.shape),
                      matrix_nnz=int(sp.L_min.nnz),
                      dedalus_version=__import__("dedalus").__version__)

        def coeff():
            solver.evaluator.require_coeff_space(state)

        def norm():
            value = constraint.evaluate()
            value.change_scales(1)
            result = float(np.max(np.abs(value["g"])))
            coeff()
            return result

        def save():
            coeff()
            return sp.gather_inputs(state).copy()

        def put(vector):
            for field in state:
                field.preset_layout("c")
            sp.scatter_inputs(vector, state)

        def rhs(label):
            x = save()
            row = {"datum": label, "tau": float(solver.sim_time),
                   "constraint_linf": norm()}
            implicit = mass.solve(-sp.L_min @ x)
            put(x)
            solver.evaluator.evaluate_group("F")
            explicit = mass.solve(sp.gather_outputs(solver.F).copy())
            for name, vector in (("implicit", implicit), ("explicit", explicit),
                                 ("complete", implicit + explicit)):
                put(vector)
                row[name + "_constraint_rhs_linf"] = norm()
            report["rhs"].append(row)
            put(x)

        # For velocity data S0=0.  G_v=A*pi is the fixed, nonzero physical
        # velocity scale; it is not a relative waveform-error normalization.
        initial_x = save()
        for field in state:
            field.change_scales(1)
        ns["coefficient_a"].change_scales(1)
        fixed_scale = float(ns["model"].mass * np.max(np.abs(
            ns["coefficient_a"]["g"] * state[2]["g"])))
        if not fixed_scale > 0:
            raise ValueError("Velocity datum must have nonzero M||G_v||inf")
        report["fixed_velocity_scale"] = fixed_scale
        report["normalization"] = "M*||G_v||_infinity (not a waveform error)"
        rhs("actual_initial_velocity")
        rho = np.asarray(ns["rho"]).ravel()
        for field in state:
            field.change_scales(1)
        state[0]["g"] = rho**2 * (1-rho)**2
        du = derivative(state[0]).evaluate()
        du.change_scales(1)
        state[1]["g"] = du["g"]
        ns["coefficient_b"].change_scales(1)
        state[2]["g"] = -ns["coefficient_b"]["g"] * du["g"]
        rhs("constraint_satisfying_polynomial")
        put(initial_x)

        def sample():
            value = norm()
            report["history"].append({"tau": float(solver.sim_time),
                                      "constraint_linf": value,
                                      "constraint_velocity_scale": value/fixed_scale})

        sample()
        original_step = solver.step
        targets = iter(rhs_times)
        next_rhs = next(targets, np.inf)
        last_history = float(solver.sim_time)
        step_start = time.perf_counter()

        def step(dt):
            nonlocal next_rhs, last_history
            original_step(dt)
            final = solver.sim_time >= solver.stop_sim_time - 1e-9
            if solver.sim_time >= next_rhs - 1e-9 or final:
                rhs("evolved_state")
                next_rhs = next(targets, np.inf)
            if solver.sim_time-last_history >= history_dt - 1e-9 or final:
                sample()
                last_history = float(solver.sim_time)
            if final:
                report["evolution_seconds"] = time.perf_counter() - step_start

        solver.step = step
        return solver

    d3.IVP.build_solver = build
    try:
        yield report
    finally:
        d3.IVP.build_solver = original_build


def run(args):
    from .large_l_tail import (TailCase, _offset, _observer_coordinates,
                              INITIAL_DATA, EXPLICIT_POTENTIAL)
    from .sds_model import SdSParameters
    from .schwarzschild_scalar import SchwarzschildScalarParameters
    from .sds_solver import (SdSNumericalParameters, run_sds_simulation,
                             run_schwarzschild_scalar_simulation)

    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    case = TailCase(args.background, args.resolution, args.dt, args.end_u,
                    3072.0, "cutoff_regression")
    offset = _offset(case)
    numerical = SdSNumericalParameters(
        resolution=args.resolution, timestep=args.dt,
        end_time=args.end_tau if args.end_tau is not None else args.end_u+offset,
        signal_dt=.05, snapshot_dt=25.0, observers=_observer_coordinates(case),
        timestepper="RK222", bridge="minimal", dealias=1.5)
    model = (SdSParameters(mass=1, ell=1, cosmological_length=3072.0)
             if args.background == "sds" else SchwarzschildScalarParameters(mass=1, ell=1))
    evolve = run_sds_simulation if args.background == "sds" else run_schwarzschild_scalar_simulation
    start = time.perf_counter()
    with instrument(ncc_cutoff=args.ncc_cutoff, entry_cutoff=args.entry_cutoff,
                    history_dt=args.history_dt) as audit:
        result = evolve(model, INITIAL_DATA, numerical, explicit_potential=EXPLICIT_POTENTIAL)
    audit["wall_seconds"] = time.perf_counter()-start
    audit["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("uniform_cutoff_audit.py", "sds_solver.py", "large_l_tail.py",
                     "sds_model.py", "schwarzschild_scalar.py")}
    audit["sampled_maximum_constraint_linf"] = max(x["constraint_linf"] for x in audit["history"])
    audit["sampled_maximum_constraint_velocity_scale"] = max(x["constraint_velocity_scale"] for x in audit["history"])
    result.metadata["dedalus_matrix_assembly"] = {
        "ncc_cutoff": args.ncc_cutoff, "entry_cutoff": args.entry_cutoff}
    result.metadata["uniform_cutoff_audit"] = audit
    result.metadata["retarded_time_offset"] = {"q": offset, "evaluation": "analytic"}
    result.metadata["large_l_tail_case"] = {**asdict(case), "explicit_potential": True,
                                          "retarded_time": "U=tau-q", "time_translation_fitted": False}
    result.save(output)
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: value for key, value in audit.items()
                      if key not in ("history", "source_sha256")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", choices=("sds", "schwarzschild"), required=True)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--dt", type=float, default=.0025)
    parser.add_argument("--end-u", type=float, default=450.0)
    parser.add_argument("--end-tau", type=float)
    parser.add_argument("--ncc-cutoff", type=float, required=True)
    parser.add_argument("--entry-cutoff", type=float, required=True)
    parser.add_argument("--history-dt", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
