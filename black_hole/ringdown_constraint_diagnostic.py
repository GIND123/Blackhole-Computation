"""Non-invasive Dedalus split/stage audit for the quadrupole comparison.

The hooks observe the actual assembled production equations and RK222 stages;
they do not replace the PDE, coefficients, endpoint treatment, or integrator.
New archives are written only to an explicitly supplied, unused destination.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import splu


@contextmanager
def instrument_solver(*, history_dt: float, stage_steps: int = 3,
                      ncc_cutoff: float | None = None,
                      entry_cutoff: float | None = None):
    """Measure the assembled implicit/explicit RHS and sampled constraints."""
    import dedalus.public as d3

    original_build = d3.IVP.build_solver
    report = {"history": [], "stages": [], "rhs": []}

    def build(problem, *args, **kwargs):
        if ncc_cutoff is not None:
            kwargs["ncc_cutoff"] = ncc_cutoff
        if entry_cutoff is not None:
            kwargs["entry_cutoff"] = entry_cutoff
        solver = original_build(problem, *args, **kwargs)
        ns = problem.namespace
        state = tuple(solver.state)
        u = state[0]
        derivative = ns["drho"]
        characteristic = state[1].name == "h"
        constraint_operator = (
            0.5 * (state[1] - state[2]) - derivative(u)
            if characteristic else state[1] - derivative(u)
        )
        grid = np.asarray(ns["rho"]).ravel()

        def coeff_space():
            solver.evaluator.require_coeff_space(state)

        def constraint_norm():
            value = constraint_operator.evaluate()
            value.change_scales(1)
            data = np.asarray(value["g"])
            absolute = float(np.max(np.abs(data)))
            coeff_space()
            return absolute

        du = derivative(u).evaluate()
        du.change_scales(1)
        scale = float(np.max(np.abs(ns["initial_psi"]))) + float(
            np.max(np.abs(du["g"]))
        )
        report.update(
            initial_constraint_scale=scale,
            history_dt=history_dt,
            stage_steps=stage_steps,
            normalized_definition="C_hat=||psi-D_rho u||_inf/S0",
            S0_definition="||psi(0)||_inf+||D_rho u(0)||_inf",
            maxima_are_sampled=True,
            ncc_cutoff=kwargs.get("ncc_cutoff"),
            entry_cutoff=kwargs.get("entry_cutoff"),
            formulation="shared_flux" if characteristic else "standard",
            implementation_sha256={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), Path(__file__).with_name("sds_solver.py"))
            },
        )

        coeff_space()
        initial_coefficients = [field["c"].copy() for field in state]
        subproblems = [sp for sp in solver.subproblems if sp.size]
        if len(subproblems) != 1:
            raise RuntimeError("This diagnostic expects one serial radial subproblem")
        sp = subproblems[0]
        mass_solver = splu(sp.M_min.tocsc())

        def save_state():
            coeff_space()
            return sp.gather_inputs(state).copy()

        def put_state(vector):
            for field in state:
                field.preset_layout("c")
            sp.scatter_inputs(vector, state)

        def audit_rhs(label):
            x = save_state()
            constraint = constraint_norm()
            implicit = mass_solver.solve(-sp.L_min @ x)
            put_state(x)
            solver.evaluator.evaluate_group("F")
            explicit = mass_solver.solve(sp.gather_outputs(solver.F).copy())
            row = {"datum": label, "initial_constraint_linf": constraint}
            for name, vector in (
                ("implicit", implicit), ("explicit", explicit),
                ("complete", implicit + explicit),
            ):
                put_state(vector)
                row[name + "_constraint_rhs_linf"] = constraint_norm()
            report["rhs"].append(row)
            put_state(x)

        audit_rhs("actual_compact_areal_bump")
        for field in state:
            field.change_scales(1)
        u["g"] = grid**2 * (1 - grid)**2
        psi = derivative(u).evaluate()
        psi.change_scales(1)
        psi_data = np.asarray(psi["g"]).ravel().copy()
        ns["coefficient_b"].change_scales(1)
        pi_data = -np.asarray(ns["coefficient_b"]["g"]).ravel() * psi_data
        if characteristic:
            state[1]["g"] = pi_data + psi_data
            state[2]["g"] = pi_data - psi_data
        else:
            state[1]["g"] = psi_data
            state[2]["g"] = pi_data
        audit_rhs("constraint_satisfying_polynomial")
        for field, values in zip(state, initial_coefficients):
            field["c"] = values

        def sample(tau, destination, **extra):
            absolute = constraint_norm()
            destination.append({
                "tau": float(tau), "constraint_linf": absolute,
                "constraint_fixed_scale": absolute / scale,
                **extra,
            })

        sample(solver.sim_time, report["history"])
        original_scatter = sp.scatter_inputs
        stage_counter = 0
        active_step = {"start": 0.0, "dt": 0.0, "stage": 0}

        def scatter(vector, fields):
            nonlocal stage_counter
            original_scatter(vector, fields)
            if solver.iteration < stage_steps and tuple(fields) == state:
                stage_counter += 1
                active_step["stage"] += 1
                actual_stage_time = active_step["start"] + active_step["dt"] * solver.timestepper.c[active_step["stage"]]
                sample(actual_stage_time, report["stages"],
                       iteration=int(solver.iteration),
                       stage_scatter=stage_counter)

        sp.scatter_inputs = scatter
        original_step = solver.step
        last_history = float(solver.sim_time)

        def step(dt):
            nonlocal last_history
            active_step.update(start=float(solver.sim_time), dt=dt, stage=0)
            original_step(dt)
            if (solver.sim_time - last_history >= history_dt - 1e-10
                    or solver.sim_time >= solver.stop_sim_time - 1e-10):
                sample(solver.sim_time, report["history"])
                last_history = float(solver.sim_time)

        solver.step = step
        return solver

    d3.IVP.build_solver = build
    try:
        yield report
    finally:
        d3.IVP.build_solver = original_build


def run(args):
    from .exterior_sds_model import ExteriorSdSParameters, retarded_time_offset
    from .regulator_suite import flat_initial_data
    from .sds_solver import SdSNumericalParameters, run_exterior_sds_simulation

    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    parameters = ExteriorSdSParameters(
        mass=1.0, cosmological_length=args.length, ell=2,
        curvature_coupling=0.0,
    )
    offset = retarded_time_offset(parameters, 4.0)
    numerical = SdSNumericalParameters(
        resolution=args.resolution, timestep=args.dt,
        end_time=args.end_tau if args.end_tau is not None else args.end_u + offset,
        signal_dt=args.signal_dt, snapshot_dt=args.snapshot_dt,
        observers=(1.0,), timestepper="RK222", bridge="minimal", dealias=1.5,
    )
    start = time.perf_counter()
    with instrument_solver(history_dt=args.constraint_dt,
                           ncc_cutoff=args.ncc_cutoff,
                           entry_cutoff=args.entry_cutoff) as audit:
        result = run_exterior_sds_simulation(
            parameters, flat_initial_data(), numerical, explicit_potential=True,
            conservative_characteristic_variables=args.formulation == "shared_flux",
        )
    audit["wall_seconds_with_audit"] = time.perf_counter() - start
    result.metadata["quadrupole_constraint_audit"] = audit
    result.metadata["retarded_time_offset"] = {
        "q": float(offset), "reference_radius": 4.0,
        "definition": "lim_(r->r_c)(h_chi+r_*chi)",
    }
    result.save(destination)
    summary = {key: value for key, value in audit.items() if key not in {"history", "stages"}}
    summary["sampled_maximum_constraint_linf"] = max(row["constraint_linf"] for row in audit["history"])
    summary["sampled_maximum_constraint_fixed_scale"] = max(row["constraint_fixed_scale"] for row in audit["history"])
    summary["stage_maximum_constraint_linf"] = max(row["constraint_linf"] for row in audit["stages"])
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, choices=(320.0, 640.0), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--end-u", type=float, default=80.0)
    parser.add_argument("--end-tau", type=float)
    parser.add_argument("--signal-dt", type=float, default=0.01)
    parser.add_argument("--constraint-dt", type=float, default=0.1)
    parser.add_argument("--snapshot-dt", type=float, default=5.0)
    parser.add_argument("--formulation", choices=("standard", "shared_flux"), default="shared_flux")
    parser.add_argument("--ncc-cutoff", type=float)
    parser.add_argument("--entry-cutoff", type=float)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
