"""
Solve the 1D periodic wave equation with Dedalus.

Equation:
    u_tt = c^2 u_xx,  x in [0, 2*pi)

The second-order equation is written as a first-order IVP:
    dt(u) = v
    dt(v) = c^2 dx(dx(u))

Run from WSL with:
    /home/govind/miniforge3/bin/mamba run -n dedalus3 python examples/wave_equation_1d.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import dedalus.public as d3


def main():
    # Numerical parameters.
    lx = 2 * np.pi
    nx = 128
    c = 1.0
    mode = 3
    stop_sim_time = 2 * np.pi
    timestep = 1e-3
    snapshot_cadence = 50
    dtype = np.float64

    # Domain and fields.
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.RealFourier(xcoord, size=nx, bounds=(0, lx), dealias=3 / 2)
    u = dist.Field(name="u", bases=xbasis)
    v = dist.Field(name="v", bases=xbasis)
    dx = lambda a: d3.Differentiate(a, xcoord)

    # Initial data: a single standing-wave mode with zero initial velocity.
    x = dist.local_grid(xbasis)
    u["g"] = np.sin(mode * x)
    v["g"] = 0

    # Dedalus IVP formulation.
    problem = d3.IVP([u, v], namespace=locals())
    problem.add_equation("dt(u) - v = 0")
    problem.add_equation("dt(v) - c**2 * dx(dx(u)) = 0")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    # Store snapshots for a space-time plot.
    u_snapshots = [u["g", 1].copy()]
    t_snapshots = [solver.sim_time]
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % snapshot_cadence == 0:
            u_snapshots.append(u["g", 1].copy())
            t_snapshots.append(solver.sim_time)

    u_array = np.array(u_snapshots)
    t_array = np.array(t_snapshots)
    analytic_final = np.sin(mode * x) * np.cos(c * mode * solver.sim_time)
    max_error = np.max(np.abs(u["g"] - analytic_final))

    output_dir = Path("results/flat_wave")
    output_dir.mkdir(exist_ok=True)

    np.savetxt(
        output_dir / "wave_equation_1d_final.csv",
        np.column_stack((x.ravel(), u["g"].ravel(), analytic_final.ravel())),
        delimiter=",",
        header="x,numerical_u,analytic_u",
        comments="",
    )

    plt.figure(figsize=(7, 4))
    plt.pcolormesh(x.ravel(), t_array, u_array, shading="auto", cmap="RdBu_r")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("1D wave equation: u_tt = c^2 u_xx")
    plt.tight_layout()
    plt.savefig(output_dir / "wave_equation_1d.png", dpi=200)
    plt.close()

    print(f"Dedalus wave equation run complete")
    print(f"iterations: {solver.iteration}")
    print(f"final simulation time: {solver.sim_time:.6f}")
    print(f"max error vs analytic standing wave: {max_error:.3e}")
    print(f"saved plot: {output_dir / 'wave_equation_1d.png'}")
    print(f"saved final data: {output_dir / 'wave_equation_1d_final.csv'}")


if __name__ == "__main__":
    main()
