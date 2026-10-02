"""High-precision continuous SBP spectral elements for scalar-wave benchmarks.

This is an independent experimental solver, not a replacement for frozen
production.  On each Legendre--Gauss--Lobatto element HD + D^T H = E.
The continuous assembly identifies field and velocity at element endpoints;
the weak form couples physical fluxes without a reduction variable or an
artificial interface boundary condition.  For stationary coefficients,

    M u_tt = C u_t - K u,
    M = sum H/A,  K = sum (D^T H p D + H P),
    C = sum (H B D - D^T H B) + diag(-B_left, ..., B_right).

Consequently E_t = -v_left^2-v_right^2 at the two future null boundaries.
Positivity of E additionally requires positive semidefinite K.  A separate
C++ backend integrates this linear system using double-double Hermite4 or
L-stable Radau IIA of order three. The latter damps unresolved stiff grid
modes, which Hermite4 can leave almost undamped on small spectral elements.
All geometry, nodes, quadrature and matrix assembly use mpmath precision;
decimal coefficients pass to the backend without a float64 conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parents[1]


def legendre_pair(n: int, x):
    """Return P_n(x), P_{n-1}(x) without floating-point constants."""
    previous, current = mp.mpf(1), mp.mpf(x)
    if n == 0:
        return previous, mp.mpf(0)
    for k in range(1, n):
        previous, current = current, ((2*k+1)*x*current-k*previous)/(k+1)
    return current, previous


def lobatto(degree: int):
    """LGL nodes, positive diagonal norm, and its SBP differentiation matrix."""
    if degree < 2:
        raise ValueError("Polynomial degree must be at least two")
    nodes = [mp.mpf(-1)]
    for i in range(1, degree):
        x = -mp.cos(mp.pi*i/degree)
        for _ in range(100):
            pn, pm = legendre_pair(degree, x)
            change = (x*pn-pm)/((degree+1)*pn)
            x -= change
            if abs(change) < 100*mp.eps:
                break
        else:
            raise ArithmeticError("Lobatto node did not converge")
        nodes.append(x)
    nodes.append(mp.mpf(1))
    pn = [legendre_pair(degree, x)[0] for x in nodes]
    weights = [mp.mpf(2)/(degree*(degree+1)*v*v) for v in pn]
    derivative = mp.matrix(degree+1)
    for i in range(degree+1):
        for j in range(degree+1):
            if i != j:
                derivative[i,j] = pn[i]/(pn[j]*(nodes[i]-nodes[j]))
    derivative[0,0] = -mp.mpf(degree*(degree+1))/4
    derivative[degree,degree] = -derivative[0,0]
    return nodes, weights, derivative


def standard_boundaries(geometry, layout: str = "standard"):
    """Resolve compact data separately from the transition and horizon cap.

    Artificial block boundaries are also used for the controls, allowing the
    same interface treatment to be tested where no physical layer exists.
    The last two boundaries are the paper's fixed-angle transition values.
    """
    if layout not in ("standard", "shifted", "split-layer", "wave-resolved"):
        raise ValueError("Unknown block layout")
    outer = [mp.mpf("0.9"), mp.mpf("0.97")]
    interior = [mp.mpf("0.55"), mp.mpf("0.70")]
    if layout == "shifted":
        outer = [mp.mpf("0.92"), mp.mpf("0.98")]
        interior = [mp.mpf("0.51"), mp.mpf("0.72")]
    # These are computation boundaries, not a modified cosmological profile.
    layer = [mp.mpf("0.9943459581991452"),
             mp.mpf("0.9985844858695327")]
    if geometry.background == "exterior":
        layer = [geometry.rho0, geometry.rho1]
    extra = []
    if layout == "wave-resolved":
        # The incoming wave scattered by the transition is short in areal
        # radius, although the outgoing wave is smooth in rho. Resolve the
        # pretransition block as well as the coefficients in the layer.
        for radius in (80,120,160,200):
            rho = geometry.compact_radius(radius*geometry.mass)
            if outer[-1] < rho < layer[0]:
                extra.append(rho)
    if layout in ("split-layer", "wave-resolved"):
        left, right = layer
        layer = [left + (right-left)*i/4 for i in range(5)]
    boundaries = [mp.mpf(0), geometry.support_rho[0],
                  *interior, geometry.support_rho[1],
                  *outer, *extra, *layer, mp.mpf(1)]
    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0 or boundaries[-1] != 1:
        raise ValueError("Domain endpoints must be zero and one")
    return boundaries


def assemble(geometry, boundaries, degree: int):
    """Assemble banded real matrices and initial state at current mp precision."""
    reference, weights, derivative = lobatto(degree)
    elements = len(boundaries)-1
    count = elements*degree+1
    mass = [mp.mpf(0) for _ in range(count)]
    grid = [mp.mpf(0) for _ in range(count)]
    stiffness = [[mp.mpf(0) for _ in range(2*degree+1)] for _ in range(count)]
    transport = [[mp.mpf(0) for _ in range(2*degree+1)] for _ in range(count)]
    sbp_defect = mp.mpf(0)
    for i in range(degree+1):
        for j in range(degree+1):
            boundary = (-1 if i == 0 else 1 if i == degree else 0) if i == j else 0
            sbp_defect = max(sbp_defect, abs(weights[i]*derivative[i,j]
                             + weights[j]*derivative[j,i]-boundary))
    for element, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        jacobian = (right-left)/2
        if jacobian <= 0:
            raise ValueError("Block boundaries must increase strictly")
        local = [(left+right)/2+jacobian*x for x in reference]
        local[0], local[-1] = left, right
        coefs = [geometry.coefficients(x) for x in local]
        weighted_derivative = mp.matrix(degree+1)
        for i, (a, boost, p, potential) in enumerate(coefs):
            if a <= 0 or p < -1000*mp.eps or not all(mp.isfinite(x) for x in (a, boost,p,potential)):
                raise ValueError(f"Invalid coefficient at rho={local[i]}")
            for j in range(degree+1):
                weighted_derivative[i,j] = weights[i]*p/jacobian*derivative[i,j]
        local_k = derivative.T*weighted_derivative
        for i, (a, boost, p, potential) in enumerate(coefs):
            row = element*degree+i
            grid[row] = local[i]
            mass[row] += weights[i]*jacobian/a
            local_k[i,i] += weights[i]*jacobian*potential
            for j in range(degree+1):
                column = element*degree+j
                slot = column-row+degree
                stiffness[row][slot] += local_k[i,j]
                transport[row][slot] += (weights[i]*boost*derivative[i,j]
                                          - derivative[j,i]*weights[j]*coefs[j][1])
    transport[0][degree] -= geometry.coefficients(mp.mpf(0))[1]
    transport[-1][degree] += geometry.coefficients(mp.mpf(1))[1]
    initial = [geometry.initial(x) for x in grid]
    energy_defect = mp.mpf(0)
    stiffness_defect = mp.mpf(0)
    for i in range(count):
        for j in range(max(0,i-degree), min(count,i+degree+1)):
            boundary = -2 if i == j and i in (0,count-1) else 0
            energy_defect = max(energy_defect,
                abs(transport[i][j-i+degree]+transport[j][i-j+degree]-boundary))
            stiffness_defect = max(stiffness_defect,
                abs(stiffness[i][j-i+degree]-stiffness[j][i-j+degree]))
    return dict(degree=degree, count=count, grid=grid, mass=mass,
                stiffness=stiffness, transport=transport, initial=initial,
                boundaries=boundaries, sbp_defect=sbp_defect,
                energy_defect=energy_defect, stiffness_defect=stiffness_defect)


def dense_matrices(operator):
    """Float64 copies for diagnostic eigenvalues/tests only, never evolution."""
    count, degree = operator["count"], operator["degree"]
    k, c = np.zeros((count,count)), np.zeros((count,count))
    for i in range(count):
        for j in range(max(0,i-degree),min(count,i+degree+1)):
            k[i,j] = float(operator["stiffness"][i][j-i+degree])
            c[i,j] = float(operator["transport"][i][j-i+degree])
    return np.array([float(x) for x in operator["mass"]]), k, c


def decimal(value):
    return mp.nstr(value, n=mp.mp.dps, strip_zeros=False)


def write_input(operator, path: Path):
    """Write one exact-decimal banded input, refusing an existing path."""
    count, degree = operator["count"], operator["degree"]
    with path.open("x") as stream:
        stream.write(f"{count} {degree}\n")
        for x, (u,v) in zip(operator["grid"],operator["initial"]):
            stream.write(" ".join(map(decimal,(x,u,v)))+"\n")
        for i in range(count):
            for j in range(max(0,i-degree),min(count,i+degree+1)):
                slot = j-i+degree
                stream.write(decimal(operator["stiffness"][i][slot]/operator["mass"][i])
                             +" "+decimal(operator["transport"][i][slot]/operator["mass"][i])+"\n")


def prepare(background: str, length: str, coupling: str, degree: int,
            destination: Path, layout="standard", dps=50, ell=1,
            initial_data="velocity"):
    from .sbp_geometry import Geometry
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    start = time.perf_counter()
    with mp.workdps(dps):
        geometry = Geometry(background, None if background == "schwarzschild" else length,
                            ell=ell, xi=coupling,dps=dps,initial_data=initial_data)
        boundaries = standard_boundaries(geometry, layout)
        operator = assemble(geometry,boundaries,degree)
        write_input(operator,destination/"operator.txt")
        mass,k,_ = dense_matrices(operator)
        low_eigenvalues = eigh(k,np.diag(mass),subset_by_index=(0,min(3,len(mass)-1)),eigvals_only=True)
        q = geometry.clock_offset()
        metadata = dict(background=background,
                        L_over_M=None if background == "schwarzschild" else length,
                        xi=coupling,ell=ell,
                        mass="1",degree=degree,nodes=operator["count"],layout=layout,
                        block_boundaries=list(map(decimal,boundaries)),
                        precision_digits=dps,q=decimal(q),
                        sbp_identity_defect=decimal(operator["sbp_defect"]),
                        energy_identity_defect=decimal(operator["energy_defect"]),
                        stiffness_symmetry_defect=decimal(operator["stiffness_defect"]),
                        stiffness_generalized_low_eigenvalues_float64=low_eigenvalues.tolist(),
                        formulation="continuous LGL SBP weak scalar u,v",
                        initial_data_kind=initial_data,
                        initial_data=geometry.metadata()["initial_data"],
                        geometry=geometry.metadata(),
                        time_alignment="U=tau-q; no fitted shift or rescaling",
                        preparation_seconds=time.perf_counter()-start)
        sources = [Path(__file__),Path(__file__).with_name("sbp_geometry.py")]
        metadata["source_sha256"] = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        metadata["operator_sha256"] = hashlib.sha256((destination/"operator.txt").read_bytes()).hexdigest()
        (destination/"configuration.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def run(prepared: Path, output: Path, backend: Path, dt: str, end_u: str,
        output_every=1, timeout=600, integrator="hermite"):
    if integrator not in ("hermite", "radau3"):
        raise ValueError("Unknown time integrator")
    configuration = json.loads((prepared/"configuration.json").read_text())
    # Run launches may be threaded; do not mutate mpmath's global context.
    clock_context = mp.mp.clone()
    clock_context.dps = 50
    end_tau = clock_context.nstr(clock_context.mpf(end_u)
                                + clock_context.mpf(configuration["q"]),
                                50, strip_zeros=False)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    command = [str(backend.resolve()),str((prepared/"operator.txt").resolve()),
               str(output.resolve()),dt,end_tau,str(output_every),str(timeout),
               "--integrator",integrator]
    backend_hash = hashlib.sha256(backend.read_bytes()).hexdigest()
    start = time.perf_counter()
    process = subprocess.run(command,text=True,capture_output=True,timeout=timeout+30)
    if output.exists():
        (output/"run.log").write_text(process.stdout+process.stderr)
        configuration.update(dt=dt,end_u=end_u,end_tau=end_tau,output_every=output_every,
                             integrator=integrator,
                             wall_seconds=time.perf_counter()-start,command=command,
                             backend_sha256=backend_hash,
                             returncode=process.returncode)
        (output/"configuration.json").write_text(json.dumps(configuration,indent=2)+"\n")
    if process.returncode:
        raise RuntimeError(process.stdout+process.stderr)
    print(process.stdout,flush=True)
    return configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--background",choices=("schwarzschild","uniform","exterior"),required=True)
    prep.add_argument("--length",default="640")
    prep.add_argument("--coupling",default="0")
    prep.add_argument("--ell",type=int,default=1)
    prep.add_argument("--initial-data",choices=("velocity","displacement"),default="velocity")
    prep.add_argument("--degree",type=int,required=True)
    prep.add_argument("--layout",choices=("standard","shifted","split-layer","wave-resolved"),default="standard")
    prep.add_argument("--dps",type=int,default=50)
    prep.add_argument("--destination",type=Path,required=True)
    evolve = sub.add_parser("run")
    evolve.add_argument("--prepared",type=Path,required=True)
    evolve.add_argument("--output",type=Path,required=True)
    evolve.add_argument("--backend",type=Path,required=True)
    evolve.add_argument("--dt",default="0.1")
    evolve.add_argument("--end-u",default="500")
    evolve.add_argument("--output-every",type=int,default=1)
    evolve.add_argument("--timeout",type=int,default=600)
    evolve.add_argument("--integrator",choices=("hermite","radau3"),default="hermite")
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare(args.background,args.length,args.coupling,args.degree,
                         args.destination,args.layout,args.dps,args.ell,args.initial_data)
    else:
        result = run(args.prepared,args.output,args.backend,args.dt,args.end_u,
                     args.output_every,args.timeout,args.integrator)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
