"""Staged stability tests for the exterior-supported dipole tail.

This is a diagnostic campaign, separate from every production archive.  It
compares the standard ``(u,psi,pi)`` reduction, the shared-flux conservative
``(u,h,j)`` repair, and the endpoint-factored ``(u,H,J)`` reduction at several
continuum-equivalent constraint-damping rates.  The first stage stops at
``U/M=150`` and records constraints every ``5M``; no long evolution is
accepted until the high-resolution ladder passes explicit amplitude,
constraint, and late-growth gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .exterior_sds_model import (
    ExteriorSdSParameters,
    background_audit,
    compact_radius,
    retarded_time_offset,
)
from .regulator_suite import _reserve_destination, _write_once
from .sds_model import ArealVelocityBumpInitialData


OUTPUT_ROOT = Path("results/exterior_tail_repair_v1")
LENGTH = 640.0
ELL = 1
END_U = 150.0
REFERENCE_RADIUS = 4.0
FINITE_OBSERVERS = (8.0, 16.0)
RESOLUTIONS = (1536, 2048, 3072)
TIMESTEP = 0.0025
HALVED_TIMESTEP = 0.00125
COUPLINGS = (0.0, 1.0 / 6.0)
INITIAL_DATA = ArealVelocityBumpInitialData(
    center_radius=6.0,
    support_half_width=3.0,
    amplitude=1.0,
)


@dataclass(frozen=True)
class RepairCase:
    """One short, isolated formulation test."""

    formulation: str
    curvature_coupling: float
    resolution: int
    timestep: float = TIMESTEP

    def __post_init__(self) -> None:
        if self.formulation not in {
            "standard",
            "conservative",
            "factored_gamma0p5",
            "factored_gamma1",
            "factored_gamma2",
        }:
            raise ValueError(f"Unknown repair formulation {self.formulation!r}.")
        if self.curvature_coupling not in COUPLINGS:
            raise ValueError("Repair tests use only xi=0 and xi=1/6.")
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {RESOLUTIONS}.")
        if self.timestep not in (TIMESTEP, HALVED_TIMESTEP):
            raise ValueError("Unsupported repair-test timestep.")

    @property
    def coupling_label(self) -> str:
        return "xi0" if self.curvature_coupling == 0.0 else "xi1o6"

    @property
    def damping(self) -> float:
        return {
            "standard": 0.0,
            "conservative": 0.0,
            "factored_gamma0p5": 0.5,
            "factored_gamma1": 1.0,
            "factored_gamma2": 2.0,
        }[self.formulation]

    @property
    def factored(self) -> bool:
        return self.formulation.startswith("factored_")

    @property
    def conservative(self) -> bool:
        return self.formulation == "conservative"

    @property
    def name(self) -> str:
        step = str(self.timestep).replace(".", "p")
        return (
            f"repair_{self.formulation}_{self.coupling_label}_"
            f"N{self.resolution}_dt{step}"
        )


def cases() -> tuple[RepairCase, ...]:
    """Return the primary ladders, timestep checks, and damping sweep."""

    values: list[RepairCase] = []
    for coupling in COUPLINGS:
        for formulation in ("standard", "conservative", "factored_gamma1"):
            for resolution in RESOLUTIONS:
                values.append(RepairCase(formulation, coupling, resolution))
            values.append(
                RepairCase(formulation, coupling, 2048, HALVED_TIMESTEP)
            )
        for formulation in ("factored_gamma0p5", "factored_gamma2"):
            values.append(RepairCase(formulation, coupling, 2048))
    return tuple(values)


def case_catalogue() -> dict[str, RepairCase]:
    catalogue = {case.name: case for case in cases()}
    if len(catalogue) != len(cases()):
        raise RuntimeError("Exterior-tail repair case names are not unique.")
    return catalogue


def archive_path(output_dir: Path, case: RepairCase) -> Path:
    step = str(case.timestep).replace(".", "p")
    return (
        Path(output_dir)
        / "raw"
        / case.formulation
        / case.coupling_label
        / f"N{case.resolution}_dt{step}.npz"
    )


def model(case: RepairCase) -> ExteriorSdSParameters:
    return ExteriorSdSParameters(
        mass=1.0,
        cosmological_length=LENGTH,
        ell=ELL,
        curvature_coupling=case.curvature_coupling,
    )


def observer_coordinates(parameters: ExteriorSdSParameters) -> tuple[float, ...]:
    finite = compact_radius(np.asarray(FINITE_OBSERVERS), parameters)
    return tuple(float(value) for value in finite) + (1.0,)


def stability_audit(result, offset: float) -> dict[str, float | bool]:
    """Apply hard failure gates that the earlier finite-only audit lacked."""

    signal_times = np.asarray(result.signal_times, dtype=float) - offset
    signals = np.asarray(result.signals, dtype=float)
    snapshots = np.asarray(result.u_snapshots, dtype=float)
    constraints = np.asarray(result.constraint_linf, dtype=float)
    snapshot_times = np.asarray(result.snapshot_times, dtype=float) - offset
    finite = bool(
        signal_times.size
        and signals.size
        and snapshots.size
        and constraints.size
        and np.all(np.isfinite(signal_times))
        and np.all(np.isfinite(signals))
        and np.all(np.isfinite(snapshots))
        and np.all(np.isfinite(constraints))
    )
    late = signal_times >= 60.0
    final_window = signal_times >= END_U - 20.0
    preceding_window = (
        (signal_times >= END_U - 40.0) & (signal_times < END_U - 20.0)
    )
    late_signal = float(np.max(np.abs(signals[late]))) if np.any(late) else np.inf
    final_rms = float(np.sqrt(np.mean(signals[final_window, -1] ** 2)))
    preceding_rms = float(np.sqrt(np.mean(signals[preceding_window, -1] ** 2)))
    growth_ratio = final_rms / max(preceding_rms, np.finfo(float).tiny)
    maximum_constraint = float(np.max(np.abs(constraints)))
    final_constraints = constraints[snapshot_times >= END_U - 10.0]
    final_constraint = (
        float(np.max(np.abs(final_constraints)))
        if final_constraints.size
        else np.inf
    )
    late_snapshots = snapshots[snapshot_times >= 60.0]
    maximum_late_snapshot = (
        float(np.max(np.abs(late_snapshots)))
        if late_snapshots.size
        else np.inf
    )
    passed = bool(
        finite
        and late_signal < 0.05
        and maximum_late_snapshot < 0.1
        and maximum_constraint < 0.05
        and final_constraint < 0.01
        and growth_ratio < 5.0
    )
    return {
        "finite": finite,
        "maximum_abs_signal_after_U60": late_signal,
        "maximum_abs_snapshot_after_U60": maximum_late_snapshot,
        "maximum_constraint_linf": maximum_constraint,
        "final_window_constraint_linf": final_constraint,
        "outer_rms_U110_130": preceding_rms,
        "outer_rms_U130_150": final_rms,
        "late_outer_rms_growth_ratio": growth_ratio,
        "signal_limit": 0.05,
        "late_snapshot_limit": 0.1,
        "constraint_limit": 0.05,
        "final_constraint_limit": 0.01,
        "growth_ratio_limit": 5.0,
        "passed": passed,
    }


def run_case(output_dir: Path, name: str) -> Path:
    catalogue = case_catalogue()
    if name not in catalogue:
        raise ValueError(f"Unknown exterior-tail repair case {name!r}.")
    case = catalogue[name]
    destination = archive_path(output_dir, case)
    reservation = _reserve_destination(destination, case.name)
    parameters = model(case)
    background = background_audit(parameters)
    required = (
        background["finite_coefficients"]
        and background["positive_interior_lapse"]
        and background["spacelike_bridge_interior"]
    )
    if not required:
        raise ValueError(f"Exterior background audit failed: {background}")

    from .sds_solver import SdSNumericalParameters, run_exterior_sds_simulation

    offset = float(retarded_time_offset(parameters, REFERENCE_RADIUS))
    numerical = SdSNumericalParameters(
        resolution=case.resolution,
        timestep=case.timestep,
        end_time=END_U + offset,
        signal_dt=0.05,
        snapshot_dt=5.0,
        observers=observer_coordinates(parameters),
        timestepper="RK222",
        bridge="minimal",
        dealias=1.5,
    )
    result = run_exterior_sds_simulation(
        parameters,
        INITIAL_DATA,
        numerical,
        explicit_potential=True,
        endpoint_factored_characteristic_variables=case.factored,
        conservative_characteristic_variables=case.conservative,
        characteristic_constraint_damping=case.damping,
    )
    audit = stability_audit(result, offset)
    result.metadata["retarded_time_offset"] = {
        "q": offset,
        "definition": "lim_(r->r_c)(h_chi+r_*chi)",
        "evaluation": "endpoint-safe numerical quadrature",
    }
    result.metadata["exterior_tail_repair"] = {
        **asdict(case),
        "case": case.name,
        "length_over_M": LENGTH,
        "ell": ELL,
        "end_U_over_M": END_U,
        "purpose": "short formulation-stability and equivalence test",
        "production_evidence": False,
    }
    result.metadata["background_audit"] = background
    result.metadata["stability_audit"] = audit
    published = _write_once(result, destination, reservation)
    print(json.dumps({"case": case.name, **audit}, sort_keys=True))
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    catalogue = case_catalogue()
    if not arguments.cases:
        for name in catalogue:
            print(name)
        return
    for name in arguments.cases:
        print(run_case(arguments.output_dir, name))


if __name__ == "__main__":
    main()
