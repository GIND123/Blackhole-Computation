"""Matched production runs for minimal and conformal scalar coupling.

The short-waveform group adds the missing conformally coupled uniform and
exterior-supported SdS sequences.  The tail group is self-contained: it
regenerates the Schwarzschild and minimally coupled controls together with the
conformally coupled cases, using identical clocks, initial data, refinement
ladders, and (within each background family) evolution formulations.

Every case has an isolated write-once archive and optional checkpoint.  A
campaign source hash supplied through ``SDS_CAMPAIGN_SOURCE_SHA256`` is copied
into each archive so an uncommitted but frozen source bundle remains exactly
identifiable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct

from .exterior_sds_model import (
    ExteriorSdSParameters,
    areal_radius as exterior_areal_radius,
    background_audit,
    chebyshev_angle,
    compact_radius as exterior_compact_radius,
    rescaled_scalar_potential as exterior_potential,
    retarded_time_offset as exterior_retarded_time_offset,
    transition_compact_radii,
)
from .regulator_suite import LEVELS, _reserve_destination, _write_once
from .reproducibility import reproducibility_metadata
from .sds_model import (
    ArealBumpInitialData,
    ArealVelocityBumpInitialData,
    SdSParameters,
    compact_radius as uniform_compact_radius,
    rescaled_scalar_potential as uniform_potential,
    retarded_time_offset as uniform_retarded_time_offset,
)
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    retarded_time_offset as schwarzschild_retarded_time_offset,
)


OUTPUT_ROOT = Path("results/curvature_coupling_production_v2")
LENGTHS = (80, 160, 320, 640)
CONFORMAL_COUPLING = 1.0 / 6.0
REFERENCE_RADIUS = 4.0
QNM_INITIAL_DATA = ArealBumpInitialData(
    center_radius=4.0,
    support_half_width=1.5,
    time_symmetric=True,
    pi_amplitude=0.0,
)
TAIL_INITIAL_DATA = ArealVelocityBumpInitialData(
    center_radius=6.0,
    support_half_width=3.0,
    amplitude=1.0,
)
QNM_TIMESTEPS = {
    "coarse": 0.005,
    "medium": 0.00375,
    "fine": 0.0025,
}
UNIFORM_QNM_RESOLUTIONS = {
    length: {"coarse": 384, "medium": 512, "fine": 768}
    for length in LENGTHS
}
EXTERIOR_QNM_RESOLUTIONS = {
    80: {"coarse": 384, "medium": 512, "fine": 768},
    160: {"coarse": 768, "medium": 1024, "fine": 1536},
    320: {"coarse": 1024, "medium": 1536, "fine": 2048},
    640: {"coarse": 768, "medium": 1024, "fine": 1536},
}
QNM_END_TIMES = {"uniform": 200.0, "exterior": 100.0}
TAIL_LENGTH = 640.0
TAIL_END_U = 1000.0
TAIL_SETTINGS = (
    (1536, 0.0025),
    (2048, 0.0025),
    (3072, 0.0025),
    (2048, 0.00125),
)
FINITE_TAIL_OBSERVERS = (8.0, 16.0)
PREFLIGHT_LIMITS = {"coarse": 0.06, "medium": 0.03, "fine": 0.01}


@dataclass(frozen=True)
class CouplingCase:
    """One immutable production calculation."""

    group: str
    background: str
    curvature_coupling: float
    ell: int
    length: float | None
    resolution: int
    timestep: float
    end_time_or_u: float
    level: str | None = None

    @property
    def coupling_label(self) -> str:
        if self.curvature_coupling == 0.0:
            return "xi0"
        if self.curvature_coupling == CONFORMAL_COUPLING:
            return "xi1o6"
        raise ValueError("Only xi=0 and xi=1/6 belong to this campaign.")

    @property
    def name(self) -> str:
        if self.group == "qnm":
            return (
                f"qnm_{self.background}_{self.coupling_label}_"
                f"L{self.length:g}_{self.level}"
            )
        step = str(self.timestep).replace(".", "p")
        length = "" if self.length is None else f"_L{self.length:g}"
        return (
            f"tail_{self.background}_{self.coupling_label}{length}_"
            f"N{self.resolution}_dt{step}"
        )


def qnm_cases() -> tuple[CouplingCase, ...]:
    """Return the conformal ladders and the ``L/M=80`` spatial check."""

    cases: list[CouplingCase] = []
    for background, resolutions in (
        ("uniform", UNIFORM_QNM_RESOLUTIONS),
        ("exterior", EXTERIOR_QNM_RESOLUTIONS),
    ):
        for length in LENGTHS:
            for level in LEVELS:
                cases.append(
                    CouplingCase(
                        group="qnm",
                        background=background,
                        curvature_coupling=CONFORMAL_COUPLING,
                        ell=2,
                        length=float(length),
                        resolution=resolutions[length][level],
                        timestep=QNM_TIMESTEPS[level],
                        end_time_or_u=QNM_END_TIMES[background],
                        level=level,
                    )
                )
    # The conformal curvature layer is sharpest at L/M=80.  This fixed-step
    # fourth grid separates its spatial truncation error from the coupled
    # three-level production ladder.
    cases.append(
        CouplingCase(
            group="qnm",
            background="exterior",
            curvature_coupling=CONFORMAL_COUPLING,
            ell=2,
            length=80.0,
            resolution=1024,
            timestep=QNM_TIMESTEPS["fine"],
            end_time_or_u=QNM_END_TIMES["exterior"],
            level="verification",
        )
    )
    return tuple(cases)


def tail_cases() -> tuple[CouplingCase, ...]:
    """Return a fully matched 20-case tail package at ``L/M=640``."""

    cases: list[CouplingCase] = []
    families = (
        ("schwarzschild", 0.0, None),
        ("uniform", 0.0, TAIL_LENGTH),
        ("uniform", CONFORMAL_COUPLING, TAIL_LENGTH),
        ("exterior", 0.0, TAIL_LENGTH),
        ("exterior", CONFORMAL_COUPLING, TAIL_LENGTH),
    )
    for background, coupling, length in families:
        for resolution, timestep in TAIL_SETTINGS:
            cases.append(
                CouplingCase(
                    group="tail",
                    background=background,
                    curvature_coupling=coupling,
                    ell=1,
                    length=length,
                    resolution=resolution,
                    timestep=timestep,
                    end_time_or_u=TAIL_END_U,
                )
            )
    return tuple(cases)


def case_catalogue() -> dict[str, CouplingCase]:
    """Return every case keyed by its command-line name."""

    cases = qnm_cases() + tail_cases()
    catalogue = {case.name: case for case in cases}
    if len(catalogue) != len(cases):
        raise RuntimeError("Curvature-coupling case names are not unique.")
    return catalogue


def archive_path(output_dir: Path, case: CouplingCase) -> Path:
    """Return the isolated write-once archive for ``case``."""

    if case.group == "qnm":
        return (
            Path(output_dir)
            / "raw"
            / "qnm"
            / case.background
            / case.coupling_label
            / f"L{case.length:g}"
            / str(case.level)
            / "waveform.npz"
        )
    length = "schwarzschild" if case.length is None else f"L{case.length:g}"
    step = str(case.timestep).replace(".", "p")
    return (
        Path(output_dir)
        / "raw"
        / "tail"
        / case.background
        / case.coupling_label
        / length
        / f"N{case.resolution}_dt{step}.npz"
    )


def _contract(case: CouplingCase) -> dict:
    contract = {
        **asdict(case),
        "equation": "(Box_g-xi*R_g) Phi = 0",
        "reduced_field": "u=r*Phi",
        "gauge": "minimal",
        "timestepper": "RK222",
        "dealias": 1.5,
        "explicit_potential": case.group == "tail" or case.background == "exterior",
        "time_translation_fitted": False,
        "amplitude_rescaling_fitted": False,
    }
    if case.group == "qnm":
        contract.update(
            {
                "initial_data": QNM_INITIAL_DATA.as_dict(),
                "signal_dt": 0.03,
                "end_time_coordinate": "tau",
                "comparison_window": "0<=U/M<=80",
            }
        )
    else:
        contract.update(
            {
                "initial_data": TAIL_INITIAL_DATA.as_dict(),
                "signal_dt": 0.05,
                "end_time_coordinate": "retarded U=tau-q",
                "finite_observers_over_M": list(FINITE_TAIL_OBSERVERS),
                "outer_observer": (
                    "future null infinity"
                    if case.background == "schwarzschild"
                    else "future cosmological horizon"
                ),
                "endpoint_factored_characteristic_variables": (
                    False
                ),
                "conservative_characteristic_variables": (
                    case.background == "exterior"
                ),
                "characteristic_flux_discretization": (
                    "conservative_nested_endpoint_flux_v1"
                    if case.background == "exterior"
                    else None
                ),
                "characteristic_constraint_damping": 0.0,
            }
        )
    return contract


def contract_sha256(case: CouplingCase) -> str:
    encoded = json.dumps(
        _contract(case), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provenance(case: CouplingCase, output_dir: Path) -> dict:
    metadata = reproducibility_metadata()
    metadata.update(
        {
            "role": "production_simulation",
            "case": case.name,
            "physical_contract_sha256": contract_sha256(case),
            "campaign_id": os.environ.get("SDS_CAMPAIGN_ID"),
            "campaign_source_sha256": os.environ.get(
                "SDS_CAMPAIGN_SOURCE_SHA256"
            ),
            "command": (
                "python -m black_hole.curvature_coupling_production "
                f"{case.name} --output-dir {Path(output_dir).as_posix()}"
            ),
        }
    )
    return metadata


def exterior_spectral_preflight(
    case: CouplingCase, *, dense_count: int = 20_001
) -> dict[str, float | int | bool]:
    """Measure representation error for the signed transition potential."""

    if case.background != "exterior" or case.length is None:
        raise ValueError("Exterior preflight requires an exterior finite-L case.")
    parameters = ExteriorSdSParameters(
        mass=1.0,
        cosmological_length=case.length,
        ell=case.ell,
        curvature_coupling=case.curvature_coupling,
    )
    rho0, rho1 = transition_compact_radii(parameters)
    theta0 = float(chebyshev_angle(np.array(rho0)))
    theta1 = float(chebyshev_angle(np.array(rho1)))
    theta_nodes = np.pi * (
        np.arange(case.resolution, dtype=float) + 0.5
    ) / case.resolution
    x_nodes = np.cos(theta_nodes)
    rho_nodes = 0.5 * (1.0 + x_nodes)
    values = exterior_potential(rho_nodes, parameters)
    coefficients = dct(values, type=2) / case.resolution
    coefficients[0] *= 0.5
    theta_dense = np.linspace(theta1, theta0, dense_count)
    x_dense = np.cos(theta_dense)
    rho_dense = 0.5 * (1.0 + x_dense)
    analytic = exterior_potential(rho_dense, parameters)
    represented = chebval(x_dense, coefficients)
    scale = float(np.max(np.abs(analytic)))
    maximum_error = float(np.max(np.abs(represented - analytic)))
    relative_error = maximum_error / scale
    transition_nodes = int(
        np.count_nonzero((rho_nodes > rho0) & (rho_nodes < rho1))
    )
    cap_nodes = int(np.count_nonzero(rho_nodes >= rho1))
    tail_ratio = float(
        np.max(np.abs(coefficients[-min(32, case.resolution) :]))
        / np.max(np.abs(coefficients))
    )
    if case.group == "tail":
        limit = 1.0e-3
    elif case.level == "verification":
        limit = 2.0e-3
    else:
        limit = PREFLIGHT_LIMITS[str(case.level)]
    passed = bool(
        np.all(np.isfinite(values))
        and np.all(np.isfinite(represented))
        and relative_error <= limit
        and transition_nodes >= 12
        and cap_nodes >= 12
    )
    return {
        "resolution": case.resolution,
        "length_over_M": case.length,
        "ell": case.ell,
        "curvature_coupling": case.curvature_coupling,
        "transition_nodes": transition_nodes,
        "outer_cap_nodes": cap_nodes,
        "analytic_minimum_P": float(np.min(analytic)),
        "analytic_maximum_P": float(np.max(analytic)),
        "maximum_absolute_P_error": maximum_error,
        "maximum_error_over_maximum_abs_P": relative_error,
        "maximum_tail_coefficient_ratio_last_32": tail_ratio,
        "relative_error_limit": limit,
        "passed": passed,
    }


def _tail_observers(case: CouplingCase) -> tuple[float, ...]:
    radii = np.asarray(FINITE_TAIL_OBSERVERS, dtype=float)
    if case.background == "schwarzschild":
        finite = 1.0 - 2.0 / radii
    elif case.background == "uniform":
        parameters = SdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=case.ell,
            curvature_coupling=case.curvature_coupling,
        )
        finite = uniform_compact_radius(radii, parameters)
    else:
        parameters = ExteriorSdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=case.ell,
            curvature_coupling=case.curvature_coupling,
        )
        finite = exterior_compact_radius(radii, parameters)
    return tuple(float(value) for value in finite) + (1.0,)


def _retarded_offset(case: CouplingCase) -> tuple[float, str]:
    if case.background == "schwarzschild":
        model = SchwarzschildScalarParameters(mass=1.0, ell=case.ell)
        return (
            float(schwarzschild_retarded_time_offset(model, REFERENCE_RADIUS)),
            "lim_(r->infinity)(h+r_*)",
        )
    if case.background == "uniform":
        model = SdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=case.ell,
            curvature_coupling=case.curvature_coupling,
        )
        return (
            float(uniform_retarded_time_offset(model, REFERENCE_RADIUS)),
            "lim_(r->r_c)(h+r_*)",
        )
    model = ExteriorSdSParameters(
        mass=1.0,
        cosmological_length=float(case.length),
        ell=case.ell,
        curvature_coupling=case.curvature_coupling,
    )
    return (
        float(exterior_retarded_time_offset(model, REFERENCE_RADIUS)),
        "lim_(r->r_c)(h_chi+r_*chi)",
    )


def _result_audit(
    result, case: CouplingCase, retarded_offset: float
) -> dict[str, float | bool]:
    """Reject finite-but-unstable tail archives before publication."""

    arrays = (
        np.asarray(result.signal_times),
        np.asarray(result.signals),
        np.asarray(result.u_snapshots),
        np.asarray(result.constraint_linf),
        np.asarray(result.constraint_l2),
    )
    finite = bool(all(value.size and np.all(np.isfinite(value)) for value in arrays))
    base_grid_snapshots = bool(
        arrays[2].ndim == 2 and arrays[2].shape[1] == np.asarray(result.rho).size
    )
    audit: dict[str, float | bool] = {
        "finite": finite,
        "base_grid_snapshot_width": base_grid_snapshots,
        "maximum_abs_signal": float(np.max(np.abs(arrays[1]))),
        "maximum_abs_snapshot": float(np.max(np.abs(arrays[2]))),
        "maximum_constraint_linf": float(np.max(np.abs(arrays[3]))),
    }
    if case.group != "tail":
        audit["passed"] = bool(finite and base_grid_snapshots)
        return audit

    signal_times = arrays[0] - retarded_offset
    snapshot_times = np.asarray(result.snapshot_times) - retarded_offset
    late_signals = arrays[1][signal_times >= 60.0]
    late_snapshots = arrays[2][snapshot_times >= 60.0]
    final_constraints = arrays[3][snapshot_times >= case.end_time_or_u - 10.0]
    final_window = signal_times >= case.end_time_or_u - 20.0
    preceding_window = (
        (signal_times >= case.end_time_or_u - 40.0)
        & (signal_times < case.end_time_or_u - 20.0)
    )
    final_rms = float(np.sqrt(np.mean(arrays[1][final_window, -1] ** 2)))
    preceding_rms = float(
        np.sqrt(np.mean(arrays[1][preceding_window, -1] ** 2))
    )
    growth_ratio = final_rms / max(preceding_rms, np.finfo(float).tiny)
    maximum_late_signal = float(np.max(np.abs(late_signals)))
    maximum_late_snapshot = float(np.max(np.abs(late_snapshots)))
    final_constraint = float(np.max(np.abs(final_constraints)))
    growth_ok = bool(final_rms <= max(5.0 * preceding_rms, 1.0e-10))
    passed = bool(
        finite
        and base_grid_snapshots
        and maximum_late_signal < 0.05
        and maximum_late_snapshot < 0.1
        and audit["maximum_constraint_linf"] < 0.05
        and final_constraint < 0.01
        and growth_ok
    )
    audit.update(
        {
            "maximum_abs_signal_after_U60": maximum_late_signal,
            "maximum_abs_snapshot_after_U60": maximum_late_snapshot,
            "final_window_constraint_linf": final_constraint,
            "outer_rms_penultimate_20M": preceding_rms,
            "outer_rms_final_20M": final_rms,
            "late_outer_rms_growth_ratio": growth_ratio,
            "late_growth_gate_passed": growth_ok,
            "late_signal_limit": 0.05,
            "late_snapshot_limit": 0.1,
            "constraint_limit": 0.05,
            "final_constraint_limit": 0.01,
            "growth_ratio_limit_above_absolute_floor": 5.0,
            "passed": passed,
        }
    )
    return audit


def run_case(
    output_dir: Path,
    name: str,
    *,
    resume_interrupted: bool = False,
) -> Path:
    """Run one named case and atomically publish its archive."""

    catalogue = case_catalogue()
    if name not in catalogue:
        raise ValueError(f"Unknown curvature-coupling case {name!r}.")
    case = catalogue[name]
    destination = archive_path(output_dir, case)
    reservation = destination.with_suffix(".running")
    checkpoint = destination.with_suffix(".checkpoint.npz")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    if resume_interrupted:
        if case.group != "tail":
            raise ValueError("Only tail cases have resumable checkpoints.")
        if not reservation.exists() or not checkpoint.exists():
            raise FileNotFoundError(
                "A resumable case requires both its reservation and checkpoint."
            )
    else:
        reservation = _reserve_destination(destination, case.name)

    from .sds_solver import (
        SdSNumericalParameters,
        run_exterior_sds_simulation,
        run_schwarzschild_scalar_simulation,
        run_sds_simulation,
    )

    offset, offset_definition = _retarded_offset(case)
    if case.group == "qnm":
        end_time = case.end_time_or_u
        signal_dt = 0.03
        snapshot_dt = end_time
        observers = (0.0, 0.25, 0.5, 0.75, 1.0)
        initial = QNM_INITIAL_DATA
        checkpoint_path = None
        checkpoint_dt = None
    else:
        end_time = case.end_time_or_u + offset
        signal_dt = 0.05
        snapshot_dt = 50.0
        observers = _tail_observers(case)
        initial = TAIL_INITIAL_DATA
        checkpoint_path = checkpoint
        checkpoint_dt = 250.0
    numerical = SdSNumericalParameters(
        resolution=case.resolution,
        timestep=case.timestep,
        end_time=end_time,
        signal_dt=signal_dt,
        snapshot_dt=snapshot_dt,
        observers=observers,
        timestepper="RK222",
        bridge="minimal",
        dealias=1.5,
    )

    preflight = None
    background = None
    if case.background == "schwarzschild":
        model = SchwarzschildScalarParameters(mass=1.0, ell=case.ell)
        result = run_schwarzschild_scalar_simulation(
            model,
            initial,
            numerical,
            checkpoint_path=checkpoint_path,
            checkpoint_dt=checkpoint_dt,
            explicit_potential=True,
        )
    elif case.background == "uniform":
        model = SdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=case.ell,
            curvature_coupling=case.curvature_coupling,
        )
        potential = uniform_potential(np.linspace(0.0, 1.0, 20_001), model)
        if not np.all(np.isfinite(potential)):
            raise FloatingPointError("Uniform-SdS potential is non-finite.")
        result = run_sds_simulation(
            model,
            initial,
            numerical,
            checkpoint_path=checkpoint_path,
            checkpoint_dt=checkpoint_dt,
            explicit_potential=case.group == "tail",
        )
    else:
        model = ExteriorSdSParameters(
            mass=1.0,
            cosmological_length=float(case.length),
            ell=case.ell,
            curvature_coupling=case.curvature_coupling,
        )
        preflight = exterior_spectral_preflight(case)
        background = background_audit(model)
        required = (
            preflight["passed"]
            and background["finite_coefficients"]
            and background["positive_interior_lapse"]
            and background["spacelike_bridge_interior"]
        )
        if not required:
            raise ValueError(
                f"Exterior background or spectral preflight failed: "
                f"{preflight}; {background}"
            )
        result = run_exterior_sds_simulation(
            model,
            initial,
            numerical,
            checkpoint_path=checkpoint_path,
            checkpoint_dt=checkpoint_dt,
            explicit_potential=True,
            endpoint_factored_characteristic_variables=False,
            conservative_characteristic_variables=case.group == "tail",
            characteristic_constraint_damping=0.0,
        )

    audit = _result_audit(result, case, offset)
    if not audit["passed"]:
        raise FloatingPointError(
            f"Result failed stability audit for {case.name}: {audit}"
        )
    result.metadata["retarded_time_offset"] = {
        "q": offset,
        "definition": offset_definition,
        "evaluation": (
            "endpoint-safe numerical quadrature"
            if case.background == "exterior"
            else "analytic"
        ),
    }
    result.metadata["physical_contract"] = _contract(case)
    result.metadata["physical_contract_sha256"] = contract_sha256(case)
    result.metadata["result_audit"] = audit
    if preflight is not None:
        result.metadata["spectral_preflight"] = preflight
    if background is not None:
        result.metadata["background_audit"] = background
    result.metadata["simulation_provenance"] = _provenance(case, output_dir)

    try:
        published = _write_once(result, destination, reservation)
        if checkpoint_path is not None:
            checkpoint.unlink(missing_ok=True)
        return published
    except BaseException:
        # Keep reservation/checkpoint for inspection and an explicit resume.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--group", choices=("qnm", "tail", "all"), default="all"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-interrupted", action="store_true")
    arguments = parser.parse_args()
    catalogue = case_catalogue()
    if arguments.cases:
        selected = list(arguments.cases)
    else:
        selected = [
            name
            for name, case in catalogue.items()
            if arguments.group == "all" or case.group == arguments.group
        ]
    if arguments.preflight:
        for name in selected:
            case = catalogue[name]
            if case.background == "exterior":
                print(json.dumps(exterior_spectral_preflight(case), sort_keys=True))
            else:
                print(json.dumps({"case": name, "passed": True}, sort_keys=True))
        return
    if not arguments.cases:
        for name in selected:
            print(name)
        return
    for name in selected:
        print(
            run_case(
                arguments.output_dir,
                name,
                resume_interrupted=arguments.resume_interrupted,
            )
        )


if __name__ == "__main__":
    main()
