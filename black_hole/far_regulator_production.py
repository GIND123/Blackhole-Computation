"""Production far-transition simulations for the regulator paper.

The frozen Schwarzschild and uniform-SdS controls are read only.  This runner
creates only the horizon-supported exterior cases, using a length-dependent
resolution ladder that keeps the narrowing Chebyshev layer resolved.  Every
archive destination is reserved atomically and existing destinations are
refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct

from .exterior_sds_model import (
    ExteriorSdSParameters,
    chebyshev_angle,
    compactification_scale,
    rescaled_scalar_potential,
    transition_compact_radii,
)
from .regulator_suite import (
    LEVELS,
    _reserve_destination,
    _write_once,
    flat_initial_data,
)
from .reproducibility import reproducibility_metadata


OUTPUT_ROOT = Path("results/exterior_regulator_far_production_v1")
CONTROL_ROOT = Path("results/regulator_production_v3")
LENGTHS = (80, 160, 320, 640)
HEIGHT_REFERENCE_RADIUS = 4.0
END_TIME = 200.0
SIGNAL_DT = 0.03
TIMESTEPS = {
    "coarse": 0.005,
    "medium": 0.00375,
    "fine": 0.0025,
}

RESOLUTIONS = {
    80: {"coarse": 384, "medium": 512, "fine": 768},
    160: {"coarse": 768, "medium": 1024, "fine": 1536},
    320: {"coarse": 1024, "medium": 1536, "fine": 2048},
    640: {"coarse": 1792, "medium": 2048, "fine": 2304},
}

PREFLIGHT_RELATIVE_LIMITS = {
    "coarse": 0.25,
    "medium": 0.15,
    "fine": 0.05,
}


def production_numerical(length: int, level: str):
    """Return the coupled spectral/time settings for one production case."""

    if length not in LENGTHS:
        raise ValueError(f"Unsupported L/M={length}.")
    if level not in LEVELS:
        raise ValueError(f"Unknown refinement level {level!r}.")
    from .sds_solver import SdSNumericalParameters

    resolution = RESOLUTIONS[length][level]
    return SdSNumericalParameters(
        resolution=resolution,
        timestep=TIMESTEPS[level],
        end_time=END_TIME,
        signal_dt=SIGNAL_DT,
        snapshot_dt=END_TIME,
        timestepper="RK222",
        bridge="minimal",
        dealias=1.5,
    )


def physical_contract() -> dict:
    """Return the fixed physics and coordinate contract for the sequence."""

    return {
        "study": "far_horizon_supported_artificial_cosmology_production",
        "mass": 1.0,
        "ell": 2,
        "gauge": "exterior_supported_minimal",
        "metric": "f_chi=1-2M/r-(r/L)^2 chi_L",
        "finite_L_compactification": (
            "rho=(1-2M/r)/(1-2M/r_c), with r_c the uniform-SdS "
            "cosmological horizon"
        ),
        "transition": {
            "outer_radius": "R1=0.9*r_c",
            "angle": "theta=acos(2*rho-1)",
            "inner_location": "theta0=2*theta1",
            "profile": "standard C-infinity step in Chebyshev endpoint angle",
            "outer_cap": "chi_L=1 for r>=R1",
        },
        "initial_data": flat_initial_data().as_dict(),
        "height_reference_radius_over_M": HEIGHT_REFERENCE_RADIUS,
        "retarded_time": "U=tau-q_chi with endpoint-safe q_chi integral",
        "imex_split": (
            "transport terms implicit; bounded order-unity potential term explicit"
        ),
        "outer_boundary_pair": (
            "exterior-supported cosmological horizon versus Schwarzschild scri+"
        ),
        "headline_observable": "raw unshifted outer waveform E2 on 0<=U/M<=80",
        "time_translation_fitted": False,
        "amplitude_rescaling_fitted": False,
        "background_transfer_correction_used_in_headline": False,
        "control_package": CONTROL_ROOT.as_posix(),
        "control_archives_modified": False,
        "production_end_time_over_M": END_TIME,
        "production_resolutions": RESOLUTIONS,
    }


def contract_sha256() -> str:
    encoded = json.dumps(
        physical_contract(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_catalogue() -> dict[str, tuple[int, str]]:
    """Return the twelve isolated production cases."""

    return {
        f"far_sds_L{length}_{level}": (length, level)
        for length in LENGTHS
        for level in LEVELS
    }


def archive_path(output_dir: Path, length: int, level: str) -> Path:
    """Return the raw archive path for one production case."""

    if length not in LENGTHS:
        raise ValueError(f"Unsupported L/M={length}.")
    if level not in LEVELS:
        raise ValueError(f"Unknown refinement level {level!r}.")
    return (
        Path(output_dir)
        / "raw"
        / "exterior"
        / f"L{length}"
        / level
        / f"sds_L{length}.npz"
    )


def spectral_preflight(
    length: int,
    level: str,
    *,
    dense_count: int = 20_001,
) -> dict[str, float | int | bool]:
    """Audit the represented transition potential before an expensive run."""

    if dense_count < 10_001:
        raise ValueError("The production dense audit requires at least 10001 points.")
    numerical = production_numerical(length, level)
    resolution = numerical.resolution
    parameters = ExteriorSdSParameters(
        mass=1.0, cosmological_length=float(length), ell=2
    )
    rho0, rho1 = transition_compact_radii(parameters)
    theta0 = float(chebyshev_angle(np.array(rho0)))
    theta1 = float(chebyshev_angle(np.array(rho1)))

    theta_nodes = np.pi * (np.arange(resolution) + 0.5) / resolution
    x_nodes = np.cos(theta_nodes)
    rho_nodes = 0.5 * (1.0 + x_nodes)
    scale = 2.0 * parameters.mass / compactification_scale(parameters)
    q_nodes = scale * rescaled_scalar_potential(rho_nodes, parameters)
    coefficients = dct(q_nodes, type=2) / resolution
    coefficients[0] *= 0.5

    theta_dense = np.linspace(theta1, theta0, dense_count)
    x_dense = np.cos(theta_dense)
    rho_dense = 0.5 * (1.0 + x_dense)
    analytic = scale * rescaled_scalar_potential(rho_dense, parameters)
    represented = chebval(x_dense, coefficients)
    maximum_error = float(np.max(np.abs(represented - analytic)))
    analytic_minimum = float(np.min(analytic))
    represented_minimum = float(np.min(represented))
    relative_error = maximum_error / analytic_minimum
    transition_nodes = int(np.count_nonzero((rho_nodes > rho0) & (rho_nodes < rho1)))
    cap_nodes = int(np.count_nonzero(rho_nodes >= rho1))
    passed = bool(
        represented_minimum > 0.0
        and relative_error <= PREFLIGHT_RELATIVE_LIMITS[level]
    )
    return {
        "length_over_M": length,
        "level": level,
        "resolution": resolution,
        "timestep_over_M": numerical.timestep,
        "transition_nodes": transition_nodes,
        "outer_cap_nodes": cap_nodes,
        "analytic_minimum_Q": analytic_minimum,
        "represented_minimum_Q": represented_minimum,
        "maximum_absolute_Q_error": maximum_error,
        "maximum_error_over_analytic_minimum": relative_error,
        "maximum_tail_coefficient_last_32": float(
            np.max(np.abs(coefficients[-min(32, resolution) :]))
        ),
        "relative_error_limit": PREFLIGHT_RELATIVE_LIMITS[level],
        "passed": passed,
    }


def _provenance(case: str, length: int, level: str, output_dir: Path) -> dict:
    metadata = reproducibility_metadata()
    metadata.update(
        {
            "role": "production_simulation",
            "case": case,
            "cosmological_length_over_M": length,
            "refinement_level": level,
            "physical_contract_sha256": contract_sha256(),
            "command": (
                f"python -m black_hole.far_regulator_production {case} "
                f"--output-dir {Path(output_dir).as_posix()}"
            ),
        }
    )
    return metadata


def run_case(output_dir: Path, case: str) -> Path:
    """Run one production case and write its archive exactly once."""

    catalogue = case_catalogue()
    if case not in catalogue:
        raise ValueError(f"Unknown far-regulator production case {case!r}.")
    length, level = catalogue[case]
    destination = archive_path(output_dir, length, level)
    if destination.exists() or destination.with_suffix(".incomplete.npz").exists():
        raise FileExistsError(f"Refusing existing destination for {case}: {destination}")

    preflight = spectral_preflight(length, level)
    if not preflight["passed"]:
        raise ValueError(f"Spectral preflight failed for {case}: {preflight}")

    from .exterior_sds_model import background_audit, retarded_time_offset
    from .sds_solver import run_exterior_sds_simulation

    model = ExteriorSdSParameters(
        mass=1.0, cosmological_length=float(length), ell=2
    )
    audit = background_audit(model)
    required_checks = (
        "finite_coefficients",
        "positive_interior_lapse",
        "spacelike_bridge_interior",
        "nonnegative_scalar_potential",
    )
    if not all(audit[check] for check in required_checks):
        raise ValueError(f"Background audit failed for {case}: {audit}")
    offset = retarded_time_offset(model, HEIGHT_REFERENCE_RADIUS)
    reservation = _reserve_destination(destination, case)
    result = run_exterior_sds_simulation(
        model,
        flat_initial_data(),
        production_numerical(length, level),
        explicit_potential=True,
    )
    result.metadata["height_normalization"] = {
        "reference_radius": HEIGHT_REFERENCE_RADIUS,
        "height_at_reference": 0.0,
    }
    result.metadata["retarded_time_offset"] = {
        "q": float(offset),
        "definition": "lim_(r->r_c) (h_chi+r_*chi)",
        "evaluation": "endpoint-safe numerical quadrature",
    }
    result.metadata["physical_contract"] = physical_contract()
    result.metadata["background_audit"] = audit
    result.metadata["spectral_preflight"] = preflight
    result.metadata["simulation_provenance"] = _provenance(
        case, length, level, output_dir
    )
    return _write_once(result, destination, reservation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Print audits without evolving any case.",
    )
    arguments = parser.parse_args()
    selected = arguments.cases or list(case_catalogue())
    if arguments.preflight:
        for case in selected:
            length, level = case_catalogue()[case]
            print(json.dumps(spectral_preflight(length, level), sort_keys=True))
        return
    if not arguments.cases:
        for name in case_catalogue():
            print(name)
        return
    for case in arguments.cases:
        print(run_case(arguments.output_dir, case))


if __name__ == "__main__":
    main()
