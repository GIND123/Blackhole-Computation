"""Fixed-transition-width exterior SdS simulations for the regulator comparison.

Only the new exterior-supported backgrounds are evolved here.  The
Schwarzschild and uniform-SdS controls remain in the frozen v3 regulator
package and are treated as read-only inputs by :mod:`exterior_regulator_analysis`.
Every destination is reserved atomically and existing archives are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .regulator_suite import (
    LEVELS,
    _reserve_destination,
    _write_once,
    flat_initial_data,
    flat_numerical,
)
from .reproducibility import reproducibility_metadata


OUTPUT_ROOT = Path("results/exterior_regulator_width_floor_v1")
CONTROL_ROOT = Path("results/regulator_production_v3")
EXTERIOR_LENGTHS = (80,)
HEIGHT_REFERENCE_RADIUS = 4.0


def physical_contract() -> dict:
    """Return the common physical and coordinate choices for this sequence."""

    from .exterior_sds_model import (
        TRANSITION_MINIMUM_ANGLE_WIDTH,
        TRANSITION_OUTER_HORIZON_FRACTION,
        TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M,
    )

    return {
        "study": "fixed_transition_width_exterior_supported_regulator",
        "mass": 1.0,
        "ell": 2,
        "gauge": "exterior_supported_minimal",
        "metric": "f_chi=1-2M/r-(r/L)^2 chi_L",
        "finite_L_compactification": (
            "rho=(1-2M/r)/(1-2M/r_c), with r_c the uniform-SdS "
            "cosmological horizon"
        ),
        "transition": {
            "outer_horizon_fraction": (
                TRANSITION_OUTER_HORIZON_FRACTION
            ),
            "width_reference_length_over_M": (
                TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M
            ),
            "minimum_transition_angle_width": TRANSITION_MINIMUM_ANGLE_WIDTH,
            "endpoint_rule": (
                f"theta1=max(theta({TRANSITION_OUTER_HORIZON_FRACTION}*r_c),"
                "delta_theta_min), theta0=2*theta1"
            ),
            "profile": "standard C-infinity step in Chebyshev endpoint angle",
            "grid_design": (
                "transition and cap retain nonzero limiting Chebyshev-angle widths"
            ),
            "outer_cap": "chi_L=1 for rho>=rho1",
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
        "time_translation_fitted": False,
        "amplitude_rescaling_fitted": False,
        "control_package": CONTROL_ROOT.as_posix(),
        "control_archives_modified": False,
    }


def contract_sha256() -> str:
    encoded = json.dumps(
        physical_contract(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_catalogue() -> dict[str, tuple[str, float]]:
    """Return the three fixed-minimum-width refinement cases."""

    return {
        f"exterior_sds_L{length}_{level}": (level, float(length))
        for level in LEVELS
        for length in EXTERIOR_LENGTHS
    }


def archive_path(output_dir: Path, level: str, length: float) -> Path:
    """Return the isolated raw-archive path for one exterior case."""

    if level not in LEVELS:
        raise ValueError(f"Unknown refinement level {level!r}.")
    if float(length) not in {float(value) for value in EXTERIOR_LENGTHS}:
        raise ValueError(f"Unsupported exterior length L/M={length:g}.")
    return (
        Path(output_dir)
        / "raw"
        / "exterior"
        / level
        / f"sds_L{length:g}.npz"
    )


def _provenance(case: str, level: str, output_dir: Path) -> dict:
    metadata = reproducibility_metadata()
    metadata.update(
        {
            "role": "simulation",
            "case": case,
            "refinement_level": level,
            "physical_contract_sha256": contract_sha256(),
            "command": (
                f"python -m black_hole.exterior_regulator_suite {case} "
                f"--output-dir {Path(output_dir).as_posix()}"
            ),
        }
    )
    return metadata


def run_case(output_dir: Path, case: str) -> Path:
    """Run one new exterior-supported case and save it exactly once."""

    catalogue = case_catalogue()
    if case not in catalogue:
        raise ValueError(f"Unknown exterior-regulator case {case!r}.")
    level, length = catalogue[case]
    destination = archive_path(output_dir, level, length)
    if destination.exists() or destination.with_suffix(".incomplete.npz").exists():
        raise FileExistsError(f"Refusing existing destination for {case}: {destination}")

    # Imported lazily so catalogue and safety checks do not require Dedalus.
    from .exterior_sds_model import (
        ExteriorSdSParameters,
        background_audit,
        retarded_time_offset,
    )
    from .sds_solver import run_exterior_sds_simulation

    model = ExteriorSdSParameters(
        mass=1.0,
        cosmological_length=length,
        ell=2,
    )
    audit = background_audit(model)
    required_checks = (
        "finite_coefficients",
        "positive_interior_lapse",
        "spacelike_bridge_interior",
        "nonnegative_scalar_potential",
    )
    if not all(audit[check] for check in required_checks):
        raise ValueError(f"Exterior background audit failed for {case}: {audit}")
    offset = retarded_time_offset(model, HEIGHT_REFERENCE_RADIUS)
    reservation = _reserve_destination(destination, case)
    result = run_exterior_sds_simulation(
        model,
        flat_initial_data(),
        flat_numerical(level),
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
    result.metadata["simulation_provenance"] = _provenance(
        case, level, output_dir
    )
    return _write_once(result, destination, reservation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.cases:
        for name in case_catalogue():
            print(name)
        return
    for case in arguments.cases:
        print(run_case(arguments.output_dir, case))


if __name__ == "__main__":
    main()
