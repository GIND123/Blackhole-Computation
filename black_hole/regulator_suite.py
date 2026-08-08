"""Frozen simulation suite for the artificial-cosmology regulator study.

The module intentionally performs simulations only.  Analysis lives in a
separate module and commit so that every raw archive has unambiguous source
provenance.  Every case writes once through an ``.incomplete.npz`` file and
refuses an existing destination.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .localized_source import LocalizedSourceParameters
from .reproducibility import reproducibility_metadata
from .sds_model import ArealBumpInitialData, SdSParameters
from .source_evolution import SourcedNumericalParameters, run_sourced_simulation


OUTPUT_ROOT = Path("results/regulator_production_v3")
FLAT_LENGTHS = (20, 40, 80, 160, 320, 640)
SOURCE_LENGTHS = (80, 160, 320, 640)
LEVELS = ("coarse", "medium", "fine")


def flat_initial_data() -> ArealBumpInitialData:
    """Return the one fixed physical datum used by the flat-limit sequence."""

    return ArealBumpInitialData(
        center_radius=4.0,
        support_half_width=1.5,
        time_symmetric=True,
        pi_amplitude=0.0,
    )


def localized_source() -> LocalizedSourceParameters:
    """Return the one fixed physical source used by the caustic sequence."""

    return LocalizedSourceParameters(
        amplitude=1.0,
        center_radius=6.0,
        radial_half_width=0.75,
        time_center=30.0,
        time_half_width=2.0,
        angular_concentration=64.0,
    )


def flat_numerical(level: str):
    """Return a coupled spectral/time refinement level for boundary waves."""

    if level not in LEVELS:
        raise ValueError(f"Unknown refinement level {level!r}.")
    from .sds_solver import SdSNumericalParameters

    settings = {
        "coarse": (384, 0.005),
        "medium": (512, 0.00375),
        "fine": (768, 0.0025),
    }
    resolution, timestep = settings[level]
    return SdSNumericalParameters(
        resolution=resolution,
        timestep=timestep,
        end_time=200.0,
        signal_dt=0.03,
        snapshot_dt=200.0,
        timestepper="RK222",
        bridge="minimal",
        dealias=1.5,
    )


def source_numerical(level: str) -> SourcedNumericalParameters:
    """Return a coupled radial/time/angular refinement for fixed-source runs."""

    if level not in LEVELS:
        raise ValueError(f"Unknown refinement level {level!r}.")
    settings = {
        "coarse": (1024, 0.001, 42),
        "medium": (1536, 1.0 / 1500.0, 46),
        "fine": (2048, 0.0005, 50),
    }
    resolution, timestep, ell_max = settings[level]
    return SourcedNumericalParameters(
        radial_resolution=resolution,
        angular_ell_max=ell_max,
        timestep=timestep,
        end_time=60.0,
        signal_dt=timestep,
        diagnostic_dt=1.0,
        snapshot_dt=60.0,
        snapshot_end_time=0.0,
        snapshot_radial_points=64,
        observer_radii=(8.0, 12.0, None),
        compact_modal_storage=True,
    )


def physical_contract(study: str) -> dict:
    """Return the frozen physical and coordinate choices for one study."""

    common = {
        "mass": 1.0,
        "gauge": "minimal",
        "finite_L_compactification": "rho=(1-r_b/r)/(1-r_b/r_c)",
        "schwarzschild_compactification": "rho=1-2M/r",
        "height_reference_radius_over_M": 4.0,
        "retarded_time": "U=tau-q with analytic q=lim(h+r_*)",
        "outer_boundary_pair": "SdS H_c+ versus Schwarzschild scri+",
    }
    if study == "flat":
        return {
            **common,
            "study": "minimal_gauge_flat_limit_waveform",
            "ell": 2,
            "initial_data": flat_initial_data().as_dict(),
        }
    if study == "source":
        return {
            **common,
            "study": "normalized_localized_retarded_response",
            "initial_data": "vanishing field and Killing-time derivative",
            "source": localized_source().as_dict(),
        }
    raise ValueError(f"Unknown regulator study {study!r}.")


def contract_sha256(study: str) -> str:
    encoded = json.dumps(
        physical_contract(study), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_catalogue() -> dict[str, tuple[str, str, float | None]]:
    """Return every frozen production and case-specific refinement case."""

    cases: dict[str, tuple[str, str, float | None]] = {}
    for level in LEVELS:
        cases[f"flat_schwarzschild_{level}"] = ("flat", level, None)
        for length in FLAT_LENGTHS:
            cases[f"flat_sds_L{length}_{level}"] = (
                "flat",
                level,
                float(length),
            )
        cases[f"source_schwarzschild_{level}"] = ("source", level, None)
        for length in SOURCE_LENGTHS:
            cases[f"source_sds_L{length}_{level}"] = (
                "source",
                level,
                float(length),
            )
    return cases


def archive_path(
    output_dir: Path, study: str, level: str, length: float | None
) -> Path:
    label = "schwarzschild" if length is None else f"sds_L{length:g}"
    return Path(output_dir) / "raw" / study / level / f"{label}.npz"


def _reserve_destination(destination: Path, case: str) -> Path:
    """Atomically reserve one archive path before expensive computation."""

    if destination.exists():
        raise FileExistsError(
            f"Refusing to reuse or overwrite regulator archive: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_suffix(".running")
    try:
        with reservation.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(case + "\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"Case is already running or needs inspection: {reservation}"
        ) from error
    return reservation


def _write_once(result, destination: Path, reservation: Path) -> Path:
    temporary = destination.with_suffix(".incomplete.npz")
    if temporary.exists():
        raise FileExistsError(
            f"Remove or inspect the incomplete archive before retrying: {temporary}"
        )
    result.save(temporary)
    temporary.rename(destination)
    reservation.unlink()
    return destination


def _provenance(
    case: str, study: str, level: str, output_dir: Path
) -> dict:
    metadata = reproducibility_metadata()
    metadata.update(
        {
            "role": "simulation",
            "case": case,
            "refinement_level": level,
            "physical_contract_sha256": contract_sha256(study),
            "command": (
                f"python -m black_hole.regulator_suite {case} "
                f"--output-dir {Path(output_dir).as_posix()}"
            ),
        }
    )
    return metadata


def run_case(output_dir: Path, case: str) -> Path:
    """Run one named case and atomically save its new raw archive."""

    catalogue = case_catalogue()
    if case not in catalogue:
        raise ValueError(f"Unknown regulator case {case!r}.")
    study, level, length = catalogue[case]
    destination = archive_path(output_dir, study, level, length)
    if destination.exists() or destination.with_suffix(".incomplete.npz").exists():
        raise FileExistsError(f"Refusing existing destination for {case}: {destination}")
    reservation = _reserve_destination(destination, case)

    if study == "flat":
        from .schwarzschild_scalar import SchwarzschildScalarParameters
        from .sds_model import retarded_time_offset as sds_offset
        from .sds_solver import (
            run_schwarzschild_scalar_simulation,
            run_sds_simulation,
        )
        from .schwarzschild_scalar import retarded_time_offset as schwarzschild_offset

        numerical = flat_numerical(level)
        initial = flat_initial_data()
        if length is None:
            model = SchwarzschildScalarParameters(mass=1.0, ell=2)
            result = run_schwarzschild_scalar_simulation(model, initial, numerical)
            offset = schwarzschild_offset(model, 4.0)
            offset_definition = "lim_(r->infinity) (h+r_*)"
        else:
            model = SdSParameters(mass=1.0, cosmological_length=length, ell=2)
            result = run_sds_simulation(model, initial, numerical)
            offset = sds_offset(model, 4.0)
            offset_definition = "lim_(r->r_c) (h+r_*)"
        result.metadata["height_normalization"] = {
            "reference_radius": 4.0,
            "height_at_reference": 0.0,
        }
        result.metadata["retarded_time_offset"] = {
            "q": float(offset),
            "definition": offset_definition,
            "evaluation": "analytic",
        }
        result.metadata["physical_contract"] = physical_contract(study)
        result.metadata["simulation_provenance"] = _provenance(
            case, study, level, output_dir
        )
        return _write_once(result, destination, reservation)

    numerical = source_numerical(level)
    background = "schwarzschild" if length is None else "sds"
    result = run_sourced_simulation(
        background=background,
        source=localized_source(),
        numerical=numerical,
        cosmological_length=80.0 if length is None else length,
    )
    result.metadata["physical_contract"] = physical_contract(study)
    result.metadata["simulation_provenance"] = _provenance(
        case, study, level, output_dir
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
