"""Targeted normalized source production and convergence suite."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from .localized_source import LocalizedSourceParameters, minimum_ell_max
from .source_evolution import SourcedNumericalParameters, run_sourced_simulation
from .sds_model import SdSParameters, sds_horizons


LOGGER = logging.getLogger(__name__)

WIDTH_SCALES = (1.0, 0.7, 0.5)


def source_for_width(scale: float) -> LocalizedSourceParameters:
    if scale not in WIDTH_SCALES:
        raise ValueError(f"Unknown source width scale {scale:g}.")
    return LocalizedSourceParameters(
        amplitude=1.0,
        center_radius=6.0,
        radial_half_width=1.5 * scale,
        time_center=30.0,
        time_half_width=4.0 * scale,
        angular_concentration=16.0 / scale**2,
    )


def angular_cutoff(scale: float, margin: int = 4) -> int:
    source = source_for_width(scale)
    return minimum_ell_max(
        source.angular_concentration, omitted_power_tolerance=1e-10
    ) + margin


BASE = SourcedNumericalParameters(
    radial_resolution=1536,
    angular_ell_max=angular_cutoff(0.5),
    timestep=0.001,
    end_time=110.0,
    signal_dt=0.001,
    diagnostic_dt=1.0,
    snapshot_dt=110.0,
    snapshot_end_time=0.0,
    snapshot_radial_points=64,
    observer_radii=(8.0, 12.0, None),
    compact_modal_storage=True,
)


def l12_safety() -> dict:
    source = source_for_width(0.5)
    horizons = sds_horizons(SdSParameters(cosmological_length=12.0))
    left, right = source.radial_support
    return {
        "black_hole_horizon": horizons.black_hole,
        "cosmological_horizon": horizons.cosmological,
        "source_support": [left, right],
        "observer_radius": 8.0,
        "safe": horizons.black_hole < left < right < 8.0 < horizons.cosmological,
        "r12_excluded": not 12.0 < horizons.cosmological,
    }


def _run(
    path: Path,
    *,
    background: str,
    source: LocalizedSourceParameters,
    numerical: SourcedNumericalParameters,
    cosmological_length: float = 80.0,
    backend: str = "finite_difference",
    force: bool = False,
) -> Path:
    if path.exists() and not force:
        LOGGER.info("reusing %s", path)
        return path
    if background == "sds" and cosmological_length == 12.0 and not l12_safety()["safe"]:
        raise ValueError("The L/M=12 source or observer is not safely inside the horizons.")
    path.parent.mkdir(parents=True, exist_ok=True)
    arguments = {
        "background": background,
        "source": source,
        "numerical": numerical,
        "cosmological_length": cosmological_length,
    }
    if backend == "finite_difference":
        result = run_sourced_simulation(**arguments)
    elif backend == "dedalus":
        from .dedalus_source_evolution import run_sourced_dedalus_simulation

        result = run_sourced_dedalus_simulation(**arguments)
    else:
        raise ValueError(f"Unknown backend {backend!r}.")
    result.metadata["production_suite"] = {
        "source_width_scale": source.radial_half_width / 1.5,
        "angular_cutoff_minimum": minimum_ell_max(
            source.angular_concentration, omitted_power_tolerance=1e-10
        ),
        "angular_cutoff_margin": numerical.angular_ell_max
        - minimum_ell_max(
            source.angular_concentration, omitted_power_tolerance=1e-10
        ),
        "l12_safety": l12_safety() if cosmological_length == 12.0 else None,
    }
    result.save(path)
    return path


def pilot_cases() -> dict[str, tuple[float, SourcedNumericalParameters]]:
    cases: dict[str, tuple[float, SourcedNumericalParameters]] = {}
    narrow_cutoff = angular_cutoff(0.5)
    for resolution in (768, 1024, 1536, 2048):
        cases[f"radial_N{resolution}"] = (
            0.5,
            replace(BASE, radial_resolution=resolution, angular_ell_max=narrow_cutoff),
        )
    for timestep in (0.004, 0.002, 0.001):
        cases[f"temporal_dt{timestep:g}"] = (
            0.5,
            replace(
                BASE,
                radial_resolution=768,
                timestep=timestep,
                signal_dt=timestep,
                angular_ell_max=narrow_cutoff,
            ),
        )
    for scale in WIDTH_SCALES:
        minimum = angular_cutoff(scale, margin=0)
        for cutoff in (minimum, minimum + 4, minimum + 8):
            label = str(scale).replace(".", "p")
            cases[f"angular_w{label}_lmax{cutoff}"] = (
                scale,
                replace(
                    BASE,
                    radial_resolution=1024,
                    timestep=0.002,
                    signal_dt=0.002,
                    angular_ell_max=cutoff,
                ),
            )
    for scale in WIDTH_SCALES:
        label = str(scale).replace(".", "p")
        cases[f"width_w{label}"] = (
            scale,
            replace(BASE, angular_ell_max=angular_cutoff(scale)),
        )
    return cases


def production_cases() -> dict[str, tuple[str, float, tuple[float | None, ...]]]:
    return {
        "schwarzschild": ("schwarzschild", 80.0, (8.0, 12.0, None)),
        "sds_L12": ("sds", 12.0, (8.0, None)),
        "sds_L20": ("sds", 20.0, (8.0, 12.0, None)),
        "sds_L40": ("sds", 40.0, (8.0, 12.0, None)),
        "sds_L80": ("sds", 80.0, (8.0, 12.0, None)),
        "sds_L160": ("sds", 160.0, (8.0, 12.0, None)),
    }


def run_named_case(
    output_dir: Path,
    name: str,
    *,
    force: bool = False,
    backend: str = "finite_difference",
) -> Path:
    pilots = pilot_cases()
    if name in pilots:
        scale, numerical = pilots[name]
        return _run(
            Path(output_dir) / "pilots" / "raw" / f"{name}.npz",
            background="schwarzschild",
            source=source_for_width(scale),
            numerical=numerical,
            backend=backend,
            force=force,
        )
    productions = production_cases()
    if name in productions:
        background, length, observers = productions[name]
        numerical = replace(BASE, observer_radii=observers)
        directory = "dedalus" if backend == "dedalus" else "raw"
        return _run(
            Path(output_dir) / directory / f"{name}.npz",
            background=background,
            source=source_for_width(0.5),
            numerical=numerical,
            cosmological_length=length,
            backend=backend,
            force=force,
        )
    raise ValueError(f"Unknown production case {name!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/caustic_production")
    )
    parser.add_argument(
        "--backend", choices=("finite_difference", "dedalus"), default="finite_difference"
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not arguments.cases:
        for name in (*pilot_cases(), *production_cases()):
            print(name)
        return
    for name in arguments.cases:
        print(
            run_named_case(
                arguments.output_dir,
                name,
                force=arguments.force,
                backend=arguments.backend,
            )
        )


if __name__ == "__main__":
    main()
