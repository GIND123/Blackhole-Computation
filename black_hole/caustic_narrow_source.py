"""A narrower emitter, run only to sharpen the caustic in a visualization.

The production localized source has an angular width of ``kappa^{-1/2}``
with ``kappa = 64``, which is ``7.16`` degrees.  Measured on the archived
production data, the antipodal focus is ``40`` degrees across and does not
change at all when the reconstruction is truncated from ``ell_max = 50`` down
to ``20``: the relative change in the peak between ``ell_max = 40`` and ``50``
is ``4e-10``.  The focus is therefore set by the width of the emitter and not
by the length of the angular sum, so a longer sum alone cannot sharpen it.

This module runs the same physics with an emitter half as wide in radius and
in time and four times as concentrated in angle.  It exists to produce a
legible picture of the caustic and nothing else:

* it is **not** part of the regulator comparison and none of its output feeds
  a production table, a manifest of the frozen package, or a claim;
* it is **not** a point source limit.  ``kappa = 256`` is a finite width of
  ``3.58`` degrees, four times the truncation scale of the sum used here, and
  the focus it produces is still a resolved feature of a smooth emitter.

The angular truncation is chosen from the emitter spectrum rather than by
habit.  For ``kappa = 256`` the retained angular power reaches one in double
precision at ``ell_max = 80``, while ``ell_max = 50`` would omit ``3.9e-5``
of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .localized_source import (
    LocalizedSourceParameters,
    minimum_ell_max,
    retained_angular_fraction,
)
from .source_evolution import SourcedNumericalParameters, run_sourced_simulation


OUTPUT_ROOT = Path("results/caustic_diagnostics")
COSMOLOGICAL_LENGTH = 80.0

PRODUCTION_SOURCE = LocalizedSourceParameters(
    amplitude=1.0,
    center_radius=6.0,
    radial_half_width=0.75,
    time_center=30.0,
    time_half_width=2.0,
    angular_concentration=64.0,
)
NARROW_SOURCE = LocalizedSourceParameters(
    amplitude=1.0,
    center_radius=6.0,
    radial_half_width=0.375,
    time_center=30.0,
    time_half_width=1.0,
    angular_concentration=256.0,
)

ANGULAR_TRUNCATION = 80
OMITTED_POWER_TOLERANCE = 1.0e-10

# Two levels.  The finer one is the picture; the coarser one exists so the
# picture can be shown to be resolved rather than asserted to be.
LEVELS = {
    "fine": (2048, 0.0005),
    "coarse": (1024, 0.001),
}


def snapshot_times(
    first: float = 40.0, last: float = 54.0, cadence: float = 0.25
) -> tuple[float, ...]:
    """Return the bridge times sampled around the antipodal focus."""

    count = int(round((last - first) / cadence)) + 1
    return tuple(
        float(value) for value in np.round(np.linspace(first, last, count), 6)
    )


def angular_budget() -> dict:
    """Return the angular truncation justification for both emitters."""

    return {
        "production": {
            "angular_concentration": PRODUCTION_SOURCE.angular_concentration,
            "angular_width_degrees": float(
                np.rad2deg(PRODUCTION_SOURCE.angular_width)
            ),
            "retained_power_at_ell_max_50": retained_angular_fraction(
                PRODUCTION_SOURCE.angular_concentration, 50
            ),
            "minimum_ell_max": minimum_ell_max(
                PRODUCTION_SOURCE.angular_concentration,
                omitted_power_tolerance=OMITTED_POWER_TOLERANCE,
            ),
        },
        "narrow": {
            "angular_concentration": NARROW_SOURCE.angular_concentration,
            "angular_width_degrees": float(np.rad2deg(NARROW_SOURCE.angular_width)),
            "retained_power_at_ell_max_50": retained_angular_fraction(
                NARROW_SOURCE.angular_concentration, 50
            ),
            "retained_power_at_chosen_ell_max": retained_angular_fraction(
                NARROW_SOURCE.angular_concentration, ANGULAR_TRUNCATION
            ),
            "minimum_ell_max": minimum_ell_max(
                NARROW_SOURCE.angular_concentration,
                omitted_power_tolerance=OMITTED_POWER_TOLERANCE,
            ),
            "chosen_ell_max": ANGULAR_TRUNCATION,
        },
        "omitted_power_tolerance": OMITTED_POWER_TOLERANCE,
    }


def run_case(level: str, output_dir: Path = OUTPUT_ROOT) -> Path:
    """Evolve the narrow emitter at one refinement level."""

    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}.")
    resolution, timestep = LEVELS[level]
    times = snapshot_times()
    destination = Path(output_dir) / "raw" / f"narrow_source_L80_{level}.npz"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_suffix(".running")
    with reservation.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(level + "\n")

    numerical = SourcedNumericalParameters(
        radial_resolution=resolution,
        angular_ell_max=ANGULAR_TRUNCATION,
        timestep=timestep,
        end_time=max(times) + 0.25,
        signal_dt=0.01,
        diagnostic_dt=1.0,
        snapshot_dt=max(times) + 1.0,
        snapshot_end_time=0.0,
        snapshot_radial_points=1024,
        requested_snapshot_times=times,
        observer_radii=(8.0, 12.0, None),
        compact_modal_storage=True,
    )
    result = run_sourced_simulation(
        background="sds",
        source=NARROW_SOURCE,
        numerical=numerical,
        cosmological_length=COSMOLOGICAL_LENGTH,
    )
    result.metadata["visualization_only"] = {
        "purpose": "sharpen the antipodal caustic for a figure",
        "excluded_from": [
            "regulator comparison",
            "production tables",
            "frozen package manifest",
        ],
        "is_point_source_limit": False,
        "emitter_angular_width_degrees": float(
            np.rad2deg(NARROW_SOURCE.angular_width)
        ),
        "production_emitter_angular_width_degrees": float(
            np.rad2deg(PRODUCTION_SOURCE.angular_width)
        ),
        "angular_budget": angular_budget(),
        "refinement_level": level,
        "time_translation_fitted": False,
    }
    temporary = destination.with_suffix(".incomplete.npz")
    result.save(temporary)
    temporary.rename(destination)
    reservation.unlink()
    return destination


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=sorted(LEVELS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--budget-only",
        action="store_true",
        help="print the angular truncation justification and stop",
    )
    arguments = parser.parse_args()
    if arguments.budget_only:
        print(json.dumps(angular_budget(), indent=2))
        return
    print(run_case(arguments.level, arguments.output_dir))


if __name__ == "__main__":
    main()
