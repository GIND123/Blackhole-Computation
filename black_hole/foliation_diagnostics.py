"""Analytic bridge foliation and conditioning diagnostics.

Every quantity here is evaluated from the closed-form bridge coefficients in
:mod:`black_hole.sds_model`.  Nothing is read from a simulation archive, so the
foliation table that selects the minimal gauge is reproducible from the
formulas alone::

    python -m black_hole.foliation_diagnostics write

That command regenerates the three CSV files under ``paper/figs/data``.
``tests/test_foliation_diagnostics.py`` checks the regenerated values against
the archived ones to nine decimal places.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from .sds_model import (
    BRIDGE_CHOICES,
    SdSParameters,
    bridge_boost,
    characteristic_speeds,
    compact_radius,
    metric_f,
    propagation_coefficient,
    retarded_time_offset,
    sds_horizons,
)


@dataclass(frozen=True)
class FoliationDiagnostic:
    """Three conditioning quantities evaluated from the model formulas."""

    length_over_mass: float
    bridge: str
    maximum_characteristic_speed: float
    minimum_propagation_coefficient: float
    retarded_time_offset: float


def general_retarded_time_offset(
    parameters: SdSParameters,
    bridge: str,
    reference_radius: float = 4.0,
) -> float:
    r"""Return ``q_B = integral_(r0)^rc (1+B)/f dr``.

    Both numerator and denominator vanish at the cosmological horizon.  The
    quadrature evaluates the analytic ratio inside the domain and uses its
    one-sided limit only in the small endpoint neighborhood where direct
    subtraction would lose precision.
    """

    if bridge not in BRIDGE_CHOICES:
        raise ValueError(f"Unknown bridge {bridge!r}.")
    if bridge == "minimal":
        return float(retarded_time_offset(parameters, reference_radius))
    horizons = sds_horizons(parameters)
    if not horizons.black_hole < reference_radius < horizons.cosmological:
        raise ValueError("The reference radius must lie between the horizons.")
    endpoint_width = 1e-8 * parameters.cosmological_length
    sample_radius = horizons.cosmological - 1e-5 * max(
        1.0, parameters.cosmological_length
    )

    def analytic_integrand(radius: float) -> float:
        evaluation_radius = (
            sample_radius
            if horizons.cosmological - radius < endpoint_width
            else radius
        )
        values = np.asarray([evaluation_radius])
        rho = compact_radius(values, parameters)
        numerator = 1.0 + bridge_boost(rho, parameters, bridge)[0]
        return float(numerator / metric_f(values, parameters)[0])

    value, _ = quad(
        analytic_integrand,
        reference_radius,
        horizons.cosmological,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )
    return float(value)


def evaluate_foliation_diagnostic(
    length_over_mass: float,
    bridge: str,
    *,
    radial_points: int = 200001,
) -> FoliationDiagnostic:
    """Evaluate one row from the regularized analytic coefficients."""

    if radial_points < 1001:
        raise ValueError("At least 1001 radial points are required.")
    parameters = SdSParameters(
        mass=1.0, cosmological_length=length_over_mass, ell=2
    )
    rho = np.linspace(0.0, 1.0, radial_points)
    ingoing, outgoing = characteristic_speeds(rho, parameters, bridge)
    coefficient = propagation_coefficient(rho, parameters, bridge)
    return FoliationDiagnostic(
        length_over_mass=float(length_over_mass),
        bridge=bridge,
        maximum_characteristic_speed=float(
            max(np.max(np.abs(ingoing)), np.max(np.abs(outgoing)))
        ),
        minimum_propagation_coefficient=float(np.min(coefficient)),
        retarded_time_offset=general_retarded_time_offset(parameters, bridge),
    )


def evaluate_foliation_table(
    lengths: tuple[float, ...],
    bridges: tuple[str, ...] = BRIDGE_CHOICES,
) -> list[FoliationDiagnostic]:
    """Evaluate the requested Cartesian product of lengths and gauges."""

    return [
        evaluate_foliation_diagnostic(length, bridge)
        for length in lengths
        for bridge in bridges
    ]


def _fixed(decimals: int):
    """Format with a fixed number of decimal places."""

    return lambda value: f"{value:.{decimals}f}"


def _significant(digits: int):
    """Format to a fixed number of significant digits, keeping trailing zeros.

    ``%g`` strips trailing zeros and switches to exponent notation, neither of
    which matches the archived tables, so the decimal count is chosen from the
    magnitude instead.
    """

    def format_value(value: float) -> str:
        if value == 0.0:
            return f"{0.0:.{max(digits - 1, 0)}f}"
        magnitude = int(np.floor(np.log10(abs(value)))) + 1
        return f"{value:.{max(digits - magnitude, 0)}f}"

    return format_value


#: Column order and numeric format of the archived figure data.
FIGURE_TABLES = (
    ("foliation_conditioning.csv", "maximum_characteristic_speed", _fixed(10)),
    ("foliation_min_A.csv", "minimum_propagation_coefficient", _fixed(10)),
    ("foliation_retarded_offsets.csv", "retarded_time_offset", _significant(12)),
)
FIGURE_LENGTHS = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
FIGURE_BRIDGES = (
    "minimum",
    "minimal",
    "linear",
    "modified_linear",
    "mavrogiannis",
    "slow_roll",
)


def write_figure_tables(
    output_dir: Path = Path("paper/figs/data"),
    lengths: tuple[float, ...] = FIGURE_LENGTHS,
    bridges: tuple[str, ...] = FIGURE_BRIDGES,
) -> list[Path]:
    """Regenerate the foliation figure data directly from the formulas."""

    table = evaluate_foliation_table(lengths, bridges)
    lookup = {(row.length_over_mass, row.bridge): row for row in table}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, attribute, format_value in FIGURE_TABLES:
        destination = output_dir / name
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["L_over_M"] + list(bridges))
            for length in lengths:
                writer.writerow(
                    [f"{length:g}"]
                    + [
                        format_value(getattr(lookup[(length, bridge)], attribute))
                        for bridge in bridges
                    ]
                )
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser(
        "write", help="regenerate the foliation figure data from the formulas"
    )
    write.add_argument("--output-dir", type=Path, default=Path("paper/figs/data"))
    show = subparsers.add_parser("show", help="print the table without writing")
    show.add_argument("lengths", nargs="*", type=float, default=list(FIGURE_LENGTHS))
    arguments = parser.parse_args()
    if arguments.command == "write":
        for path in write_figure_tables(arguments.output_dir):
            print(path)
    else:
        for row in evaluate_foliation_table(
            tuple(arguments.lengths), FIGURE_BRIDGES
        ):
            print(
                f"L/M={row.length_over_mass:7g}  {row.bridge:16s}"
                f"  max|speed|={row.maximum_characteristic_speed:.10f}"
                f"  min A={row.minimum_propagation_coefficient:.10f}"
                f"  q={row.retarded_time_offset:.11f}"
            )


if __name__ == "__main__":
    main()
