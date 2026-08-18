"""Analytic bridge foliation and conditioning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

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
