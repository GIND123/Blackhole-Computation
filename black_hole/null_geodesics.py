"""Exact equatorial null ray timing for Schwarzschild and SdS."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    minimal_height as schwarzschild_height,
    retarded_time_offset as schwarzschild_offset,
    tortoise_coordinate as schwarzschild_tortoise,
)
from .sds_model import (
    SdSParameters,
    metric_f,
    minimal_height as sds_height,
    retarded_time_offset as sds_offset,
    sds_horizons,
    tortoise_coordinate as sds_tortoise,
)


REFERENCE_RADIUS = 4.0


@dataclass(frozen=True)
class RayTiming:
    target_angle: float
    winding: int
    turning_radius: float
    impact_parameter: float
    arrival_u: float
    photon_frequency: float

    def as_dict(self) -> dict:
        return {
            "target_angle_over_pi": self.target_angle / np.pi,
            "winding": self.winding,
            "turning_radius_over_M": self.turning_radius,
            "impact_parameter_over_M": self.impact_parameter,
            "predicted_U_over_M": self.arrival_u,
            "Omega_ph_M": self.photon_frequency,
        }


class _RayBackground:
    def __init__(self, cosmological_length: float | None, mass: float) -> None:
        self.mass = float(mass)
        self.length = cosmological_length
        if cosmological_length is None:
            self.parameters = SchwarzschildScalarParameters(mass=mass, ell=0)
            self.outer_radius = np.inf
            self.q = schwarzschild_offset(self.parameters, REFERENCE_RADIUS)
        else:
            self.parameters = SdSParameters(
                mass=mass, cosmological_length=cosmological_length, ell=0
            )
            self.outer_radius = sds_horizons(self.parameters).cosmological
            self.q = sds_offset(self.parameters, REFERENCE_RADIUS)

    def lapse(self, radius: float) -> float:
        if self.length is None:
            return 1.0 - 2.0 * self.mass / radius
        return float(metric_f(np.asarray(radius), self.parameters))

    def height(self, radius: float) -> float:
        if self.length is None:
            return float(
                schwarzschild_height(
                    np.asarray(radius), self.parameters, REFERENCE_RADIUS
                )
            )
        return float(sds_height(np.asarray(radius), self.parameters, REFERENCE_RADIUS))

    def tortoise(self, radius: float) -> float:
        if self.length is None:
            return float(
                schwarzschild_tortoise(
                    np.asarray(radius), self.parameters, REFERENCE_RADIUS
                )
            )
        return float(
            sds_tortoise(np.asarray(radius), self.parameters, REFERENCE_RADIUS)
        )

    @property
    def photon_frequency(self) -> float:
        factor = 1.0
        if self.length is not None:
            factor -= 27.0 * self.mass**2 / self.length**2
        return np.sqrt(factor) / (3.0 * np.sqrt(3.0) * self.mass)


def _turning_integral(
    background: _RayBackground,
    turning_radius: float,
    endpoint: float,
    impact_parameter: float,
    kind: str,
) -> float:
    upper = np.inf if np.isinf(endpoint) else np.sqrt(endpoint - turning_radius)
    f0 = background.lapse(turning_radius)
    derivative_q = (
        2.0
        * impact_parameter**2
        * (turning_radius - 3.0 * background.mass)
        / turning_radius**4
    )

    def integrand(x: float) -> float:
        if x == 0.0:
            root_factor = np.sqrt(max(derivative_q, np.finfo(float).tiny))
            if kind == "angle":
                return 2.0 * impact_parameter / (turning_radius**2 * root_factor)
            return 2.0 / (f0 * root_factor)
        radius = turning_radius + x * x
        lapse = background.lapse(radius)
        difference = radius - turning_radius
        numerator = difference * (
            2.0
            * turning_radius**2
            * (turning_radius - 3.0 * background.mass)
            + 3.0
            * turning_radius
            * (turning_radius - 2.0 * background.mass)
            * difference
            + (turning_radius - 2.0 * background.mass) * difference**2
        )
        radial_factor = max(
            impact_parameter**2
            * numerator
            / (turning_radius**3 * radius**3),
            np.finfo(float).tiny,
        )
        if kind == "angle":
            value = impact_parameter / (radius**2 * np.sqrt(radial_factor))
        elif kind == "time":
            value = 1.0 / (lapse * np.sqrt(radial_factor))
        elif kind == "retarded":
            if abs(lapse) < 1e-8:
                value = impact_parameter**2 / (2.0 * radius**2)
            else:
                value = (1.0 / np.sqrt(radial_factor) - 1.0) / lapse
        else:
            raise ValueError(f"Unknown ray integral kind {kind!r}.")
        return 2.0 * x * value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value = float(
            quad(
                integrand,
                0.0,
                upper,
                epsabs=2e-10,
                epsrel=2e-10,
                limit=500,
            )[0]
        )
    if not np.isfinite(value):
        raise FloatingPointError("The null ray integral is not finite.")
    return value


def _impact_parameter(background: _RayBackground, turning_radius: float) -> float:
    return turning_radius / np.sqrt(background.lapse(turning_radius))


def _outward_integral(
    background: _RayBackground,
    source_radius: float,
    endpoint: float,
    impact_parameter: float,
    kind: str,
) -> float:
    upper = np.inf if np.isinf(endpoint) else np.sqrt(endpoint - source_radius)

    def integrand(x: float) -> float:
        radius = source_radius + x * x
        lapse = background.lapse(radius)
        radial_factor = max(
            1.0 - impact_parameter**2 * lapse / radius**2,
            np.finfo(float).tiny,
        )
        if kind == "angle":
            value = impact_parameter / (radius**2 * np.sqrt(radial_factor))
        elif kind == "time":
            value = 1.0 / (lapse * np.sqrt(radial_factor))
        elif kind == "retarded":
            if abs(lapse) < 1e-8:
                value = impact_parameter**2 / (2.0 * radius**2)
            else:
                value = (1.0 / np.sqrt(radial_factor) - 1.0) / lapse
        else:
            raise ValueError(f"Unknown ray integral kind {kind!r}.")
        return 2.0 * x * value

    return float(quad(integrand, 0.0, upper, epsabs=2e-10, epsrel=2e-10, limit=500)[0])


def _total_angle(
    background: _RayBackground,
    source_radius: float,
    observer_radius: float,
    turning_radius: float,
) -> float:
    impact = _impact_parameter(background, turning_radius)
    return _turning_integral(
        background, turning_radius, source_radius, impact, "angle"
    ) + _turning_integral(
        background, turning_radius, observer_radius, impact, "angle"
    )


def trace_null_ray(
    *,
    source_radius: float,
    observer_radius: float | None,
    target_angle: float,
    emission_time: float,
    cosmological_length: float | None = None,
    mass: float = 1.0,
    winding: int = 0,
) -> RayTiming:
    """Return the full arrival time of one inward turning null ray."""

    background = _RayBackground(cosmological_length, mass)
    observer = background.outer_radius if observer_radius is None else float(observer_radius)
    if target_angle <= 0.0:
        if observer_radius is None:
            arrival = emission_time - background.tortoise(source_radius)
        else:
            arrival = (
                emission_time
                + background.tortoise(observer)
                - background.tortoise(source_radius)
                + background.height(observer)
                - background.q
            )
        return RayTiming(
            target_angle=0.0,
            winding=winding,
            turning_radius=np.nan,
            impact_parameter=0.0,
            arrival_u=float(arrival),
            photon_frequency=background.photon_frequency,
        )

    if winding == 0:
        maximum_impact = source_radius / np.sqrt(background.lapse(source_radius))

        def angle_residual(impact: float) -> float:
            return _outward_integral(
                background, source_radius, observer, impact, "angle"
            ) - target_angle

        upper_impact = maximum_impact * (1.0 - 1e-10)
        if angle_residual(upper_impact) >= 0.0:
            impact = float(brentq(angle_residual, 0.0, upper_impact, xtol=1e-12))
            if observer_radius is None:
                arrival = (
                    emission_time
                    + _outward_integral(
                        background, source_radius, observer, impact, "retarded"
                    )
                    - background.tortoise(source_radius)
                )
            else:
                arrival = (
                    emission_time
                    + _outward_integral(
                        background, source_radius, observer, impact, "time"
                    )
                    + background.height(observer)
                    - background.q
                )
            return RayTiming(
                target_angle=float(target_angle),
                winding=winding,
                turning_radius=np.nan,
                impact_parameter=impact,
                arrival_u=float(arrival),
                photon_frequency=background.photon_frequency,
            )

    photon_radius = 3.0 * mass
    upper_endpoint = min(source_radius, observer)
    lower = photon_radius * (1.0 + 1e-10)
    upper = photon_radius + 0.99 * (upper_endpoint - photon_radius)

    def residual(turning_radius: float) -> float:
        return (
            _total_angle(
                background, source_radius, observer, turning_radius
            )
            - target_angle
        )

    if residual(upper) > 0.0:
        raise ValueError(
            "The requested angle is below the minimum for an inward turning ray."
        )
    turning = float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-13))
    impact = _impact_parameter(background, turning)
    inward_time = _turning_integral(
        background, turning, source_radius, impact, "time"
    )
    if observer_radius is None:
        outward_retarded = _turning_integral(
            background, turning, observer, impact, "retarded"
        ) - background.tortoise(turning)
        arrival = emission_time + inward_time + outward_retarded
    else:
        outward_time = _turning_integral(
            background, turning, observer, impact, "time"
        )
        arrival = (
            emission_time
            + inward_time
            + outward_time
            + background.height(observer)
            - background.q
        )
    return RayTiming(
        target_angle=float(target_angle),
        winding=winding,
        turning_radius=turning,
        impact_parameter=impact,
        arrival_u=float(arrival),
        photon_frequency=background.photon_frequency,
    )


def generic_target_angle(gamma: float, pulse: int) -> float:
    """Return the alternating angular path length for a generic observer."""

    if pulse < 0 or not 0.0 <= gamma <= np.pi:
        raise ValueError("Pulse must be nonnegative and gamma must lie in [0, pi].")
    turns = pulse // 2
    if pulse % 2 == 0:
        return 2.0 * turns * np.pi + gamma
    return 2.0 * (turns + 1) * np.pi - gamma
