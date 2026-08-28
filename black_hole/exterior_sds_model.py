"""Exterior-supported Schwarzschild--de Sitter scalar-wave coefficients.

The geometry is exactly Schwarzschild through an inner compact region and
becomes exactly Schwarzschild--de Sitter before the cosmological horizon.  A
``C-infinity`` switch in the Chebyshev endpoint angle joins the two regions.
The original horizon-scaled switch is retained through ``L/M=160``.  Above
that value both the transition and the exact-SdS cap retain their reference
Chebyshev-angle widths.  Neither outer region therefore collapses under
endpoint clustering.  This module is
deliberately independent of the archived uniform-SdS model so the new
experiment cannot alter the frozen controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .sds_model import (
    ArealBumpInitialData,
    ArealVelocityBumpInitialData,
    ScalarInitialData,
    SdSParameters,
    compact_areal_profile,
    compact_areal_velocity_profile,
    sds_horizons,
)


# Through L/M=160, the exact-SdS region starts at R1=0.9*r_c and the
# transition and cap have equal Chebyshev-angle widths.  Thereafter both
# widths are held at their L/M=160 value.  This leaves every completed
# L/M<=160 background unchanged and prevents either outer region from
# collapsing on the global Chebyshev grid.
TRANSITION_OUTER_HORIZON_FRACTION = 0.9
TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M = 160.0
TRANSITION_MINIMUM_ANGLE_WIDTH = 0.07526440137447271


@dataclass(frozen=True)
class ExteriorSdSParameters:
    """Parameters for the exterior-supported artificial cosmology.

    The cosmological horizon ``r_c`` is the outer positive root of the
    corresponding uniform-SdS lapse.  The black-hole horizon is exactly
    ``2M`` because the cosmological term vanishes throughout the inner region.
    """

    mass: float = 1.0
    cosmological_length: float = 80.0
    ell: int = 2
    curvature_coupling: float = 0.0

    def __post_init__(self) -> None:
        # Reuse the uniform model's physical-domain validation and root
        # calculation.  The exterior construction has the same Nariai bound.
        SdSParameters(
            mass=self.mass,
            cosmological_length=self.cosmological_length,
            ell=self.ell,
            curvature_coupling=self.curvature_coupling,
        )
        r0, r1 = transition_radii(self)
        rc = self.cosmological_horizon
        if not 2.0 * self.mass < r0 < r1 < rc:
            raise ValueError(
                "The exterior transition must satisfy 2M < R0 < R1 < r_c; "
                "increase L/M."
            )

    @property
    def black_hole_horizon(self) -> float:
        return 2.0 * self.mass

    @property
    def cosmological_horizon(self) -> float:
        uniform = SdSParameters(
            mass=self.mass,
            cosmological_length=self.cosmological_length,
            ell=self.ell,
        )
        return sds_horizons(uniform).cosmological

    def as_dict(self) -> dict[str, float | int | str | bool]:
        r0, r1 = transition_radii(self)
        rho0, rho1 = transition_compact_radii(self)
        rc = self.cosmological_horizon
        return {
            "mass": self.mass,
            "cosmological_length": self.cosmological_length,
            "ell": self.ell,
            "curvature_coupling": self.curvature_coupling,
            "black_hole_horizon": self.black_hole_horizon,
            "cosmological_horizon": self.cosmological_horizon,
            "transition_inner_radius": r0,
            "transition_outer_radius": r1,
            "transition_inner_rho": rho0,
            "transition_outer_rho": rho1,
            "transition_compact_width": rho1 - rho0,
            "outer_cap_compact_width": 1.0 - rho1,
            "transition_width_reference_length_over_M": (
                TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M
            ),
            "transition_outer_horizon_fraction": (
                TRANSITION_OUTER_HORIZON_FRACTION
            ),
            "transition_minimum_angle_width": TRANSITION_MINIMUM_ANGLE_WIDTH,
            "transition_width_floor_active": (
                self.cosmological_length / self.mass
                > TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M
            ),
            "actual_transition_outer_horizon_fraction": r1 / rc,
            "transition": (
                "C-infinity fixed-minimum transition-and-cap angle widths"
            ),
        }


def _uniform_parameters(parameters: ExteriorSdSParameters) -> SdSParameters:
    return SdSParameters(
        mass=parameters.mass,
        cosmological_length=parameters.cosmological_length,
        ell=parameters.ell,
    )


def compactification_scale(parameters: ExteriorSdSParameters) -> float:
    """Return ``D=1-2M/r_c`` in ``rho=(1-2M/r)/D``."""

    return 1.0 - 2.0 * parameters.mass / parameters.cosmological_horizon


def areal_radius(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Map ``rho in [0,1]`` to ``r in [2M,r_c]`` analytically."""

    rho = np.asarray(rho, dtype=float)
    mass = parameters.mass
    rc = parameters.cosmological_horizon
    radius = 2.0 * mass / (1.0 - compactification_scale(parameters) * rho)
    radius = np.where(rho == 0.0, 2.0 * mass, radius)
    return np.where(rho == 1.0, rc, radius)


def compact_radius(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Map areal radius to the exterior construction's compact coordinate."""

    radius = np.asarray(radius, dtype=float)
    return (
        1.0 - 2.0 * parameters.mass / radius
    ) / compactification_scale(parameters)


def compactification_derivative(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``d rho/d r``."""

    radius = np.asarray(radius, dtype=float)
    return (
        2.0
        * parameters.mass
        / (compactification_scale(parameters) * radius**2)
    )


def transition_outer_radius(parameters: ExteriorSdSParameters) -> float:
    r"""Return the start of the exact-SdS outer cap."""

    _, rho1 = transition_compact_radii(parameters)
    return float(areal_radius(np.array(rho1), parameters))


def chebyshev_angle(rho: np.ndarray) -> np.ndarray:
    r"""Return the endpoint angle ``theta=acos(2 rho-1)``.

    Chebyshev--Lobatto nodes are uniformly spaced in this angle.  Defining the
    transition in ``theta`` therefore prevents an apparently narrow compact
    layer from receiving far fewer nodes than the final exact-SdS cap.
    """

    rho = np.asarray(rho, dtype=float)
    return np.arccos(np.clip(2.0 * rho - 1.0, -1.0, 1.0))


def transition_compact_radii(
    parameters: ExteriorSdSParameters,
) -> tuple[float, float]:
    """Return endpoints with nonvanishing transition and cap angle widths."""

    horizon_scaled_r1 = (
        TRANSITION_OUTER_HORIZON_FRACTION
        * parameters.cosmological_horizon
    )
    horizon_scaled_rho1 = float(
        compact_radius(np.array(horizon_scaled_r1), parameters)
    )
    horizon_scaled_theta1 = float(
        chebyshev_angle(np.array(horizon_scaled_rho1))
    )
    theta1 = max(horizon_scaled_theta1, TRANSITION_MINIMUM_ANGLE_WIDTH)
    theta0 = 2.0 * theta1
    rho1 = 0.5 * (1.0 + np.cos(theta1))
    rho0 = 0.5 * (1.0 + np.cos(theta0))
    return float(rho0), float(rho1)


def transition_radii(
    parameters: ExteriorSdSParameters,
) -> tuple[float, float]:
    """Return the areal radii ``(R0,R1)`` bounding the transition."""

    rho0, rho1 = transition_compact_radii(parameters)
    r0 = float(areal_radius(np.array(rho0), parameters))
    r1 = float(areal_radius(np.array(rho1), parameters))
    return float(r0), float(r1)


def smooth_step(x: np.ndarray) -> np.ndarray:
    r"""Standard ``C-infinity`` step, zero below 0 and one above 1."""

    x = np.asarray(x, dtype=float)
    value = np.zeros_like(x)
    value[x >= 1.0] = 1.0
    interior = (x > 0.0) & (x < 1.0)
    xi = x[interior]
    log_left = -1.0 / xi
    log_right = -1.0 / (1.0 - xi)
    value[interior] = np.exp(
        log_left - np.logaddexp(log_left, log_right)
    )
    return value


def smooth_step_derivative(x: np.ndarray) -> np.ndarray:
    """Analytic derivative of :func:`smooth_step`."""

    x = np.asarray(x, dtype=float)
    derivative = np.zeros_like(x)
    interior = (x > 0.0) & (x < 1.0)
    xi = x[interior]
    log_left = -1.0 / xi
    log_right = -1.0 / (1.0 - xi)
    log_normalization = np.logaddexp(log_left, log_right)
    logit_prime = 1.0 / xi**2 + 1.0 / (1.0 - xi) ** 2
    derivative[interior] = np.exp(
        log_left
        + log_right
        - 2.0 * log_normalization
        + np.log(logit_prime)
    )
    return derivative


def smooth_step_second_derivative(x: np.ndarray) -> np.ndarray:
    """Analytic second derivative of :func:`smooth_step`."""

    x = np.asarray(x, dtype=float)
    derivative = np.zeros_like(x)
    interior = (x > 0.0) & (x < 1.0)
    xi = x[interior]
    step = smooth_step(xi)
    q = 1.0 / xi**2 + 1.0 / (1.0 - xi) ** 2
    q_prime = -2.0 / xi**3 + 2.0 / (1.0 - xi) ** 3
    derivative[interior] = step * (1.0 - step) * (
        (1.0 - 2.0 * step) * q**2 + q_prime
    )
    return derivative


def transition_profile(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the exterior-support switch ``chi_L(rho)``."""

    rho = np.asarray(rho, dtype=float)
    rho0, rho1 = transition_compact_radii(parameters)
    theta = chebyshev_angle(rho)
    theta0 = float(chebyshev_angle(np.array(rho0)))
    theta1 = float(chebyshev_angle(np.array(rho1)))
    return smooth_step((theta0 - theta) / (theta0 - theta1))


def transition_rho_derivative(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``d chi/d rho``."""

    rho = np.asarray(rho, dtype=float)
    rho0, rho1 = transition_compact_radii(parameters)
    derivative = np.zeros_like(rho)
    interior = (rho > rho0) & (rho < rho1)
    theta0 = float(chebyshev_angle(np.array(rho0)))
    theta1 = float(chebyshev_angle(np.array(rho1)))
    theta = chebyshev_angle(rho[interior])
    x = (theta0 - theta) / (theta0 - theta1)
    dx_drho = 1.0 / (
        (theta0 - theta1)
        * np.sqrt(rho[interior] * (1.0 - rho[interior]))
    )
    derivative[interior] = smooth_step_derivative(x) * dx_drho
    return derivative


def transition_rho_second_derivative(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``d^2 chi/d rho^2``."""

    rho = np.asarray(rho, dtype=float)
    rho0, rho1 = transition_compact_radii(parameters)
    derivative = np.zeros_like(rho)
    interior = (rho > rho0) & (rho < rho1)
    theta0 = float(chebyshev_angle(np.array(rho0)))
    theta1 = float(chebyshev_angle(np.array(rho1)))
    width = theta0 - theta1
    rho_interior = rho[interior]
    theta = chebyshev_angle(rho_interior)
    x = (theta0 - theta) / width
    product = rho_interior * (1.0 - rho_interior)
    dx_drho = 1.0 / (width * np.sqrt(product))
    d2x_drho2 = (2.0 * rho_interior - 1.0) / (
        2.0 * width * product ** 1.5
    )
    derivative[interior] = (
        smooth_step_second_derivative(x) * dx_drho**2
        + smooth_step_derivative(x) * d2x_drho2
    )
    return derivative


def transition_radial_derivative(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``d chi/d r``."""

    radius = np.asarray(radius, dtype=float)
    rho = compact_radius(radius, parameters)
    return transition_rho_derivative(
        rho, parameters
    ) * compactification_derivative(radius, parameters)


def transition_radial_second_derivative(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``d^2 chi/d r^2``."""

    radius = np.asarray(radius, dtype=float)
    rho = compact_radius(radius, parameters)
    rho_prime = compactification_derivative(radius, parameters)
    rho_second = -2.0 * rho_prime / radius
    return (
        transition_rho_second_derivative(rho, parameters) * rho_prime**2
        + transition_rho_derivative(rho, parameters) * rho_second
    )


def ricci_scalar(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return the Ricci scalar of the exterior-supported metric.

    For ``f=1-2M/r-r^2 chi/L^2``,

    .. math::

       R_\chi=L^{-2}(12\chi+8r\chi'+r^2\chi'').
    """

    radius = np.asarray(radius, dtype=float)
    rho = compact_radius(radius, parameters)
    chi = transition_profile(rho, parameters)
    chi_prime = transition_radial_derivative(radius, parameters)
    chi_second = transition_radial_second_derivative(radius, parameters)
    return (
        12.0 * chi + 8.0 * radius * chi_prime + radius**2 * chi_second
    ) / parameters.cosmological_length**2


def metric_f(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return ``f_chi=1-2M/r-r^2 chi/L^2`` without horizon cancellation."""

    radius = np.asarray(radius, dtype=float)
    rho = compact_radius(radius, parameters)
    chi = transition_profile(rho, parameters)
    mass = parameters.mass
    rc = parameters.cosmological_horizon
    compactification = compactification_scale(parameters)
    delta_c = (
        rc
        * compactification
        * (1.0 - rho)
        / (1.0 - compactification * rho)
    )
    cap_factor = (
        (rc + radius) / parameters.cosmological_length**2
        - 2.0 * mass / (radius * rc)
    )
    value = (
        delta_c * cap_factor
        + radius**2 * (1.0 - chi) / parameters.cosmological_length**2
    )
    # In the exact inner region, use f_Schw=D*rho directly.  The general
    # positive-factor expression is used throughout the transition and cap.
    return np.where(chi == 0.0, compactification * rho, value)


def metric_f_prime(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic radial derivative of the exterior lapse."""

    radius = np.asarray(radius, dtype=float)
    rho = compact_radius(radius, parameters)
    chi = transition_profile(rho, parameters)
    chi_prime = transition_radial_derivative(radius, parameters)
    return (
        2.0 * parameters.mass / radius**2
        - (2.0 * radius * chi + radius**2 * chi_prime)
        / parameters.cosmological_length**2
    )


def bridge_one_plus_boost(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return ``1+B_chi`` in a cancellation-free endpoint form."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    chi = transition_profile(rho, parameters)
    mass = parameters.mass
    rc = parameters.cosmological_horizon
    compactification = compactification_scale(parameters)
    delta_c = (
        rc
        * compactification
        * (1.0 - rho)
        / (1.0 - compactification * rho)
    )
    return 8.0 * mass**2 / rc**2 * (
        delta_c * (rc + radius) / radius**2 + (1.0 - chi)
    )


def bridge_one_minus_boost(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return ``1-B_chi`` in a cancellation-free endpoint form."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    chi = transition_profile(rho, parameters)
    mass = parameters.mass
    rc = parameters.cosmological_horizon
    compactification = compactification_scale(parameters)
    schwarzschild = (
        2.0
        * compactification
        * rho
        * (radius + 2.0 * mass)
        / radius
    )
    cosmological = 8.0 * mass**2 * chi / rc**2
    return schwarzschild + cosmological


def bridge_boost(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return the exterior minimal bridge boost ``B_chi``."""

    one_plus = bridge_one_plus_boost(rho, parameters)
    one_minus = bridge_one_minus_boost(rho, parameters)
    return 0.5 * (one_plus - one_minus)


def bridge_boost_radial_derivative(
    radius: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``dB_chi/dr``."""

    radius = np.asarray(radius, dtype=float)
    mass = parameters.mass
    rc = parameters.cosmological_horizon
    return (
        -16.0 * mass**2 / radius**3
        - 8.0
        * mass**2
        * transition_radial_derivative(radius, parameters)
        / rc**2
    )


def propagation_endpoint_coefficients(
    parameters: ExteriorSdSParameters,
) -> tuple[float, float]:
    """Return exact l'Hopital limits of ``A`` at both horizons."""

    mass = parameters.mass
    rc = parameters.cosmological_horizon
    compactification = compactification_scale(parameters)
    left = 1.0 / (16.0 * mass * compactification)
    right = (rc - 3.0 * mass) / (8.0 * mass * (rc - 2.0 * mass))
    return float(left), float(right)


def propagation_coefficient(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return regular ``A=(f d rho/dr)/(1-B^2)``."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    chi = transition_profile(rho, parameters)
    speed = metric_f(radius, parameters) * compactification_derivative(
        radius, parameters
    )
    one_plus = bridge_one_plus_boost(rho, parameters)
    one_minus = bridge_one_minus_boost(rho, parameters)
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = speed / (one_minus * one_plus)
    mass = parameters.mass
    rc = parameters.cosmological_horizon
    compactification = compactification_scale(parameters)
    inner = radius / (
        8.0 * mass * compactification * (radius + 2.0 * mass)
    )
    cap_factor = (
        (rc + radius) / parameters.cosmological_length**2
        - 2.0 * mass / (radius * rc)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = (
            rc**2
            * cap_factor
            / (
                4.0
                * mass
                * compactification
                * (rc + radius)
                * one_minus
            )
        )
    coefficient = np.where(chi == 0.0, inner, coefficient)
    coefficient = np.where(chi == 1.0, cap, coefficient)
    left, right = propagation_endpoint_coefficients(parameters)
    coefficient = np.where(rho == 0.0, left, coefficient)
    return np.where(rho == 1.0, right, coefficient)


def rescaled_scalar_potential(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> np.ndarray:
    r"""Return the reduced potential for ``(Box-xi R_chi)Phi=0``."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    ell = parameters.ell
    compactification = compactification_scale(parameters)
    mass = parameters.mass
    # Since d rho/dr=2M/(D r^2), this form avoids separate small terms at the
    # cosmological horizon and remains well conditioned as L grows.
    potential = compactification / (2.0 * mass) * (
        ell * (ell + 1.0)
        + radius * metric_f_prime(radius, parameters)
        + parameters.curvature_coupling
        * radius**2
        * ricci_scalar(radius, parameters)
    )
    left = compactification / (2.0 * mass) * (ell * (ell + 1.0) + 1.0)
    right = compactification / (2.0 * mass) * (
        ell * (ell + 1.0)
        - 2.0
        + 6.0 * mass / parameters.cosmological_horizon
        + parameters.curvature_coupling
        * 12.0
        * parameters.cosmological_horizon**2
        / parameters.cosmological_length**2
    )
    potential = np.where(rho == 0.0, left, potential)
    return np.where(rho == 1.0, right, potential)


def characteristic_speeds(
    rho: np.ndarray, parameters: ExteriorSdSParameters
) -> tuple[np.ndarray, np.ndarray]:
    """Return ingoing and outgoing radial light speeds ``d rho/d tau``."""

    coefficient = propagation_coefficient(rho, parameters)
    one_plus = bridge_one_plus_boost(rho, parameters)
    one_minus = bridge_one_minus_boost(rho, parameters)
    return -coefficient * one_plus, coefficient * one_minus


def scalar_gaussian_initial_data(
    rho: np.ndarray,
    parameters: ExteriorSdSParameters,
    initial: ScalarInitialData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the legacy compact-coordinate Gaussian data."""

    rho = np.asarray(rho, dtype=float)
    displacement = rho - initial.center_fraction
    u = np.exp(-(displacement**2) / (2.0 * initial.width**2))
    psi = -displacement * u / initial.width**2
    if initial.time_symmetric:
        pi = -bridge_boost(rho, parameters) * psi
    else:
        pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi


def scalar_areal_bump_initial_data(
    rho: np.ndarray,
    parameters: ExteriorSdSParameters,
    initial: ArealBumpInitialData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the common compact areal profile on the exterior background."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    support_left = initial.center_radius - initial.support_half_width
    support_right = initial.center_radius + initial.support_half_width
    if not (
        parameters.black_hole_horizon
        < support_left
        < support_right
        < parameters.cosmological_horizon
    ):
        raise ValueError("The compact areal-radius pulse must lie between horizons.")
    u, du_dr = compact_areal_profile(radius, initial)
    psi = du_dr / compactification_derivative(radius, parameters)
    if initial.time_symmetric:
        pi = -bridge_boost(rho, parameters) * psi
    else:
        pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi


def scalar_areal_velocity_initial_data(
    rho: np.ndarray,
    parameters: ExteriorSdSParameters,
    initial: ArealVelocityBumpInitialData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``u=psi=0`` and ``pi=G(r)/A_chi(r)``."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    support_left = initial.center_radius - initial.support_half_width
    support_right = initial.center_radius + initial.support_half_width
    if (
        support_left <= parameters.black_hole_horizon
        or support_right >= parameters.cosmological_horizon
    ):
        raise ValueError("The physical velocity bump must lie between horizons.")
    velocity = compact_areal_velocity_profile(radius, initial)
    coefficient = propagation_coefficient(rho, parameters)
    u = np.zeros_like(rho)
    psi = np.zeros_like(rho)
    pi = np.zeros_like(rho)
    support = velocity != 0.0
    pi[support] = velocity[support] / coefficient[support]
    if not np.all(np.isfinite(pi)):
        raise FloatingPointError("Non-finite momentum in physical velocity data.")
    return u, psi, pi


def retarded_time_offset(
    parameters: ExteriorSdSParameters, reference_radius: float
) -> float:
    r"""Return ``q_chi=int_(r_ref)^(r_c) (1+B_chi)/f_chi dr``.

    The integrand's removable cosmological-horizon limit is assigned
    analytically, and the quadrature is split at both transition boundaries.
    """

    rb = parameters.black_hole_horizon
    rc = parameters.cosmological_horizon
    reference = float(reference_radius)
    if not rb < reference < rc:
        raise ValueError("The retarded-time reference radius must lie between horizons.")

    right_limit = 8.0 * parameters.mass**2 / (
        rc * (rc - 3.0 * parameters.mass)
    )
    r0, r1 = transition_radii(parameters)

    def integrand(radius: float) -> float:
        if radius == rc:
            return right_limit
        if radius >= r1:
            cap_factor = (
                (rc + radius) / parameters.cosmological_length**2
                - 2.0 * parameters.mass / (radius * rc)
            )
            return (
                8.0
                * parameters.mass**2
                * (rc + radius)
                / (radius**2 * rc**2 * cap_factor)
            )
        rho = float(compact_radius(np.array(radius), parameters))
        numerator = float(
            bridge_one_plus_boost(np.array(rho), parameters)
        )
        denominator = float(metric_f(np.array(radius), parameters))
        if denominator == 0.0:
            return right_limit
        return numerator / denominator

    boundaries = [reference]
    boundaries.extend(value for value in (r0, r1) if reference < value < rc)
    boundaries.append(rc)
    total = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        contribution, _ = quad(
            integrand,
            left,
            right,
            epsabs=2.0e-12 * parameters.mass,
            epsrel=2.0e-12,
            limit=300,
        )
        total += contribution
    return float(total)


def background_audit(
    parameters: ExteriorSdSParameters, sample_count: int = 20_001
) -> dict[str, float | int | bool]:
    """Return inexpensive regularity and resolution-independent diagnostics."""

    if sample_count < 101:
        raise ValueError("sample_count must be at least 101.")
    rho = np.linspace(0.0, 1.0, sample_count)
    radius = areal_radius(rho, parameters)
    lapse = metric_f(radius, parameters)
    boost = bridge_boost(rho, parameters)
    coefficient = propagation_coefficient(rho, parameters)
    potential = rescaled_scalar_potential(rho, parameters)
    rho0, rho1 = transition_compact_radii(parameters)
    r0, r1 = transition_radii(parameters)
    return {
        "sample_count": sample_count,
        "transition_inner_radius": r0,
        "transition_outer_radius": r1,
        "transition_inner_rho": rho0,
        "transition_outer_rho": rho1,
        "transition_rho_width": rho1 - rho0,
        "outer_cap_rho_width": 1.0 - rho1,
        "minimum_interior_f": float(np.min(lapse[1:-1])),
        "maximum_interior_abs_boost": float(np.max(np.abs(boost[1:-1]))),
        "minimum_A": float(np.min(coefficient)),
        "maximum_A": float(np.max(coefficient)),
        "maximum_abs_P": float(np.max(np.abs(potential))),
        "minimum_P": float(np.min(potential)),
        "finite_coefficients": bool(
            np.all(np.isfinite(coefficient)) and np.all(np.isfinite(potential))
        ),
        "positive_interior_lapse": bool(np.all(lapse[1:-1] > 0.0)),
        "spacelike_bridge_interior": bool(np.all(np.abs(boost[1:-1]) < 1.0)),
        "nonnegative_scalar_potential": bool(np.all(potential >= 0.0)),
    }
