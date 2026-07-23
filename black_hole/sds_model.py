"""Schwarzschild-de Sitter scalar-wave coefficients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import asin, pi, sin, sqrt

import numpy as np

BRIDGE_CHOICES = (
    "minimum",
    "minimal",
    "linear",
    "modified_linear",
    "mavrogiannis",
    "slow_roll",
)

BRIDGE_LABELS = {
    "minimum": "minimum height",
    "minimal": "minimal gauge",
    "linear": "linear boost",
    "modified_linear": "flat-limit linear",
    "mavrogiannis": "Mavrogiannis",
    "slow_roll": "slow-roll",
}


@dataclass(frozen=True)
class SdSParameters:
    """Physical parameters for a scalar mode on Schwarzschild-de Sitter."""

    mass: float = 1.0
    cosmological_length: float = 10.0
    ell: int = 2

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("Black-hole mass must be positive.")
        if self.cosmological_length <= 0:
            raise ValueError("Cosmological length must be positive.")
        upper_mass = self.cosmological_length / (3.0 * sqrt(3.0))
        if self.mass >= upper_mass:
            raise ValueError(
                "Schwarzschild-de Sitter requires M < L/(3*sqrt(3))."
            )
        if self.ell < 0:
            raise ValueError("Scalar spherical-harmonic index ell must be >= 0.")

    @property
    def cosmological_constant(self) -> float:
        return 3.0 / self.cosmological_length**2

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SdSHorizons:
    """Horizon radii and positive surface gravities."""

    black_hole: float
    cosmological: float
    unphysical: float
    kappa_black_hole: float
    kappa_cosmological: float
    kappa_unphysical: float

    @property
    def width(self) -> float:
        return self.cosmological - self.black_hole

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ScalarInitialData:
    """Localized Gaussian data for the reduced scalar field ``u=r Phi``.

    The center and width are specified in the compact coordinate ``rho``.
    This legacy profile is retained for the bridge-comparison study.  The
    controlled flat-limit study instead uses :class:`ArealBumpInitialData`,
    because equal profiles in ``rho`` are not equal physical pulses when the
    map ``rho(r)`` depends on the cosmological length.
    """

    center_fraction: float = 0.45
    width: float = 0.06
    time_symmetric: bool = True
    pi_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.center_fraction < 1.0:
            raise ValueError("Gaussian center fraction must lie inside (0, 1).")
        if self.width <= 0:
            raise ValueError("Gaussian width must be positive.")

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ArealBumpInitialData:
    r"""Smooth compactly supported initial data for ``u=r Phi``.

    The normalized standard bump is

    .. math::

       u(r)=\exp\left[1-\frac{1}{1-x^2}\right],\qquad
       x=\frac{r-r_0}{w},\quad |x|<1,

    and is zero for ``|x| >= 1``.  It is infinitely differentiable at the
    support boundary and has unit amplitude at ``r=r_0``.
    """

    center_radius: float = 4.0
    support_half_width: float = 1.5
    time_symmetric: bool = True
    pi_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if self.center_radius <= 0.0:
            raise ValueError("Areal-radius pulse center must be positive.")
        if self.support_half_width <= 0.0:
            raise ValueError("Areal-radius pulse half-width must be positive.")
        if self.center_radius <= self.support_half_width:
            raise ValueError("The compact pulse support must have positive radius.")

    def as_dict(self) -> dict[str, float | bool | str]:
        return {"profile": "C-infinity areal-radius bump", **asdict(self)}


@dataclass(frozen=True)
class ArealVelocityBumpInitialData:
    r"""Initially dynamical, physically matched data for ``u=r Phi``.

    The displacement and its radial derivative vanish initially,

    .. math::

       u(0,r)=0,\qquad \psi(0,r)=0,

    while the physical Killing-time velocity is the same smooth compact
    bump ``G(r)`` on every background:

    .. math::

       \partial_t u(0,r)=\partial_\tau u(0,r)=G(r).

    Since the first-order evolution equation gives
    ``partial_tau u=A(B psi+pi)``, the evolved momentum is initialized as
    ``pi=G/A``.  The support is specified in areal radius and must lie
    strictly between the physical horizons.
    """

    center_radius: float = 6.0
    support_half_width: float = 3.0
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.center_radius <= 0.0:
            raise ValueError("Areal-radius velocity center must be positive.")
        if self.support_half_width <= 0.0:
            raise ValueError("Velocity-bump half-width must be positive.")
        if self.center_radius <= self.support_half_width:
            raise ValueError("The velocity-bump support must have positive radius.")
        if not np.isfinite(self.amplitude) or self.amplitude == 0.0:
            raise ValueError("Velocity-bump amplitude must be finite and nonzero.")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "profile": "C-infinity physical areal-radius velocity bump",
            "displacement": "u=0",
            "radial_derivative": "psi=0",
            "momentum": "pi=G(r)/A",
            **asdict(self),
        }


def compact_areal_profile(
    radius: np.ndarray, initial: ArealBumpInitialData
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common compact bump ``(u, du/dr)`` in areal radius."""

    radius = np.asarray(radius, dtype=float)
    x = (radius - initial.center_radius) / initial.support_half_width
    inside = np.abs(x) < 1.0
    u = np.zeros_like(radius)
    du_dr = np.zeros_like(radius)
    x_inside = x[inside]
    denominator = 1.0 - x_inside**2
    u[inside] = np.exp(1.0 - 1.0 / denominator)
    du_dr[inside] = (
        -2.0
        * x_inside
        * u[inside]
        / (initial.support_half_width * denominator**2)
    )
    return u, du_dr


def compact_areal_velocity_profile(
    radius: np.ndarray, initial: ArealVelocityBumpInitialData
) -> np.ndarray:
    """Return the common physical initial velocity ``G(r)``."""

    profile_data = ArealBumpInitialData(
        center_radius=initial.center_radius,
        support_half_width=initial.support_half_width,
    )
    profile, _ = compact_areal_profile(radius, profile_data)
    return initial.amplitude * profile


def sds_horizons(parameters: SdSParameters) -> SdSHorizons:
    """Return the three real roots and positive surface gravities."""

    mass = parameters.mass
    length = parameters.cosmological_length
    argument = 3.0 * sqrt(3.0) * mass / length
    angle = asin(argument)
    black_hole = 2.0 * length / sqrt(3.0) * sin(angle / 3.0)
    cosmological = (
        2.0 * length / sqrt(3.0) * sin((pi - angle) / 3.0)
    )
    unphysical = -(black_hole + cosmological)
    length_squared = length**2
    kappa_black_hole = (
        (cosmological - black_hole)
        * (black_hole - unphysical)
        / (2.0 * length_squared * black_hole)
    )
    kappa_cosmological = (
        (cosmological - black_hole)
        * (cosmological - unphysical)
        / (2.0 * length_squared * cosmological)
    )
    kappa_unphysical = (
        (black_hole - unphysical)
        * (cosmological - unphysical)
        / (2.0 * length_squared * (-unphysical))
    )
    return SdSHorizons(
        black_hole=black_hole,
        cosmological=cosmological,
        unphysical=unphysical,
        kappa_black_hole=kappa_black_hole,
        kappa_cosmological=kappa_cosmological,
        kappa_unphysical=kappa_unphysical,
    )


def areal_radius(rho: np.ndarray, parameters: SdSParameters) -> np.ndarray:
    r"""Map the professor's compact coordinate to areal radius.

    .. math::

       \rho=\frac{1-r_b/r}{1-r_b/r_c},\qquad
       r=\frac{r_b}{1-(1-r_b/r_c)\rho}.

    The endpoints are assigned explicitly so their floating-point values are
    the analytically known horizon radii.
    """

    horizons = sds_horizons(parameters)
    rho = np.asarray(rho, dtype=float)
    compactification = 1.0 - horizons.black_hole / horizons.cosmological
    radius = horizons.black_hole / (1.0 - compactification * rho)
    radius = np.where(rho == 0.0, horizons.black_hole, radius)
    return np.where(rho == 1.0, horizons.cosmological, radius)


def compact_radius(r: np.ndarray, parameters: SdSParameters) -> np.ndarray:
    """Map areal radius to the common compact coordinate ``rho``."""

    horizons = sds_horizons(parameters)
    r = np.asarray(r, dtype=float)
    denominator = 1.0 - horizons.black_hole / horizons.cosmological
    return (1.0 - horizons.black_hole / r) / denominator


def compactification_derivative(
    r: np.ndarray, parameters: SdSParameters
) -> np.ndarray:
    """Return the analytic derivative ``d rho / d r``."""

    horizons = sds_horizons(parameters)
    r = np.asarray(r, dtype=float)
    denominator = 1.0 - horizons.black_hole / horizons.cosmological
    return horizons.black_hole / (denominator * r**2)


def metric_f(r: np.ndarray, parameters: SdSParameters) -> np.ndarray:
    """The SdS lapse function f=1-2M/r-r^2/L^2."""

    r = np.asarray(r)
    return 1.0 - 2.0 * parameters.mass / r - (
        r / parameters.cosmological_length
    ) ** 2


def metric_f_prime(r: np.ndarray, parameters: SdSParameters) -> np.ndarray:
    """Radial derivative of the SdS lapse function."""

    r = np.asarray(r)
    return 2.0 * parameters.mass / r**2 - (
        2.0 * r / parameters.cosmological_length**2
    )


def tortoise_grid_speed(
    rho: np.ndarray, parameters: SdSParameters
) -> np.ndarray:
    """Return d rho / d r_* = f d rho / dr."""

    rho = np.asarray(rho, dtype=float)
    r = areal_radius(rho, parameters)
    speed = metric_f(r, parameters) * compactification_derivative(r, parameters)
    # The exact horizon values are zero.  Explicit assignment avoids retaining
    # roundoff from evaluating f at a numerically computed polynomial root.
    speed = np.where((rho == 0.0) | (rho == 1.0), 0.0, speed)
    return speed


def _singular_boost(
    r: np.ndarray,
    parameters: SdSParameters,
    bridge: str,
) -> np.ndarray:
    """Evaluate the minimum and minimal boosts in regularized closed form.

    The height-function derivatives contain individual horizon poles.  Here
    those poles are cancelled analytically against the factorized metric
    function before numerical evaluation.  No displaced horizon points or
    epsilon offsets are used.
    """

    horizons = sds_horizons(parameters)
    r = np.asarray(r, dtype=float)
    rb = horizons.black_hole
    rc = horizons.cosmological
    ru = horizons.unphysical
    length_squared = parameters.cosmological_length**2
    if bridge == "minimum":
        boost = (
            (rc - r)
            * (r - ru)
            / (2.0 * horizons.kappa_black_hole * length_squared * r)
            - (r - rb)
            * (r - ru)
            / (2.0 * horizons.kappa_cosmological * length_squared * r)
        )
    elif bridge == "minimal":
        boost = (
            rb
            * (rc - r)
            * (r - ru)
            / (2.0 * horizons.kappa_black_hole * length_squared * r**2)
            - rc
            * (r - rb)
            * (r - ru)
            / (2.0 * horizons.kappa_cosmological * length_squared * r**2)
            - ru
            * (r - rb)
            * (rc - r)
            / (2.0 * horizons.kappa_unphysical * length_squared * r**2)
        )
    else:
        raise ValueError(f"Unsupported singular bridge {bridge!r}.")
    boost = np.where(r == rb, 1.0, boost)
    return np.where(r == rc, -1.0, boost)


def _boost_radial_derivative(
    r: np.ndarray, parameters: SdSParameters, bridge: str
) -> np.ndarray:
    """Return the analytic radial derivative ``dB/dr``."""

    horizons = sds_horizons(parameters)
    r = np.asarray(r, dtype=float)
    rb = horizons.black_hole
    rc = horizons.cosmological
    ru = horizons.unphysical
    length_squared = parameters.cosmological_length**2

    if bridge in {"minimum", "minimal"}:
        numerator_b = (rc - r) * (r - ru)
        derivative_b = rc + ru - 2.0 * r
        numerator_c = (r - rb) * (r - ru)
        derivative_c = 2.0 * r - rb - ru
        if bridge == "minimum":
            term_b = (derivative_b / r - numerator_b / r**2) / (
                2.0 * horizons.kappa_black_hole * length_squared
            )
            term_c = (derivative_c / r - numerator_c / r**2) / (
                2.0 * horizons.kappa_cosmological * length_squared
            )
            return term_b - term_c

        numerator_u = (r - rb) * (rc - r)
        derivative_u = rc + rb - 2.0 * r
        quotient_b = derivative_b / r**2 - 2.0 * numerator_b / r**3
        quotient_c = derivative_c / r**2 - 2.0 * numerator_c / r**3
        quotient_u = derivative_u / r**2 - 2.0 * numerator_u / r**3
        return (
            rb
            * quotient_b
            / (2.0 * horizons.kappa_black_hole * length_squared)
            - rc
            * quotient_c
            / (2.0 * horizons.kappa_cosmological * length_squared)
            - ru
            * quotient_u
            / (2.0 * horizons.kappa_unphysical * length_squared)
        )

    if bridge == "linear":
        return np.full_like(r, -2.0 / horizons.width)
    if bridge == "modified_linear":
        factor = rb**2 / horizons.width
        return 2.0 * factor * (-1.0 / r**2 - 2.0 * (rc - r) / r**3)
    if bridge == "mavrogiannis":
        mass = parameters.mass
        denominator = sqrt(
            1.0 - 9.0 * mass**2 * parameters.cosmological_constant
        )
        first = 1.0 - 3.0 * mass / r
        root = np.sqrt(1.0 + 6.0 * mass / r)
        first_prime = 3.0 * mass / r**2
        root_prime = -3.0 * mass / (r**2 * root)
        return -(first_prime * root + first * root_prime) / denominator
    if bridge == "slow_roll":
        gamma = (rc**2 + rb**2) / (rc**3 - rb**3)
        beta = rc**2 * rb**2 * (rc + rb) / (rc**3 - rb**3)
        return -gamma - 2.0 * beta / r**3
    raise ValueError(f"Unknown bridge {bridge!r}; choose from {BRIDGE_CHOICES}.")


def propagation_endpoint_coefficients(
    parameters: SdSParameters, bridge: str
) -> tuple[float, float]:
    """Return exact l'Hopital limits of ``A`` at both horizons."""

    horizons = sds_horizons(parameters)
    radii = np.array([horizons.black_hole, horizons.cosmological])
    f_prime = metric_f_prime(radii, parameters)
    rho_prime = compactification_derivative(radii, parameters)
    boost_prime = _boost_radial_derivative(radii, parameters, bridge)
    signs = np.array([1.0, -1.0])
    limits = f_prime * rho_prime / (-2.0 * signs * boost_prime)
    return float(limits[0]), float(limits[1])


def bridge_boost(
    rho: np.ndarray, parameters: SdSParameters, bridge: str
) -> np.ndarray:
    """Return B=f dh/dr for a future-directed bridge foliation."""

    if bridge not in BRIDGE_CHOICES:
        raise ValueError(f"Unknown bridge {bridge!r}; choose from {BRIDGE_CHOICES}.")

    horizons = sds_horizons(parameters)
    r = areal_radius(rho, parameters)
    if bridge in {"minimum", "minimal"}:
        return _singular_boost(r, parameters, bridge)

    if bridge == "linear":
        return -1.0 + 2.0 * (horizons.cosmological - r) / horizons.width

    if bridge == "modified_linear":
        return -1.0 + 2.0 * (
            (horizons.cosmological - r)
            / horizons.width
            * horizons.black_hole**2
            / r**2
        )

    if bridge == "mavrogiannis":
        denominator = sqrt(
            1.0 - 9.0 * parameters.mass**2 * parameters.cosmological_constant
        )
        return -(
            (1.0 - 3.0 * parameters.mass / r)
            / denominator
            * np.sqrt(1.0 + 6.0 * parameters.mass / r)
        )

    gamma = (
        horizons.cosmological**2 + horizons.black_hole**2
    ) / (horizons.cosmological**3 - horizons.black_hole**3)
    beta = (
        horizons.cosmological**2
        * horizons.black_hole**2
        * (horizons.cosmological + horizons.black_hole)
        / (horizons.cosmological**3 - horizons.black_hole**3)
    )
    return -gamma * r + beta / r**2


def propagation_coefficient(
    rho: np.ndarray, parameters: SdSParameters, bridge: str
) -> np.ndarray:
    """Return A=(d rho/d r_*)/(1-B^2), finite inside the horizons."""

    rho = np.asarray(rho)
    speed = tortoise_grid_speed(rho, parameters)
    boost = bridge_boost(rho, parameters, bridge)
    denominator = (1.0 - boost) * (1.0 + boost)
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = speed / denominator
    left_value, right_value = propagation_endpoint_coefficients(parameters, bridge)
    coefficient = np.where(rho == 0.0, left_value, coefficient)
    coefficient = np.where(rho == 1.0, right_value, coefficient)
    return coefficient


def rescaled_scalar_potential(
    rho: np.ndarray, parameters: SdSParameters
) -> np.ndarray:
    """Return V/(d rho/d r_*) for the reduced scalar wave equation."""

    r = areal_radius(rho, parameters)
    ell = parameters.ell
    angular = ell * (ell + 1.0) / r**2
    curvature = metric_f_prime(r, parameters) / r
    return (angular + curvature) / compactification_derivative(r, parameters)


def characteristic_speeds(
    rho: np.ndarray, parameters: SdSParameters, bridge: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return ingoing and outgoing radial light speeds d rho / d tau."""

    coefficient = propagation_coefficient(rho, parameters, bridge)
    boost = bridge_boost(rho, parameters, bridge)
    ingoing = -coefficient * (1.0 + boost)
    outgoing = coefficient * (1.0 - boost)
    return ingoing, outgoing


def scalar_gaussian_initial_data(
    rho: np.ndarray,
    parameters: SdSParameters,
    initial: ScalarInitialData,
    bridge: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, psi, pi) with psi=d_rho u."""

    rho = np.asarray(rho)
    displacement = rho - initial.center_fraction
    u = np.exp(-(displacement**2) / (2.0 * initial.width**2))
    psi = -displacement * u / initial.width**2
    if initial.time_symmetric:
        pi = -bridge_boost(rho, parameters, bridge) * psi
    else:
        pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi


def scalar_areal_bump_initial_data(
    rho: np.ndarray,
    parameters: SdSParameters,
    initial: ArealBumpInitialData,
    bridge: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return identical physical data with ``psi=(du/dr)(dr/d rho)``."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    horizons = sds_horizons(parameters)
    support_left = initial.center_radius - initial.support_half_width
    support_right = initial.center_radius + initial.support_half_width
    if not horizons.black_hole < support_left < support_right < horizons.cosmological:
        raise ValueError("The compact areal-radius pulse must lie between horizons.")
    u, du_dr = compact_areal_profile(radius, initial)
    dr_drho = 1.0 / compactification_derivative(radius, parameters)
    psi = du_dr * dr_drho
    if initial.time_symmetric:
        pi = -bridge_boost(rho, parameters, bridge) * psi
    else:
        pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi


def scalar_areal_velocity_initial_data(
    rho: np.ndarray,
    parameters: SdSParameters,
    initial: ArealVelocityBumpInitialData,
    bridge: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``u=psi=0`` and ``pi=G(r)/A_L(r)`` for SdS."""

    rho = np.asarray(rho, dtype=float)
    radius = areal_radius(rho, parameters)
    horizons = sds_horizons(parameters)
    support_left = initial.center_radius - initial.support_half_width
    support_right = initial.center_radius + initial.support_half_width
    if support_left <= horizons.black_hole or support_right >= horizons.cosmological:
        raise ValueError(
            "The physical velocity bump must lie strictly between the SdS horizons."
        )

    velocity = compact_areal_velocity_profile(radius, initial)
    coefficient_a = propagation_coefficient(rho, parameters, bridge)
    u = np.zeros_like(rho)
    psi = np.zeros_like(rho)
    pi = np.zeros_like(rho)
    support = velocity != 0.0
    pi[support] = velocity[support] / coefficient_a[support]
    if not np.all(np.isfinite(pi)):
        raise FloatingPointError("Non-finite momentum in physical velocity data.")
    return u, psi, pi


def minimal_height(
    r: np.ndarray, parameters: SdSParameters, reference_radius: float
) -> np.ndarray:
    """Minimal-gauge height with ``h(reference_radius)=0``.

    Only the additive constant is fixed here; the logarithmic divergences at
    the two null horizons are the expected behavior of a stationary bridge
    height function.
    """

    horizons = sds_horizons(parameters)
    if not horizons.black_hole < reference_radius < horizons.cosmological:
        raise ValueError("The height reference radius must lie between horizons.")

    def primitive(radius: np.ndarray) -> np.ndarray:
        rb = horizons.black_hole
        rc = horizons.cosmological
        ru = horizons.unphysical
        radius = np.asarray(radius, dtype=float)
        return (
            np.log((radius - rb) / radius) / (2.0 * horizons.kappa_black_hole)
            + np.log((rc - radius) / radius)
            / (2.0 * horizons.kappa_cosmological)
            + np.log(radius / (radius - ru))
            / (2.0 * horizons.kappa_unphysical)
        )

    return primitive(np.asarray(r, dtype=float)) - primitive(
        np.asarray(reference_radius, dtype=float)
    )


def tortoise_coordinate(
    r: np.ndarray, parameters: SdSParameters, reference_radius: float
) -> np.ndarray:
    r"""SdS tortoise coordinate normalized by ``r_*(reference_radius)=0``.

    The three logarithmic terms are the exact partial-fraction integral of
    ``dr_*/dr=1/f``.  Ratios to the reference-radius factors make every
    logarithm dimensionless and fix the additive constant geometrically.
    """

    horizons = sds_horizons(parameters)
    rb, rc, ru = (
        horizons.black_hole,
        horizons.cosmological,
        horizons.unphysical,
    )
    if not rb < reference_radius < rc:
        raise ValueError("The tortoise reference radius must lie between horizons.")
    radius = np.asarray(r, dtype=float)
    if np.any((radius <= rb) | (radius >= rc)):
        raise ValueError("The SdS tortoise coordinate is defined between horizons.")
    return (
        np.log((radius - rb) / (reference_radius - rb))
        / (2.0 * horizons.kappa_black_hole)
        - np.log((rc - radius) / (rc - reference_radius))
        / (2.0 * horizons.kappa_cosmological)
        + np.log((radius - ru) / (reference_radius - ru))
        / (2.0 * horizons.kappa_unphysical)
    )


def retarded_time_offset(
    parameters: SdSParameters, reference_radius: float
) -> float:
    r"""Return the analytic limit ``q_L=lim_{r->r_c}(h+r_*)``.

    The cosmological-horizon logarithms in the normalized minimal height and
    tortoise coordinate cancel algebraically.  Evaluating this expression
    therefore uses neither an endpoint offset nor a fitted time translation.
    """

    horizons = sds_horizons(parameters)
    rb, rc = horizons.black_hole, horizons.cosmological
    ru = horizons.unphysical
    r0 = float(reference_radius)
    if not rb < r0 < rc:
        raise ValueError("The retarded-time reference radius must lie between horizons.")
    black_hole_term = np.log((rc - rb) ** 2 * r0 / (rc * (r0 - rb) ** 2))
    scale_log = np.log(rc / r0)
    return float(
        black_hole_term / (2.0 * horizons.kappa_black_hole)
        + (1.0 / (2.0 * horizons.kappa_unphysical)
           - 1.0 / (2.0 * horizons.kappa_cosmological))
        * scale_log
    )


def turning_radius(parameters: SdSParameters, bridge: str) -> float:
    """Locate the bridge turning point B=0 by dense sampling."""

    rho = np.linspace(0.0, 1.0, 20_001)
    boost = bridge_boost(rho, parameters, bridge)
    index = int(np.argmin(np.abs(boost)))
    return float(areal_radius(rho[index], parameters))
