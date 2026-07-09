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
    """Localized Gaussian data for the reduced scalar field u=r Phi."""

    center_fraction: float = 0.45
    width: float = 0.35
    time_symmetric: bool = True
    pi_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.center_fraction < 1.0:
            raise ValueError("Gaussian center fraction must lie inside (0, 1).")
        if self.width <= 0:
            raise ValueError("Gaussian width must be positive.")

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


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
    """Map rho in [0, 1] to the areal radius between the two horizons."""

    horizons = sds_horizons(parameters)
    rho = np.asarray(rho)
    return horizons.black_hole + horizons.width * rho


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

    horizons = sds_horizons(parameters)
    r = areal_radius(rho, parameters)
    return metric_f(r, parameters) / horizons.width


def _singular_boost(
    r: np.ndarray,
    parameters: SdSParameters,
    bridge: str,
) -> np.ndarray:
    """Evaluate boost functions whose derivative form is singular at horizons."""

    horizons = sds_horizons(parameters)
    r = np.asarray(r, dtype=float)
    boost = np.empty_like(r)
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, horizons.width)
    left = np.abs(r - horizons.black_hole) <= tolerance
    right = np.abs(r - horizons.cosmological) <= tolerance
    interior = ~(left | right)
    boost[left] = 1.0
    boost[right] = -1.0

    ri = r[interior]
    fi = metric_f(ri, parameters)
    if bridge == "minimum":
        boost[interior] = fi * (
            1.0
            / (
                2.0
                * horizons.kappa_black_hole
                * (ri - horizons.black_hole)
            )
            - 1.0
            / (
                2.0
                * horizons.kappa_cosmological
                * (horizons.cosmological - ri)
            )
        )
        return boost

    if bridge == "minimal":
        boost[interior] = fi * (
            horizons.black_hole
            / (
                2.0
                * horizons.kappa_black_hole
                * ri
                * (ri - horizons.black_hole)
            )
            - horizons.cosmological
            / (
                2.0
                * horizons.kappa_cosmological
                * ri
                * (horizons.cosmological - ri)
            )
            - horizons.unphysical
            / (
                2.0
                * horizons.kappa_unphysical
                * ri
                * (ri - horizons.unphysical)
            )
        )
        return boost

    raise ValueError(f"Unsupported singular bridge {bridge!r}.")


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
    denominator = 1.0 - boost**2
    coefficient = np.empty_like(speed, dtype=float)
    mask = np.abs(denominator) > 1e-13
    coefficient[mask] = speed[mask] / denominator[mask]

    if np.any(~mask):
        offset = 1e-7
        left_value = _coefficient_at_fraction(offset, parameters, bridge)
        right_value = _coefficient_at_fraction(1.0 - offset, parameters, bridge)
        left = rho <= 0.5
        coefficient[~mask & left] = left_value
        coefficient[~mask & ~left] = right_value
    return coefficient


def _coefficient_at_fraction(
    rho: float, parameters: SdSParameters, bridge: str
) -> float:
    sample = np.array([rho], dtype=float)
    speed = tortoise_grid_speed(sample, parameters)
    boost = bridge_boost(sample, parameters, bridge)
    return float(speed[0] / (1.0 - boost[0] ** 2))


def rescaled_scalar_potential(
    rho: np.ndarray, parameters: SdSParameters
) -> np.ndarray:
    """Return V/(d rho/d r_*) for the reduced scalar wave equation."""

    horizons = sds_horizons(parameters)
    r = areal_radius(rho, parameters)
    ell = parameters.ell
    angular = ell * (ell + 1.0) / r**2
    curvature = metric_f_prime(r, parameters) / r
    return horizons.width * (angular + curvature)


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

    horizons = sds_horizons(parameters)
    rho = np.asarray(rho)
    r = areal_radius(rho, parameters)
    center = horizons.black_hole + initial.center_fraction * horizons.width
    displacement = r - center
    u = np.exp(-(displacement**2) / (2.0 * initial.width**2))
    psi = -horizons.width * displacement * u / initial.width**2
    if initial.time_symmetric:
        pi = -bridge_boost(rho, parameters, bridge) * psi
    else:
        pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi


def turning_radius(parameters: SdSParameters, bridge: str) -> float:
    """Locate the bridge turning point B=0 by dense sampling."""

    rho = np.linspace(0.0, 1.0, 20_001)
    boost = bridge_boost(rho, parameters, bridge)
    index = int(np.argmin(np.abs(boost)))
    return float(areal_radius(rho[index], parameters))
