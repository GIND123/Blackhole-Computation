"""Analytic minimal-gauge scalar-wave model on Schwarzschild spacetime."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .sds_model import ScalarInitialData


@dataclass(frozen=True)
class SchwarzschildScalarParameters:
    """Physical parameters for a scalar spherical-harmonic mode."""

    mass: float = 1.0
    ell: int = 2

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("Black-hole mass must be positive.")
        if self.ell < 0:
            raise ValueError("Scalar spherical-harmonic index ell must be >= 0.")

    @property
    def black_hole_horizon(self) -> float:
        return 2.0 * self.mass

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def areal_radius(
    rho: np.ndarray, parameters: SchwarzschildScalarParameters
) -> np.ndarray:
    r"""Return ``r=2M/(1-rho)``, with ``rho=1`` at future null infinity."""

    rho = np.asarray(rho, dtype=float)
    denominator = 1.0 - rho
    return np.divide(
        2.0 * parameters.mass,
        denominator,
        out=np.full_like(rho, np.inf, dtype=float),
        where=denominator != 0.0,
    )


def minimal_boost(rho: np.ndarray) -> np.ndarray:
    """Return the Schwarzschild minimal-gauge boost ``B=f dh/dr``."""

    rho = np.asarray(rho, dtype=float)
    return -1.0 + 2.0 * (1.0 - rho) ** 2


def propagation_coefficient(
    rho: np.ndarray, parameters: SchwarzschildScalarParameters
) -> np.ndarray:
    """Return the fully regular coefficient ``A=p/(1-B^2)``."""

    rho = np.asarray(rho, dtype=float)
    return 1.0 / (8.0 * parameters.mass * (2.0 - rho))


def rescaled_scalar_potential(
    rho: np.ndarray, parameters: SchwarzschildScalarParameters
) -> np.ndarray:
    r"""Return the regular scalar potential ``V/(f d rho/dr)``.

    For ``rho=1-2M/r`` this reduces analytically to

    .. math::

       P=\frac{\ell(\ell+1)+(1-\rho)}{2M}.
    """

    rho = np.asarray(rho, dtype=float)
    return (
        parameters.ell * (parameters.ell + 1.0) + (1.0 - rho)
    ) / (2.0 * parameters.mass)


def scalar_gaussian_initial_data(
    rho: np.ndarray,
    parameters: SchwarzschildScalarParameters,
    initial: ScalarInitialData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the same ``rho``-Gaussian used by every finite-L evolution."""

    del parameters
    rho = np.asarray(rho, dtype=float)
    displacement = rho - initial.center_fraction
    u = np.exp(-(displacement**2) / (2.0 * initial.width**2))
    psi = -displacement * u / initial.width**2
    if initial.time_symmetric:
        pi = -minimal_boost(rho) * psi
    else:
        pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi


def minimal_height(
    r: np.ndarray,
    parameters: SchwarzschildScalarParameters,
    reference_radius: float,
) -> np.ndarray:
    """Minimal-gauge height normalized by ``h(reference_radius)=0``."""

    mass = parameters.mass
    if reference_radius <= 2.0 * mass:
        raise ValueError("The height reference radius must satisfy r > 2M.")

    def primitive(radius: np.ndarray) -> np.ndarray:
        radius = np.asarray(radius, dtype=float)
        if np.any(radius <= 2.0 * mass):
            raise ValueError("The Schwarzschild height is defined for r > 2M.")
        return (
            -radius
            - 4.0 * mass * np.log(radius / mass)
            + 2.0 * mass * np.log((radius - 2.0 * mass) / mass)
        )

    return primitive(np.asarray(r, dtype=float)) - primitive(
        np.asarray(reference_radius, dtype=float)
    )


def characteristic_speeds(
    rho: np.ndarray, parameters: SchwarzschildScalarParameters
) -> tuple[np.ndarray, np.ndarray]:
    """Return ingoing and outgoing coordinate light speeds."""

    coefficient = propagation_coefficient(rho, parameters)
    boost = minimal_boost(rho)
    return -coefficient * (1.0 + boost), coefficient * (1.0 - boost)
