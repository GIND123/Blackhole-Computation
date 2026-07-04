"""Analytic coefficients and initial data for the Regge--Wheeler system."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    """Physical parameters for an axial Schwarzschild perturbation."""

    mass: float = 1.0
    ell: int = 2

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("Black-hole mass must be positive.")
        if self.ell < 2:
            raise ValueError("Gravitational perturbations require ell >= 2.")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class InitialData:
    """Localized Gaussian data used in the professor's reference notebook."""

    center: float = 0.5
    width: float = 0.04
    pi_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.center < 1:
            raise ValueError("Gaussian center must lie inside (0, 1).")
        if self.width <= 0:
            raise ValueError("Gaussian width must be positive.")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def boost(rho: np.ndarray) -> np.ndarray:
    """Minimal-gauge boost H_*(rho)."""

    rho = np.asarray(rho)
    return -1.0 + 2.0 * (1.0 - rho) ** 2


def propagation_coefficient(
    rho: np.ndarray, parameters: ModelParameters
) -> np.ndarray:
    """Regular coefficient fG/(1-H_*^2)."""

    rho = np.asarray(rho)
    return 1.0 / (8.0 * parameters.mass * (2.0 - rho))


def rescaled_potential(
    rho: np.ndarray, parameters: ModelParameters
) -> np.ndarray:
    """Positive rescaled Regge--Wheeler potential V_l/(fG)."""

    rho = np.asarray(rho)
    ell = parameters.ell
    return (ell * (ell + 1) - 3.0 * (1.0 - rho)) / (
        2.0 * parameters.mass
    )


def gaussian_initial_data(
    rho: np.ndarray, initial: InitialData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, psi, pi), with psi exactly equal to d_rho u."""

    rho = np.asarray(rho)
    displacement = rho - initial.center
    u = np.exp(-(displacement**2) / (2.0 * initial.width**2))
    psi = -displacement * u / initial.width**2
    pi = np.full_like(u, initial.pi_amplitude)
    return u, psi, pi
