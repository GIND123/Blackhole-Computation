r"""An independent static-coordinate reference for the sourced mode equation.

The hyperboloidal evolution of :mod:`black_hole.source_evolution` carries
three separate ingredients that all have to be right at once: the reduction
of ``Box Phi = S`` to a mode source ``f r S_{lm}``, the extra term
``-(r/G) S_{lm}`` in the first-order system, and the evaluation of the
emitter at ``t = tau - h_L(r)``.  A discretization check inside that scheme
cannot separate them.

This module therefore solves the very same physical problem by a route that
shares none of that machinery.  It integrates

.. math::

   \partial_t^2 u_{\ell m} = \partial_{r_*}^2 u_{\ell m} - V_\ell u_{\ell m}
        - f\,r\,S_{\ell m}(t, r)

directly in *static* Schwarzschild coordinates ``(t, r_*)``, on a uniform
tortoise grid, with a second-order leapfrog stepper and one-way outflow
conditions at both truncation boundaries.  There is no height function, no
compactification, no hyperboloidal boost, and no first-order reduction.

At a fixed areal radius the two clocks differ only by the constant
``h_L(r_{\rm obs})``, so the two waveforms may be compared directly once the
hyperboloidal signal is shifted by that constant.  Agreement therefore
validates the source normalization and the Killing-time evaluation of the
emitter, not merely the radial discretization.

The comparison is only meaningful before the boundary truncation can
influence the observer.  :func:`reflection_free_time` returns that time, and
the study restricts every reported norm to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .localized_source import (
    LocalizedSourceParameters,
    radial_profile,
    time_profile,
)

__all__ = [
    "StaticReferenceGrid",
    "reflection_free_time",
    "solve_static_mode",
]


def _tortoise(radius: np.ndarray, mass: float) -> np.ndarray:
    return radius + 2.0 * mass * np.log(radius / (2.0 * mass) - 1.0)


def _invert_tortoise(
    tortoise: np.ndarray, mass: float, iterations: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    r"""Invert ``r_*(r)`` and return both ``r`` and the lapse ``f``.

    Writing ``r = 2M(1+e^\lambda)`` with ``\lambda = \ln[(r-2M)/2M]`` turns
    the relation into

    .. math::

       r_* = 2M\left(1 + e^\lambda + \lambda\right),

    whose right-hand side is convex and strictly increasing in ``lambda``,
    so Newton's method converges from any seed.  Solving for ``lambda``
    rather than for ``r`` keeps the near-horizon region accurate long after
    ``r-2M`` has become too small to resolve as a difference of doubles, and
    it yields the lapse ``f = e^\lambda/(1+e^\lambda)`` directly instead of
    through the catastrophic cancellation in ``1-2M/r``.
    """

    tortoise = np.asarray(tortoise, dtype=float)
    horizon = 2.0 * mass
    far = tortoise > 4.0 * horizon
    lam = np.where(
        far,
        np.log(np.maximum(tortoise, 3.0 * mass) / horizon),
        (tortoise - horizon) / horizon,
    )
    for _ in range(iterations):
        exponential = np.exp(np.clip(lam, -700.0, 700.0))
        step = (horizon * (1.0 + exponential + lam) - tortoise) / (
            horizon * (exponential + 1.0)
        )
        lam = lam - step
        if np.max(np.abs(step)) < 1e-15:
            break
    exponential = np.exp(np.clip(lam, -700.0, 700.0))
    return horizon * (1.0 + exponential), exponential / (1.0 + exponential)


@dataclass(frozen=True)
class StaticReferenceGrid:
    """Uniform tortoise grid for the static-coordinate reference solve."""

    tortoise_min: float = -160.0
    tortoise_max: float = 260.0
    points: int = 8401
    courant: float = 0.5
    mass: float = 1.0

    def __post_init__(self) -> None:
        if self.tortoise_max <= self.tortoise_min:
            raise ValueError("The tortoise grid must have positive extent.")
        if self.points < 128:
            raise ValueError("The static reference grid is too coarse.")
        if not 0.0 < self.courant <= 0.9:
            raise ValueError("The leapfrog Courant number must lie in (0, 0.9].")

    @property
    def spacing(self) -> float:
        return (self.tortoise_max - self.tortoise_min) / (self.points - 1)

    @property
    def timestep(self) -> float:
        return self.courant * self.spacing

    def coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the tortoise grid with its areal radius and lapse."""

        tortoise = np.linspace(self.tortoise_min, self.tortoise_max, self.points)
        radius, lapse = _invert_tortoise(tortoise, self.mass)
        return tortoise, radius, lapse


def reflection_free_time(
    grid: StaticReferenceGrid,
    source: LocalizedSourceParameters,
    observer_radius: float,
) -> float:
    """Return the latest time unaffected by the truncation boundaries.

    A signal must leave the emitter, reach a boundary, and return to the
    observer.  Both boundaries are checked and the earlier arrival wins.
    """

    emitter = _tortoise(np.asarray(source.radial_support), grid.mass)
    observer = float(_tortoise(np.asarray(observer_radius), grid.mass))
    emission = source.killing_time_support[0]
    left = emission + (float(np.max(emitter)) - grid.tortoise_min) + (
        observer - grid.tortoise_min
    )
    right = emission + (grid.tortoise_max - float(np.min(emitter))) + (
        grid.tortoise_max - observer
    )
    return float(min(left, right))


def _observer_stencils(
    tortoise: np.ndarray,
    observer_radii: tuple[float, ...],
    mass: float,
    width: int = 4,
) -> tuple[list[slice], list[np.ndarray]]:
    """Return local Lagrange stencils evaluating the field at exact radii."""

    windows: list[slice] = []
    weights: list[np.ndarray] = []
    spacing = float(tortoise[1] - tortoise[0])
    for value in observer_radii:
        target = float(_tortoise(np.asarray(float(value)), mass))
        if not tortoise[0] < target < tortoise[-1]:
            raise ValueError(f"Observer r={value} lies outside the static grid.")
        floating = (target - tortoise[0]) / spacing
        start = int(np.clip(round(floating) - width // 2, 0, tortoise.size - width))
        nodes = tortoise[start : start + width]
        weight = np.ones(width)
        for row in range(width):
            for column in range(width):
                if row != column:
                    weight[row] *= (target - nodes[column]) / (
                        nodes[row] - nodes[column]
                    )
        windows.append(slice(start, start + width))
        weights.append(weight)
    return windows, weights


def solve_static_mode(
    *,
    ell: int,
    mode_amplitude: float,
    source: LocalizedSourceParameters,
    observer_radii: tuple[float, ...],
    end_time: float,
    grid: StaticReferenceGrid | None = None,
    sample_dt: float = 0.05,
) -> dict[str, np.ndarray]:
    r"""Integrate one sourced mode in static Schwarzschild coordinates.

    ``mode_amplitude`` is the constant ``g_\ell Y^{\rm R}_{\ell m}`` of the
    angular decomposition, so the returned waveform is directly the mode
    coefficient ``u_{\ell m}`` of the three-dimensional problem.
    """

    grid = grid or StaticReferenceGrid()
    if ell < 0:
        raise ValueError("The harmonic index must be nonnegative.")
    tortoise, radius, lapse = grid.coordinates()
    mass = grid.mass
    potential = lapse * (
        ell * (ell + 1.0) / radius**2 + 2.0 * mass / radius**3
    )
    kernel = (
        -source.amplitude
        * mode_amplitude
        * lapse
        * radius
        * radial_profile(radius, source)
    )
    spacing = grid.spacing
    step = grid.timestep

    def spatial(values: np.ndarray) -> np.ndarray:
        second = np.zeros_like(values)
        second[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / spacing**2
        return second - potential * values

    def forcing(current_time: float) -> np.ndarray:
        return kernel * time_profile(current_time, source)

    previous = np.zeros_like(radius)
    current = 0.5 * step**2 * (spatial(previous) + forcing(0.0))
    # Observers are placed at their exact areal radius by a local cubic
    # Lagrange interpolation, so the comparison is not limited by where the
    # requested radius happens to fall between grid points.
    observer_slice, observer_weights = _observer_stencils(
        tortoise, observer_radii, mass
    )

    def sample() -> np.ndarray:
        return np.asarray(
            [
                float(current[window] @ weight)
                for window, weight in zip(observer_slice, observer_weights)
            ]
        )

    sample_stride = max(1, int(round(sample_dt / step)))
    steps = int(np.ceil(end_time / step))
    times = [0.0, step]
    samples = [np.zeros(len(observer_radii)), sample()]

    courant = step / spacing
    for number in range(2, steps + 1):
        following = (
            2.0 * current
            - previous
            + step**2 * (spatial(current) + forcing((number - 1) * step))
        )
        # One-way outflow at both truncations: u(x_b, t+dt) advected inward.
        following[0] = current[0] + courant * (current[1] - current[0])
        following[-1] = current[-1] - courant * (current[-1] - current[-2])
        previous, current = current, following
        if number % sample_stride == 0 or number == steps:
            times.append(number * step)
            samples.append(sample())
    return {
        "times": np.asarray(times),
        "observer_radii": np.asarray(observer_radii, dtype=float),
        "signals": np.asarray(samples),
        "spacing": np.asarray(spacing),
        "timestep": np.asarray(step),
    }
