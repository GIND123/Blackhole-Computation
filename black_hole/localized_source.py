r"""A genuinely three-dimensional localized source for the scalar wave equation.

The pure-mode validation stage solved the homogeneous problem
``Box Phi = 0`` for one spherical harmonic at a time.  The next stage models
the retarded Green function of the scalar wave operator by solving

.. math::

   \Box\Phi = S,\qquad \Phi(\tau=0,\cdot)=\partial_\tau\Phi(\tau=0,\cdot)=0,

with a smooth source ``S`` localized near ``r=6M`` in the equatorial plane.
A source that is localized in *all three* spatial directions excites mixed
angular modes and, in the limit of vanishing support, reproduces the
retarded Green function of a point emitter.  On Schwarzschild this produces
the direct signal followed by the caustic echoes of Zenginoglu and Galley,
Phys. Rev. D 86, 064030 (2012), arXiv:1206.1109.

Source specification
--------------------

The source is defined once and for all in the *background-independent*
labels ``(t, r, theta, phi)``, where ``t`` is Killing time and ``r`` is
areal radius:

.. math::

   S(t,r,\theta,\varphi) = \mathcal{A}\,T(t)\,R(r)\,\hat\Omega(\gamma),
   \qquad \cos\gamma = \hat n\cdot\hat n_{\rm s}.

Every factor is infinitely differentiable, and ``T`` and ``R`` have compact
support.  Because the Schwarzschild and Schwarzschild--de Sitter evolutions
in this project use the bridge time ``tau = t + h_L(r)``, the source is
evaluated in the code at ``t = tau - h_L(r)``.  The *same* function of
``(t, r, theta, phi)`` therefore acts on every background, which is what
makes the cosmological-length sequence a controlled flat-limit experiment.

Angular decomposition
---------------------

The angular factor is the von Mises--Fisher profile

.. math::

   \Omega(\gamma) = \exp\left[-\frac{1-\cos\gamma}{\sigma^2}\right]
                  = e^{-\kappa}e^{\kappa\cos\gamma},\qquad \kappa=\sigma^{-2},

whose Legendre expansion is known in closed form through the modified
spherical Bessel functions ``i_l``,

.. math::

   e^{\kappa x} = \sum_{\ell}(2\ell+1)\,i_\ell(\kappa)\,P_\ell(x).

Normalizing the profile to unit angular integral and applying the addition
theorem for real orthonormal harmonics gives the exact modal amplitudes

.. math::

   S_{\ell m}(t,r) = \mathcal{A}\,T(t)\,R(r)\,g_\ell\,
                     Y^{\rm R}_{\ell m}(\theta_{\rm s},\varphi_{\rm s}),
   \qquad g_\ell = \frac{i_\ell(\kappa)}{i_0(\kappa)} .

The weights ``g_l`` fall off like ``exp[-l(l+1)/(2 kappa)]``, so ``kappa``
directly controls both the physical width of the emitter and the angular
bandwidth that the evolution must carry.  As ``kappa -> infinity`` the
weights tend to unity and the source becomes a point emitter, recovering the
delta-function decomposition of the retarded Green function.

Selection rule
--------------

For an emitter in the equatorial plane at ``phi_s = 0`` the configuration is
invariant under the reflections ``theta -> pi - theta`` and
``phi -> -phi``.  The amplitudes therefore vanish identically unless
``m >= 0`` and ``l + m`` is even.  Since the field equations do not couple
angular modes on a spherically symmetric background and the initial data
vanish, only that subset ever becomes nonzero.  The evolution carries
exactly those modes, and the selection rule doubles as a check on the
angular bookkeeping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import pi

import numpy as np
from scipy import special
from scipy.integrate import quad

from .three_d_solver import real_spherical_harmonic

__all__ = [
    "LocalizedSourceParameters",
    "SourceModeCatalogue",
    "angular_spectral_weights",
    "angular_profile",
    "build_mode_catalogue",
    "compact_bump",
    "minimum_ell_max",
    "radial_profile",
    "retained_angular_fraction",
    "source_normalizations",
    "time_profile",
    "verify_angular_expansion",
    "weak_source_integral",
]


def compact_bump(x: np.ndarray) -> np.ndarray:
    r"""Return the unit-amplitude ``C^\infty`` bump ``exp[1-1/(1-x^2)]``.

    The bump is identically zero for ``|x| >= 1`` and equals one at
    ``x = 0``.  It is the same profile already used by the initial-data
    families of this project, so source and initial-data studies share one
    smoothness convention.
    """

    x = np.asarray(x, dtype=float)
    values = np.zeros_like(x)
    inside = np.abs(x) < 1.0
    interior = x[inside]
    values[inside] = np.exp(1.0 - 1.0 / (1.0 - interior**2))
    return values


@dataclass(frozen=True)
class LocalizedSourceParameters:
    r"""Parameters of the smooth localized emitter.

    Attributes
    ----------
    amplitude:
        Overall scale ``\mathcal{A}``.  The problem is linear, so this only
        fixes the units of the reported waveforms.
    center_radius, radial_half_width:
        Centre and compact-support half-width of ``R(r)`` in areal radius.
    time_center, time_half_width:
        Centre and compact-support half-width of ``T(t)`` in Killing time.
    angular_concentration:
        ``kappa = 1/sigma^2`` of the angular profile.  Larger values give a
        narrower emitter and a wider angular bandwidth.
    source_theta, source_phi:
        Angular position of the emitter.  The default places it in the
        equatorial plane, matching the visualizations of the earlier
        Schwarzschild and Kerr caustic studies.
    """

    amplitude: float = 1.0
    center_radius: float = 6.0
    radial_half_width: float = 1.5
    time_center: float = 30.0
    time_half_width: float = 4.0
    angular_concentration: float = 16.0
    source_theta: float = pi / 2.0
    source_phi: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude) or self.amplitude == 0.0:
            raise ValueError("The source amplitude must be finite and nonzero.")
        if self.radial_half_width <= 0.0 or self.time_half_width <= 0.0:
            raise ValueError("Source support half-widths must be positive.")
        if self.center_radius <= self.radial_half_width:
            raise ValueError("The radial source support must have positive radius.")
        if self.angular_concentration <= 0.0:
            raise ValueError("The angular concentration kappa must be positive.")
        if not 0.0 <= self.source_theta <= pi:
            raise ValueError("The source colatitude must lie in [0, pi].")

    @property
    def angular_width(self) -> float:
        """Return ``sigma = kappa^{-1/2}`` in radians."""

        return float(self.angular_concentration**-0.5)

    @property
    def radial_support(self) -> tuple[float, float]:
        return (
            self.center_radius - self.radial_half_width,
            self.center_radius + self.radial_half_width,
        )

    @property
    def killing_time_support(self) -> tuple[float, float]:
        return (
            self.time_center - self.time_half_width,
            self.time_center + self.time_half_width,
        )

    def as_dict(self) -> dict:
        temporal_normalization, radial_normalization = source_normalizations(self)
        return {
            "profile": "covariantly normalized C-infinity compact delta family",
            "angular_profile": "normalized von Mises-Fisher exp[-(1-cos gamma)/sigma^2]",
            "equation": "Box Phi = S with vanishing initial data",
            "angular_width_radians": self.angular_width,
            "radial_support": list(self.radial_support),
            "killing_time_support": list(self.killing_time_support),
            "temporal_normalization": temporal_normalization,
            "radial_normalization": radial_normalization,
            "covariant_integral": self.amplitude,
            **asdict(self),
        }


@lru_cache(maxsize=None)
def _unit_bump_integral() -> float:
    return float(quad(lambda x: float(compact_bump(np.asarray(x))), -1.0, 1.0)[0])


@lru_cache(maxsize=None)
def _radial_measure_integral(center: float, half_width: float) -> float:
    left = center - half_width
    right = center + half_width
    return float(
        quad(
            lambda radius: float(
                compact_bump(np.asarray((radius - center) / half_width))
            )
            * radius**2,
            left,
            right,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
    )


def source_normalizations(source: LocalizedSourceParameters) -> tuple[float, float]:
    r"""Return constants that give unit temporal and radial integrals.

    The returned factors enforce ``integral T dt = 1`` and
    ``integral R r^2 dr = 1``.  Together with the normalized angular profile,
    this gives ``integral sqrt(-g) S d4x = amplitude``.
    """

    temporal = 1.0 / (source.time_half_width * _unit_bump_integral())
    radial = 1.0 / _radial_measure_integral(
        source.center_radius, source.radial_half_width
    )
    return temporal, radial


def time_profile(
    killing_time: np.ndarray, source: LocalizedSourceParameters
) -> np.ndarray:
    """Return ``T(t)``, the compact temporal factor in Killing time."""

    killing_time = np.asarray(killing_time, dtype=float)
    temporal_normalization, _ = source_normalizations(source)
    return temporal_normalization * compact_bump(
        (killing_time - source.time_center) / source.time_half_width
    )


def radial_profile(
    radius: np.ndarray, source: LocalizedSourceParameters
) -> np.ndarray:
    """Return ``R(r)``, the compact radial factor in areal radius."""

    radius = np.asarray(radius, dtype=float)
    _, radial_normalization = source_normalizations(source)
    return radial_normalization * compact_bump(
        (radius - source.center_radius) / source.radial_half_width
    )


def angular_profile(
    cosine_gamma: np.ndarray, source: LocalizedSourceParameters
) -> np.ndarray:
    r"""Return the angular factor normalized to unit integral over the sphere.

    The normalization ``\int\hat\Omega\,d\Omega = 1`` makes the emitter a
    smoothed ``delta^2(\hat n-\hat n_{\rm s})`` with an amplitude that is
    independent of the angular width.
    """

    cosine_gamma = np.asarray(cosine_gamma, dtype=float)
    kappa = source.angular_concentration
    # exp[kappa (x-1)] is bounded by one, so no overflow protection is needed.
    unnormalized = np.exp(kappa * (cosine_gamma - 1.0))
    # int exp[kappa(x-1)] dOmega = 2 pi (1 - e^{-2 kappa}) / kappa.
    normalization = 2.0 * pi * (-np.expm1(-2.0 * kappa)) / kappa
    return unnormalized / normalization


def angular_spectral_weights(kappa: float, ell_max: int) -> np.ndarray:
    r"""Return ``g_\ell = i_\ell(\kappa)/i_0(\kappa)`` for ``0 <= l <= l_max``.

    The ratio is evaluated through the exponentially scaled modified Bessel
    function ``ive``, so it is computed without intermediate overflow or
    underflow for any concentration used in practice.
    """

    if kappa <= 0.0:
        raise ValueError("The angular concentration kappa must be positive.")
    if ell_max < 0:
        raise ValueError("ell_max must be nonnegative.")
    orders = np.arange(ell_max + 1, dtype=float) + 0.5
    scaled = special.ive(orders, float(kappa))
    return np.asarray(scaled / scaled[0], dtype=float)


def retained_angular_fraction(kappa: float, ell_max: int) -> float:
    r"""Return the fraction of angular power kept by an ``l_max`` truncation.

    The exact angular power of the normalized profile is

    .. math::

       \int\hat\Omega^2\,d\Omega
         = \sum_\ell \frac{2\ell+1}{4\pi}\,g_\ell^2 \Big/
           \left(\frac{1}{4\pi}\right),

    up to the common normalization, so the truncated-to-total ratio of
    ``(2l+1) g_l^2`` measures how much of the emitter the evolution carries.
    """

    weights = angular_spectral_weights(kappa, max(ell_max, 0))
    generous = angular_spectral_weights(kappa, max(4 * ell_max + 40, 60))
    multiplicity = 2.0 * np.arange(weights.size) + 1.0
    full_multiplicity = 2.0 * np.arange(generous.size) + 1.0
    kept = float(np.sum(multiplicity * weights**2))
    total = float(np.sum(full_multiplicity * generous**2))
    return kept / total


def minimum_ell_max(
    kappa: float,
    *,
    omitted_power_tolerance: float = 1e-10,
    maximum: int = 512,
) -> int:
    """Return the first angular cutoff satisfying an omitted power target."""

    if not 0.0 < omitted_power_tolerance < 1.0:
        raise ValueError("The omitted power tolerance must lie between zero and one.")
    for ell_max in range(maximum + 1):
        if 1.0 - retained_angular_fraction(kappa, ell_max) <= omitted_power_tolerance:
            return ell_max
    raise ValueError(
        f"No ell_max up to {maximum} satisfies the omitted power tolerance."
    )


def weak_source_integral(
    source: LocalizedSourceParameters,
    test_function,
    *,
    quadrature_order: int = 24,
) -> float:
    r"""Integrate the normalized source against a smooth test function.

    ``test_function`` receives arrays ``(t, r, cosine_gamma)``.  Azimuthal
    symmetry around the source direction has already been integrated out.
    """

    if quadrature_order < 4:
        raise ValueError("At least four quadrature points are required.")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    time = source.time_center + source.time_half_width * nodes
    radius = source.center_radius + source.radial_half_width * nodes
    cosine = nodes
    time_weights = source.time_half_width * weights
    radial_weights = source.radial_half_width * weights
    angular_weights = 2.0 * pi * weights
    temporal = time_profile(time, source)
    radial = radial_profile(radius, source)
    angular = angular_profile(cosine, source)
    values = test_function(
        time[:, None, None], radius[None, :, None], cosine[None, None, :]
    )
    integral = np.einsum(
        "i,j,k,i,j,k,ijk->",
        time_weights,
        radial_weights,
        angular_weights,
        temporal,
        radial * radius**2,
        angular,
        np.broadcast_to(values, (nodes.size, nodes.size, nodes.size)),
    )
    return float(source.amplitude * integral)


def verify_angular_expansion(
    source: LocalizedSourceParameters,
    ell_max: int,
    quadrature_points: int = 400,
) -> dict[str, float]:
    r"""Compare the closed-form weights with Gauss--Legendre quadrature.

    The Legendre coefficients of the normalized profile satisfy

    .. math::

       \hat\Omega(x)=\frac{1}{4\pi}\sum_\ell (2\ell+1)\,g_\ell\,P_\ell(x),

    which is checked here against a direct numerical projection and against
    a pointwise reconstruction.  Both residuals are dominated by the
    truncation at ``l_max`` and by quadrature roundoff.
    """

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(quadrature_points)
    profile = angular_profile(nodes, source)
    legendre = np.asarray(
        [np.polynomial.legendre.Legendre.basis(ell)(nodes) for ell in range(ell_max + 1)]
    )
    measured = (
        2.0
        * pi
        * np.einsum("lx,x,x->l", legendre, profile, quadrature_weights)
    )
    exact = angular_spectral_weights(source.angular_concentration, ell_max)
    reconstruction = np.einsum(
        "l,lx->x",
        (2.0 * np.arange(ell_max + 1) + 1.0) * exact / (4.0 * pi),
        legendre,
    )
    scale = float(np.max(np.abs(profile)))
    return {
        "maximum_weight_error": float(np.max(np.abs(measured - exact))),
        "maximum_relative_reconstruction_error": float(
            np.max(np.abs(reconstruction - profile)) / scale
        ),
        "retained_angular_fraction": retained_angular_fraction(
            source.angular_concentration, ell_max
        ),
        "smallest_retained_weight": float(exact[-1]),
        "quadrature_points": int(quadrature_points),
    }


@dataclass(frozen=True)
class SourceModeCatalogue:
    r"""The angular modes that a localized emitter actually excites.

    ``amplitude[i]`` is the constant ``g_l Y^R_{lm}(theta_s, phi_s)``
    multiplying ``\mathcal{A} T(t) R(r)`` in the mode source ``S_{lm}``.
    """

    ell: np.ndarray
    m: np.ndarray
    amplitude: np.ndarray
    ell_max: int
    discarded_maximum_amplitude: float

    @property
    def count(self) -> int:
        return int(self.ell.size)

    def as_dict(self) -> dict:
        return {
            "ell_max": int(self.ell_max),
            "excited_mode_count": self.count,
            "selection_rule": "m >= 0 and (ell + m) even for an equatorial emitter",
            "largest_discarded_amplitude": float(self.discarded_maximum_amplitude),
            "largest_amplitude": float(np.max(np.abs(self.amplitude))),
            "smallest_retained_amplitude": float(np.min(np.abs(self.amplitude))),
        }


def build_mode_catalogue(
    source: LocalizedSourceParameters,
    ell_max: int,
    *,
    relative_threshold: float = 1e-13,
) -> SourceModeCatalogue:
    """Return the excited ``(l, m)`` list and its constant modal amplitudes.

    Modes whose amplitude is below ``relative_threshold`` times the largest
    amplitude are dropped.  For an equatorial emitter this removes exactly
    the modes forbidden by the reflection symmetries, and the largest
    discarded amplitude is reported so that the claim can be audited.
    """

    if ell_max < 0:
        raise ValueError("ell_max must be nonnegative.")
    weights = angular_spectral_weights(source.angular_concentration, ell_max)
    ells: list[int] = []
    orders: list[int] = []
    amplitudes: list[float] = []
    for ell in range(ell_max + 1):
        for m in range(-ell, ell + 1):
            harmonic = float(
                real_spherical_harmonic(
                    ell,
                    m,
                    np.asarray(source.source_theta),
                    np.asarray(source.source_phi),
                )
            )
            ells.append(ell)
            orders.append(m)
            amplitudes.append(weights[ell] * harmonic)
    amplitude_array = np.asarray(amplitudes, dtype=float)
    scale = float(np.max(np.abs(amplitude_array)))
    keep = np.abs(amplitude_array) > relative_threshold * scale
    discarded = amplitude_array[~keep]
    return SourceModeCatalogue(
        ell=np.asarray(ells, dtype=int)[keep],
        m=np.asarray(orders, dtype=int)[keep],
        amplitude=amplitude_array[keep],
        ell_max=int(ell_max),
        discarded_maximum_amplitude=float(
            np.max(np.abs(discarded)) if discarded.size else 0.0
        ),
    )
