"""High-precision coefficients for the independent SBP--Hermite experiment.

This adapter reproduces the frozen scalar backgrounds and physical velocity
data without importing their double-precision evaluation paths.  It does not
change the archived production models.  ``p=f*d(rho)/dr`` and ``P=V/p`` use
the first-order production convention; the equivalent second-order equation
is ``u_tt/A = d_rho(p*u_rho) + 2*B*u_trho + B_rho*u_t - P*u``.

All inputs, horizons, transition derivatives, and endpoint limits are evaluated
with an instance-local mpmath context.  Numbers intended to carry more than
binary64 precision should be supplied as strings or mpmath numbers.
"""

from __future__ import annotations

import mpmath


class Geometry:
    """Schwarzschild, uniform SdS, or the frozen exterior width-floor family.

    Parameters ``L`` and ``mass`` are dimensional lengths in the same units.
    The velocity bump is centered at ``6M`` with half-width ``3M``; its peak
    Killing-time velocity is one, as in the production tail experiment.
    ``rho0`` and ``rho1`` exist only for the exterior background.  ``rc`` is
    infinity for Schwarzschild.  ``clock_offset`` fixes ``U=tau-q`` using the
    same height and tortoise normalization at areal radius ``4M``.
    """

    def __init__(
        self, background, L=None, ell=1, xi="0", mass="1", dps=50,
        initial_data="velocity"
    ):
        self.mp = mpmath.mp.clone()
        self.mp.dps = int(dps)
        if self.mp.dps < 30:
            raise ValueError("The high-precision adapter requires dps >= 30.")
        aliases = {"sds": "uniform", "exterior_sds": "exterior", "schw": "schwarzschild"}
        self.background = aliases.get(str(background), str(background))
        if self.background not in {"schwarzschild", "uniform", "exterior"}:
            raise ValueError("Unknown scalar background.")
        self.mass = self.mp.mpf(mass)
        if initial_data not in {"velocity", "displacement"}:
            raise ValueError("Initial data must be velocity or displacement.")
        self.initial_data = initial_data
        self.initial_center = self.mp.mpf(6 if initial_data == "velocity" else 4) * self.mass
        self.initial_half_width = self.mp.mpf(3 if initial_data == "velocity" else "1.5") * self.mass
        self.ell = int(ell)
        if self.ell != ell or self.ell < 0 or self.mass <= 0:
            raise ValueError("Require positive mass and integer ell >= 0.")
        self.xi = self.mp.mpf(xi)
        if not self.mp.isfinite(self.xi):
            raise ValueError("Curvature coupling must be finite.")
        self.L = self.mp.inf if L is None else self.mp.mpf(L)
        self.rho0 = self.rho1 = None
        self.theta0 = self.theta1 = None
        self._clock = None
        m = self.mass
        if self.background == "schwarzschild":
            self.rb, self.rc, self.ru = 2 * m, self.mp.inf, -self.mp.inf
            self.D = self.mp.one
        else:
            if not self.mp.isfinite(self.L) or self.L <= 3 * self.mp.sqrt(3) * m:
                raise ValueError("Require finite L > 3*sqrt(3)*M.")
            angle = self.mp.asin(3 * self.mp.sqrt(3) * m / self.L)
            self.uniform_rb = 2 * self.L / self.mp.sqrt(3) * self.mp.sin(angle / 3)
            self.rc = 2 * self.L / self.mp.sqrt(3) * self.mp.sin((self.mp.pi - angle) / 3)
            self.ru = -self.uniform_rb - self.rc
            self.rb = self.uniform_rb if self.background == "uniform" else 2 * m
            self.D = 1 - self.rb / self.rc
            rb, rc, ru = self.uniform_rb, self.rc, self.ru
            self.kb = (rc - rb) * (rb - ru) / (2 * self.L**2 * rb)
            self.kc = (rc - rb) * (rc - ru) / (2 * self.L**2 * rc)
            self.ku = (rb - ru) * (rc - ru) / (2 * self.L**2 * (-ru))
            if self.background == "uniform":
                # The exact minimal-gauge boost is quadratic in rho.  Its
                # prescribed endpoints give B=1+c*rho-(2+c)*rho**2.
                r = self.areal_radius(self.mp.mpf("0.5"))
                boost = (
                    rb * (rc - r) * (r - ru) / (2 * self.kb * self.L**2 * r**2)
                    - rc * (r - rb) * (r - ru) / (2 * self.kc * self.L**2 * r**2)
                    - ru * (r - rb) * (rc - r) / (2 * self.ku * self.L**2 * r**2)
                )
                self.boost_linear = 4 * boost - 2
            else:
                horizon_scaled_rho1 = self.compact_radius(self.mp.mpf("0.9") * rc)
                if not 0 < horizon_scaled_rho1 < 1:
                    raise ValueError("Exterior transition is outside the physical domain.")
                # The frozen model defines this decimal constant, not a
                # newly recomputed L/M=160 value.  Preserve its exact value.
                minimum_angle = self.mp.mpf("0.07526440137447271")
                self.theta1 = max(self.mp.acos(2 * horizon_scaled_rho1 - 1), minimum_angle)
                self.theta0 = 2 * self.theta1
                self.rho0 = (1 + self.mp.cos(self.theta0)) / 2
                self.rho1 = (1 + self.mp.cos(self.theta1)) / 2
                if not 0 < self.rho0 < self.rho1 < 1:
                    raise ValueError("Exterior transition is outside the physical domain.")
        support = (self.initial_center-self.initial_half_width,
                   self.initial_center+self.initial_half_width)
        if self.rb >= support[0] or self.rc <= support[1]:
            raise ValueError("The frozen bump support must lie between horizons.")
        self.support_rho = tuple(self.compact_radius(r) for r in support)

    def areal_radius(self, rho):
        """Areal radius, with exact endpoint assignments."""
        rho = self.mp.mpf(rho)
        if rho == 1:
            return self.rc
        return self.rb / (1 - self.D * rho)

    def compact_radius(self, radius):
        """Compact coordinate of an areal radius."""
        return (1 - self.rb / self.mp.mpf(radius)) / self.D

    def transition(self, rho):
        """Return ``chi, dchi/drho, d2chi/drho2`` in high precision."""
        rho = self.mp.mpf(rho)
        if self.background != "exterior" or rho <= self.rho0:
            return self.mp.zero, self.mp.zero, self.mp.zero
        if rho >= self.rho1:
            return self.mp.one, self.mp.zero, self.mp.zero
        width = self.theta0 - self.theta1
        theta = self.mp.acos(2 * rho - 1)
        x = (self.theta0 - theta) / width
        logit = -1 / x + 1 / (1 - x)
        # A symmetric logistic evaluation avoids overflow and also retains
        # the tiny complementary step near the upper endpoint.
        if logit >= 0:
            exp_minus = self.mp.exp(-logit)
            chi = 1 / (1 + exp_minus)
            product_step = exp_minus / (1 + exp_minus)**2
        else:
            exp_plus = self.mp.exp(logit)
            chi = exp_plus / (1 + exp_plus)
            product_step = exp_plus / (1 + exp_plus)**2
        q = 1 / x**2 + 1 / (1 - x)**2
        q_prime = -2 / x**3 + 2 / (1 - x)**3
        chi_x = product_step * q
        chi_xx = product_step * ((1 - 2 * chi) * q**2 + q_prime)
        product = rho * (1 - rho)
        x_rho = 1 / (width * self.mp.sqrt(product))
        x_rhorho = (2 * rho - 1) / (2 * width * product**self.mp.mpf("1.5"))
        return chi, chi_x * x_rho, chi_xx * x_rho**2 + chi_x * x_rhorho

    def coefficients(self, rho):
        """Return regular ``(A, B, p, P)`` including both null boundaries."""
        rho = self.mp.mpf(rho)
        if not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0,1].")
        m, angular = self.mass, self.ell * (self.ell + 1)
        if self.background == "schwarzschild":
            w = 1 - rho
            A = 1 / (8 * m * (2 - rho))
            B = -1 + 2 * w**2
            p = rho * w**2 / (2 * m)
            P = (angular + w) / (2 * m)
            return A, B, p, P
        r, D, rb, rc = self.areal_radius(rho), self.D, self.rb, self.rc
        G = rb / (D * r**2)
        if self.background == "uniform":
            w, c = 1 - D * rho, self.boost_linear
            one_minus_factor = -c + (2 + c) * rho
            one_plus_factor = 2 + (2 + c) * rho
            common = rc * D * (rb - self.ru * w) / (self.L**2 * rb)
            p = common * rho * (1 - rho)
            A = common / (one_minus_factor * one_plus_factor)
            B = 1 + rho * (c - (2 + c) * rho)
            P = D / rb * (
                angular + 2 * m / r + (-2 + 12 * self.xi) * r**2 / self.L**2
            )
            return A, B, p, P
        chi, chi_rho, chi_rhorho = self.transition(rho)
        chi_r = chi_rho * G
        chi_rr = chi_rhorho * G**2 - 2 * chi_rho * G / r
        # Factor rc-r analytically in the compact coordinate.
        delta_c = rc * D * (1 - rho) / (1 - D * rho)
        cap_factor = (rc + r) / self.L**2 - 2 * m / (r * rc)
        lapse = D * rho if chi == 0 else delta_c * cap_factor + r**2 * (1 - chi) / self.L**2
        p = lapse * G
        one_plus = 8 * m**2 / rc**2 * (delta_c * (rc + r) / r**2 + 1 - chi)
        one_minus = 2 * D * rho * (r + 2 * m) / r + 8 * m**2 * chi / rc**2
        B = (one_plus - one_minus) / 2
        if chi == 0:
            A = r / (8 * m * D * (r + 2 * m))
        elif chi == 1:
            A = rc**2 * cap_factor / (4 * m * D * (rc + r) * one_minus)
        else:
            A = p / (one_minus * one_plus)
        if rho == 0:
            B, p = self.mp.one, self.mp.zero
        elif rho == 1:
            B, p = -self.mp.one, self.mp.zero
        ricci = (12 * chi + 8 * r * chi_r + r**2 * chi_rr) / self.L**2
        lapse_r = 2 * m / r**2 - (2 * r * chi + r**2 * chi_r) / self.L**2
        P = D / (2 * m) * (angular + r * lapse_r + self.xi * r**2 * ricci)
        return A, B, p, P

    def initial(self, rho):
        """Frozen physical initial data ``(u, u_tau)`` without momentum conversion."""
        rho = self.mp.mpf(rho)
        # Test the exact compact support before the inverse map.  Otherwise
        # roundoff in r(rho(9M)) can move a support endpoint infinitesimally
        # inward and manufacture a nonzero bump with an enormous exponent.
        if rho <= self.support_rho[0] or rho >= self.support_rho[1]:
            return self.mp.zero, self.mp.zero
        r = self.areal_radius(rho)
        x = (r - self.initial_center) / self.initial_half_width
        if abs(x) >= 1:
            return self.mp.zero, self.mp.zero
        bump = self.mp.exp(1 - 1 / (1 - x**2))
        return ((self.mp.zero, bump) if self.initial_data == "velocity"
                else (bump, self.mp.zero))

    def clock_offset(self):
        """Return the fixed outer retarded offset q, never fitted to a waveform."""
        if self._clock is not None:
            return self._clock
        m, ref = self.mass, 4 * self.mass
        if self.background == "schwarzschild":
            self._clock = 4 * m * self.mp.log(2)
        elif self.background == "uniform":
            rb, rc = self.rb, self.rc
            self._clock = self.mp.log((rc - rb)**2 * ref / (rc * (ref - rb)**2)) / (2 * self.kb) + (
                1 / (2 * self.ku) - 1 / (2 * self.kc)
            ) * self.mp.log(rc / ref)
        else:
            left = self.compact_radius(ref)
            boundaries = [left] + [x for x in (self.rho0, self.rho1) if left < x < 1] + [self.mp.one]

            def integrand(rho):
                A, B, _, _ = self.coefficients(rho)
                return 1 / (A * (1 - B))

            self._clock = self.mp.quad(integrand, boundaries)
        return self._clock

    def metadata(self):
        """Serializable decimal metadata for an independent-run manifest."""
        number = lambda value: None if value is None else self.mp.nstr(value, self.mp.dps)
        return {
            "background": self.background,
            "mass": number(self.mass),
            "cosmological_length": number(self.L),
            "ell": self.ell,
            "curvature_coupling": number(self.xi),
            "coefficient_decimal_digits": self.mp.dps,
            "black_hole_radius": number(self.rb),
            "outer_radius": number(self.rc),
            "transition_inner_rho": number(self.rho0),
            "transition_outer_rho": number(self.rho1),
            "initial_support_rho": [number(x) for x in self.support_rho],
            "initial_data_kind": self.initial_data,
            "initial_data": ("u=0, d_tau u=C-infinity areal bump centered at 6M with half-width 3M"
                             if self.initial_data == "velocity" else
                             "u=C-infinity areal bump centered at 4M with half-width 1.5M, d_tau u=0"),
            "retarded_offset": number(self.clock_offset()),
            "retarded_clock": "U=tau-q; height and tortoise normalized at r=4M",
        }
