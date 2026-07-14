"""Analytic tests for the controlled Schwarzschild flat limit."""

import unittest

import numpy as np

from black_hole.schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    minimal_boost as schwarzschild_boost,
    minimal_height as schwarzschild_height,
    propagation_coefficient as schwarzschild_propagation,
    rescaled_scalar_potential as schwarzschild_potential,
    scalar_gaussian_initial_data as schwarzschild_initial_data,
)
from black_hole.sds_model import (
    ScalarInitialData,
    SdSParameters,
    areal_radius,
    bridge_boost,
    compact_radius,
    metric_f,
    minimal_height as sds_height,
    propagation_coefficient,
    propagation_endpoint_coefficients,
    rescaled_scalar_potential,
    scalar_gaussian_initial_data,
    sds_horizons,
)


class FlatLimitModelTests(unittest.TestCase):
    def test_professor_compactification_and_inverse(self) -> None:
        model = SdSParameters(cosmological_length=20.0)
        horizons = sds_horizons(model)
        rho = np.linspace(0.0, 1.0, 1001)
        radius = areal_radius(rho, model)
        self.assertEqual(radius[0], horizons.black_hole)
        self.assertEqual(radius[-1], horizons.cosmological)
        np.testing.assert_allclose(compact_radius(radius, model), rho, atol=2e-15)

    def test_analytic_endpoint_coefficients_match_interior_limits(self) -> None:
        model = SdSParameters(cosmological_length=20.0)
        left, right = propagation_endpoint_coefficients(model, "minimal")
        epsilon = 1e-7
        interior = propagation_coefficient(
            np.array([epsilon, 1.0 - epsilon]), model, "minimal"
        )
        np.testing.assert_allclose(interior, [left, right], rtol=2e-6)
        endpoints = propagation_coefficient(
            np.array([0.0, 1.0]), model, "minimal"
        )
        np.testing.assert_array_equal(endpoints, [left, right])

    def test_sds_coefficients_approach_schwarzschild(self) -> None:
        rho = np.linspace(0.0, 1.0, 2001)
        sds = SdSParameters(cosmological_length=10_000.0)
        schwarzschild = SchwarzschildScalarParameters()
        self.assertLess(
            np.max(np.abs(bridge_boost(rho, sds, "minimal") - schwarzschild_boost(rho))),
            3e-4,
        )
        self.assertLess(
            np.max(
                np.abs(
                    propagation_coefficient(rho, sds, "minimal")
                    - schwarzschild_propagation(rho, schwarzschild)
                )
            ),
            2e-5,
        )
        # The flat limit is pointwise on compact subsets rho<1.  It is not
        # uniform exactly at the receding cosmological horizon r_c~L, where
        # the O(L^-2) curvature terms and d rho/dr are of the same order.
        compact_subset = rho <= 0.99
        self.assertLess(
            np.max(
                np.abs(
                    rescaled_scalar_potential(rho[compact_subset], sds)
                    - schwarzschild_potential(
                        rho[compact_subset], schwarzschild
                    )
                )
            ),
            2e-3,
        )

    def test_initial_profile_is_identical_in_rho(self) -> None:
        rho = np.linspace(0.0, 1.0, 1001)
        initial = ScalarInitialData()
        sds = SdSParameters(cosmological_length=20.0)
        schwarzschild = SchwarzschildScalarParameters()
        sds_u, sds_psi, _ = scalar_gaussian_initial_data(
            rho, sds, initial, "minimal"
        )
        schwarzschild_u, schwarzschild_psi, _ = schwarzschild_initial_data(
            rho, schwarzschild, initial
        )
        np.testing.assert_array_equal(sds_u, schwarzschild_u)
        np.testing.assert_array_equal(sds_psi, schwarzschild_psi)
        self.assertLess(max(sds_u[0], sds_u[-1]), 1e-12)

    def test_common_height_normalization(self) -> None:
        reference_radius = 4.0
        sds = SdSParameters(cosmological_length=20.0)
        schwarzschild = SchwarzschildScalarParameters()
        self.assertAlmostEqual(
            float(sds_height(reference_radius, sds, reference_radius)), 0.0
        )
        self.assertAlmostEqual(
            float(
                schwarzschild_height(
                    reference_radius, schwarzschild, reference_radius
                )
            ),
            0.0,
        )

        radius = 5.0
        step = 1e-5
        sds_derivative = float(
            (
                sds_height(radius + step, sds, reference_radius)
                - sds_height(radius - step, sds, reference_radius)
            )
            / (2.0 * step)
        )
        sds_expected = float(
            bridge_boost(compact_radius(radius, sds), sds, "minimal")
            / metric_f(radius, sds)
        )
        self.assertAlmostEqual(sds_derivative, sds_expected, places=9)

        schwarzschild_derivative = float(
            (
                schwarzschild_height(
                    radius + step, schwarzschild, reference_radius
                )
                - schwarzschild_height(
                    radius - step, schwarzschild, reference_radius
                )
            )
            / (2.0 * step)
        )
        schwarzschild_rho = 1.0 - 2.0 * schwarzschild.mass / radius
        schwarzschild_expected = float(
            schwarzschild_boost(schwarzschild_rho)
            / (1.0 - 2.0 * schwarzschild.mass / radius)
        )
        self.assertAlmostEqual(
            schwarzschild_derivative, schwarzschild_expected, places=9
        )


if __name__ == "__main__":
    unittest.main()
