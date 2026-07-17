"""Analytic tests for the controlled Schwarzschild flat limit."""

import unittest

import numpy as np

from black_hole.schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    minimal_boost as schwarzschild_boost,
    minimal_height as schwarzschild_height,
    propagation_coefficient as schwarzschild_propagation,
    retarded_time_offset as schwarzschild_retarded_time_offset,
    rescaled_scalar_potential as schwarzschild_potential,
    scalar_gaussian_initial_data as schwarzschild_initial_data,
    scalar_areal_bump_initial_data as schwarzschild_areal_initial_data,
    tortoise_coordinate as schwarzschild_tortoise,
)
from black_hole.sds_model import (
    ArealBumpInitialData,
    ScalarInitialData,
    SdSParameters,
    areal_radius,
    bridge_boost,
    compact_radius,
    metric_f,
    minimal_height as sds_height,
    propagation_coefficient,
    propagation_endpoint_coefficients,
    retarded_time_offset as sds_retarded_time_offset,
    rescaled_scalar_potential,
    scalar_gaussian_initial_data,
    scalar_areal_bump_initial_data,
    sds_horizons,
    tortoise_coordinate as sds_tortoise,
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

    def test_legacy_initial_profile_is_identical_in_rho(self) -> None:
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

    def test_compact_initial_profile_is_identical_in_areal_radius(self) -> None:
        initial = ArealBumpInitialData(center_radius=4.0, support_half_width=1.0)
        sds = SdSParameters(cosmological_length=20.0)
        schwarzschild = SchwarzschildScalarParameters()
        radius = np.linspace(2.8, 5.2, 2001)
        rho_sds = compact_radius(radius, sds)
        rho_schwarzschild = 1.0 - 2.0 * schwarzschild.mass / radius
        sds_u, sds_psi, sds_pi = scalar_areal_bump_initial_data(
            rho_sds, sds, initial, "minimal"
        )
        schwarzschild_u, schwarzschild_psi, schwarzschild_pi = (
            schwarzschild_areal_initial_data(
                rho_schwarzschild, schwarzschild, initial
            )
        )
        np.testing.assert_allclose(sds_u, schwarzschild_u, atol=2e-15)
        sds_du_dr = sds_psi * (
            sds_horizons(sds).black_hole
            / (
                (1.0 - sds_horizons(sds).black_hole / sds_horizons(sds).cosmological)
                * radius**2
            )
        )
        schwarzschild_du_dr = schwarzschild_psi * (
            2.0 * schwarzschild.mass / radius**2
        )
        np.testing.assert_allclose(sds_du_dr, schwarzschild_du_dr, atol=2e-14)
        np.testing.assert_allclose(
            sds_pi,
            -bridge_boost(rho_sds, sds, "minimal") * sds_psi,
        )
        np.testing.assert_allclose(
            schwarzschild_pi,
            -schwarzschild_boost(rho_schwarzschild) * schwarzschild_psi,
        )
        self.assertTrue(np.all(sds_u[(radius <= 3.0) | (radius >= 5.0)] == 0.0))

    def test_retarded_time_offsets_are_analytic_endpoint_limits(self) -> None:
        reference_radius = 4.0
        sds = SdSParameters(cosmological_length=20.0)
        horizons = sds_horizons(sds)
        q_sds = sds_retarded_time_offset(sds, reference_radius)
        radius = horizons.cosmological - 1e-8 * horizons.width
        endpoint_sum = float(
            sds_height(radius, sds, reference_radius)
            + sds_tortoise(radius, sds, reference_radius)
        )
        self.assertAlmostEqual(endpoint_sum, q_sds, places=6)

        schwarzschild = SchwarzschildScalarParameters()
        q_schwarzschild = schwarzschild_retarded_time_offset(
            schwarzschild, reference_radius
        )
        radius = 1e8 * schwarzschild.mass
        endpoint_sum = float(
            schwarzschild_height(radius, schwarzschild, reference_radius)
            + schwarzschild_tortoise(radius, schwarzschild, reference_radius)
        )
        self.assertAlmostEqual(endpoint_sum, q_schwarzschild, places=6)
        self.assertAlmostEqual(q_schwarzschild, 4.0 * np.log(2.0), places=14)

    def test_retarded_time_offsets_approach_schwarzschild(self) -> None:
        reference_radius = 4.0
        q_zero = schwarzschild_retarded_time_offset(
            SchwarzschildScalarParameters(), reference_radius
        )
        errors = [
            abs(
                sds_retarded_time_offset(
                    SdSParameters(cosmological_length=length), reference_radius
                )
                - q_zero
            )
            for length in (40.0, 80.0, 160.0, 320.0)
        ]
        self.assertTrue(all(later < earlier for earlier, later in zip(errors, errors[1:])))

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
