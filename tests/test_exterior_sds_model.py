"""Tests for the exterior-supported artificial cosmology."""

import unittest

import numpy as np

from black_hole.exterior_sds_model import (
    ExteriorSdSParameters,
    TRANSITION_MINIMUM_ANGLE_WIDTH,
    TRANSITION_OUTER_HORIZON_FRACTION,
    TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M,
    areal_radius,
    background_audit,
    bridge_boost,
    bridge_one_minus_boost,
    bridge_one_plus_boost,
    chebyshev_angle,
    compact_radius,
    compactification_derivative,
    metric_f,
    metric_f_prime,
    propagation_coefficient,
    propagation_endpoint_coefficients,
    rescaled_scalar_potential,
    retarded_time_offset,
    smooth_step,
    smooth_step_derivative,
    transition_compact_radii,
    transition_profile,
    transition_radial_derivative,
    transition_radii,
)


class ExteriorSdSModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = ExteriorSdSParameters(
            mass=1.0, cosmological_length=160.0, ell=2
        )

    def test_compactification_and_transition_geometry(self) -> None:
        rho = np.linspace(0.0, 1.0, 1001)
        radius = areal_radius(rho, self.parameters)
        np.testing.assert_allclose(compact_radius(radius, self.parameters), rho)
        self.assertEqual(radius[0], 2.0)
        self.assertEqual(radius[-1], self.parameters.cosmological_horizon)

        r0, r1 = transition_radii(self.parameters)
        rho0, rho1 = transition_compact_radii(self.parameters)
        self.assertAlmostEqual(compact_radius(np.array(r0), self.parameters), rho0)
        self.assertAlmostEqual(compact_radius(np.array(r1), self.parameters), rho1)
        theta0 = float(chebyshev_angle(np.array(rho0)))
        theta1 = float(chebyshev_angle(np.array(rho1)))
        self.assertAlmostEqual(theta0 - theta1, theta1, places=14)
        self.assertGreater(r0, 6.0)
        self.assertAlmostEqual(
            r1,
            TRANSITION_OUTER_HORIZON_FRACTION
            * self.parameters.cosmological_horizon,
            places=12,
        )

        reference = ExteriorSdSParameters(
            mass=1.0,
            cosmological_length=TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M,
            ell=2,
        )
        reference_rho0, reference_rho1 = transition_compact_radii(reference)
        _, reference_r1 = transition_radii(reference)
        self.assertAlmostEqual(
            reference_r1,
            TRANSITION_OUTER_HORIZON_FRACTION
            * reference.cosmological_horizon,
            places=11,
        )
        self.assertAlmostEqual(
            float(chebyshev_angle(np.array(reference_rho1))),
            TRANSITION_MINIMUM_ANGLE_WIDTH,
            places=14,
        )
        self.assertAlmostEqual(reference_rho0, 0.9943459581991452, places=14)
        self.assertAlmostEqual(reference_rho1, 0.9985844858695327, places=14)

    def test_transition_angle_width_has_a_floor(self) -> None:
        lengths = (80.0, 160.0, 320.0, 640.0, 5120.0)
        endpoints = [
            transition_compact_radii(
                ExteriorSdSParameters(cosmological_length=length)
            )
            for length in lengths
        ]
        angle_widths = np.asarray(
            [
                float(chebyshev_angle(np.array(rho0)))
                - float(chebyshev_angle(np.array(rho1)))
                for rho0, rho1 in endpoints
            ]
        )
        self.assertTrue(np.all(np.diff(angle_widths[:2]) < 0.0))
        np.testing.assert_allclose(
            angle_widths[1:],
            TRANSITION_MINIMUM_ANGLE_WIDTH,
            rtol=0.0,
            atol=1.0e-14,
        )
        for length, (rho0, rho1) in zip(lengths, endpoints):
            parameters = ExteriorSdSParameters(cosmological_length=length)
            _, r1 = transition_radii(parameters)
            if length <= TRANSITION_WIDTH_REFERENCE_LENGTH_OVER_M:
                np.testing.assert_allclose(
                    r1,
                    TRANSITION_OUTER_HORIZON_FRACTION
                    * parameters.cosmological_horizon,
                    rtol=2.0e-12,
                    atol=2.0e-14,
                )
            else:
                self.assertLess(
                    r1,
                    TRANSITION_OUTER_HORIZON_FRACTION
                    * parameters.cosmological_horizon,
                )
            self.assertGreater(rho1 - rho0, 0.0)

    def test_model_metadata_records_width_floor_geometry(self) -> None:
        metadata = ExteriorSdSParameters(cosmological_length=640.0).as_dict()
        self.assertTrue(metadata["transition_width_floor_active"])
        self.assertGreater(metadata["transition_compact_width"], 0.0)
        self.assertGreater(metadata["outer_cap_compact_width"], 0.0)
        self.assertLess(metadata["actual_transition_outer_horizon_fraction"], 0.9)
        self.assertAlmostEqual(
            metadata["transition_outer_rho"], 0.9985844858695327
        )

    def test_smooth_step_and_analytic_derivative(self) -> None:
        x = np.linspace(-0.1, 1.1, 5001)
        step = smooth_step(x)
        derivative = smooth_step_derivative(x)
        self.assertTrue(np.all(step[x <= 0.0] == 0.0))
        self.assertTrue(np.all(step[x >= 1.0] == 1.0))
        self.assertTrue(np.all(derivative[(x <= 0.0) | (x >= 1.0)] == 0.0))
        interior = (x > 0.02) & (x < 0.98)
        numerical = np.gradient(step, x, edge_order=2)
        np.testing.assert_allclose(
            derivative[interior], numerical[interior], rtol=2.0e-4, atol=2.0e-7
        )

    def test_exact_schwarzschild_and_uniform_sds_regions(self) -> None:
        r0, r1 = transition_radii(self.parameters)
        radius = np.array([4.0, 0.9 * r0, 1.1 * r1, 0.9 * self.parameters.cosmological_horizon])
        rho = compact_radius(radius, self.parameters)
        chi = transition_profile(rho, self.parameters)
        self.assertEqual(chi[0], 0.0)
        self.assertEqual(chi[1], 0.0)
        self.assertEqual(chi[2], 1.0)
        self.assertEqual(chi[3], 1.0)

        mass = self.parameters.mass
        length = self.parameters.cosmological_length
        np.testing.assert_allclose(
            metric_f(radius[:2], self.parameters),
            1.0 - 2.0 * mass / radius[:2],
        )
        np.testing.assert_allclose(
            metric_f(radius[2:], self.parameters),
            1.0 - 2.0 * mass / radius[2:] - radius[2:] ** 2 / length**2,
        )

    def test_analytic_radial_derivatives(self) -> None:
        r0, r1 = transition_radii(self.parameters)
        radius = np.linspace(r0 + 0.05 * (r1 - r0), r1 - 0.05 * (r1 - r0), 4001)
        chi = transition_profile(compact_radius(radius, self.parameters), self.parameters)
        numerical_chi_prime = np.gradient(chi, radius, edge_order=2)
        numerical_f_prime = np.gradient(
            metric_f(radius, self.parameters), radius, edge_order=2
        )
        interior = slice(20, -20)
        np.testing.assert_allclose(
            transition_radial_derivative(radius, self.parameters)[interior],
            numerical_chi_prime[interior],
            rtol=3.0e-4,
            atol=3.0e-7,
        )
        np.testing.assert_allclose(
            metric_f_prime(radius, self.parameters)[interior],
            numerical_f_prime[interior],
            rtol=3.0e-4,
            atol=3.0e-7,
        )

    def test_horizons_and_regular_coefficients(self) -> None:
        rho = np.linspace(0.0, 1.0, 20_001)
        radius = areal_radius(rho, self.parameters)
        lapse = metric_f(radius, self.parameters)
        self.assertEqual(lapse[0], 0.0)
        self.assertEqual(lapse[-1], 0.0)
        self.assertTrue(np.all(lapse[1:-1] > 0.0))

        boost = bridge_boost(rho, self.parameters)
        self.assertEqual(boost[0], 1.0)
        self.assertEqual(boost[-1], -1.0)
        self.assertTrue(np.all(np.abs(boost[1:-1]) < 1.0))

        coefficient = propagation_coefficient(rho, self.parameters)
        potential = rescaled_scalar_potential(rho, self.parameters)
        self.assertTrue(np.all(np.isfinite(coefficient)))
        self.assertTrue(np.all(np.isfinite(potential)))
        self.assertTrue(np.all(coefficient > 0.0))

    def test_factorized_endpoint_coefficients_remain_resolved(self) -> None:
        parameters = ExteriorSdSParameters(cosmological_length=640.0)
        resolution = 2304
        theta = np.pi * (np.arange(resolution) + 0.5) / resolution
        rho = 0.5 * (1.0 + np.cos(theta))
        one_plus = bridge_one_plus_boost(rho, parameters)
        one_minus = bridge_one_minus_boost(rho, parameters)
        boost = bridge_boost(rho, parameters)
        coefficient = propagation_coefficient(rho, parameters)

        self.assertTrue(np.all(one_plus > 0.0))
        self.assertTrue(np.all(one_minus > 0.0))
        np.testing.assert_allclose(one_plus + one_minus, 2.0, atol=5.0e-16)
        np.testing.assert_allclose(
            boost, 0.5 * (one_plus - one_minus), rtol=0.0, atol=0.0
        )
        self.assertTrue(np.all(np.isfinite(coefficient)))
        self.assertGreater(np.min(coefficient), 0.0)
        self.assertLess(
            abs(coefficient[0] - propagation_endpoint_coefficients(parameters)[1]),
            2.0e-7,
        )

    def test_retarded_time_has_schwarzschild_limit(self) -> None:
        lengths = (80.0, 160.0, 320.0, 640.0)
        offsets = np.array(
            [
                retarded_time_offset(
                    ExteriorSdSParameters(cosmological_length=length), 4.0
                )
                for length in lengths
            ]
        )
        schwarzschild_offset = 4.0 * np.log(2.0)
        errors = schwarzschild_offset - offsets
        self.assertTrue(np.all(errors > 0.0))
        np.testing.assert_allclose(errors[:-1] / errors[1:], 2.0, rtol=0.03)

    def test_fixed_radius_coefficients_have_schwarzschild_limit(self) -> None:
        radius = np.array([4.0, 100.0, 1000.0])
        schwarzschild_lapse = 1.0 - 2.0 / radius
        errors = []
        for length in (1280.0, 2560.0, 5120.0, 10240.0):
            parameters = ExteriorSdSParameters(cosmological_length=length)
            errors.append(
                np.max(np.abs(metric_f(radius, parameters) - schwarzschild_lapse))
            )
        self.assertTrue(np.all(np.diff(errors) <= 1.0e-14))
        self.assertLess(errors[-1], 0.02 * errors[0])

    def test_background_audit_passes_production_lengths(self) -> None:
        for length in (80.0, 160.0, 320.0, 640.0, 5120.0):
            with self.subTest(length=length):
                audit = background_audit(
                    ExteriorSdSParameters(cosmological_length=length)
                )
                self.assertTrue(audit["finite_coefficients"])
                self.assertTrue(audit["positive_interior_lapse"])
                self.assertTrue(audit["spacelike_bridge_interior"])
                self.assertTrue(audit["nonnegative_scalar_potential"])


if __name__ == "__main__":
    unittest.main()
