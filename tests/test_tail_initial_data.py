"""Tests for physically matched initially dynamical tail data."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.sds_model import (
    ArealVelocityBumpInitialData,
    SdSParameters,
    areal_radius,
    compact_areal_velocity_profile,
    propagation_coefficient,
    scalar_areal_velocity_initial_data,
)
from black_hole.schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    areal_radius as schwarzschild_areal_radius,
    propagation_coefficient as schwarzschild_propagation_coefficient,
    scalar_areal_velocity_initial_data as schwarzschild_velocity_data,
)


class TailInitialDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = ArealVelocityBumpInitialData(
            center_radius=6.0, support_half_width=3.0
        )

    def test_invalid_velocity_profiles_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArealVelocityBumpInitialData(support_half_width=0.0)
        with self.assertRaises(ValueError):
            ArealVelocityBumpInitialData(center_radius=2.0, support_half_width=3.0)
        with self.assertRaises(ValueError):
            ArealVelocityBumpInitialData(amplitude=0.0)

    def test_sds_data_has_common_physical_velocity(self) -> None:
        rho = np.linspace(0.0, 1.0, 2001)
        for length in (20.0, 40.0, 80.0, 160.0):
            model = SdSParameters(cosmological_length=length, ell=1)
            u, psi, pi = scalar_areal_velocity_initial_data(
                rho, model, self.initial, "minimal"
            )
            radius = areal_radius(rho, model)
            expected = compact_areal_velocity_profile(radius, self.initial)
            reconstructed = propagation_coefficient(
                rho, model, "minimal"
            ) * pi
            np.testing.assert_array_equal(u, np.zeros_like(u))
            np.testing.assert_array_equal(psi, np.zeros_like(psi))
            np.testing.assert_allclose(reconstructed, expected, rtol=2e-15, atol=0.0)
            self.assertEqual(pi[0], 0.0)
            self.assertEqual(pi[-1], 0.0)

    def test_schwarzschild_data_has_common_physical_velocity(self) -> None:
        rho = np.linspace(0.0, 1.0, 2001)
        model = SchwarzschildScalarParameters(ell=2)
        u, psi, pi = schwarzschild_velocity_data(rho, model, self.initial)
        radius = schwarzschild_areal_radius(rho, model)
        expected = compact_areal_velocity_profile(radius, self.initial)
        reconstructed = schwarzschild_propagation_coefficient(rho, model) * pi
        np.testing.assert_array_equal(u, np.zeros_like(u))
        np.testing.assert_array_equal(psi, np.zeros_like(psi))
        np.testing.assert_allclose(reconstructed, expected, rtol=2e-15, atol=0.0)

    def test_profile_is_identical_when_sampled_at_same_areal_radius(self) -> None:
        radius = np.linspace(2.1, 12.0, 1000)
        expected = compact_areal_velocity_profile(radius, self.initial)
        for length in (20.0, 160.0):
            model = SdSParameters(cosmological_length=length)
            from black_hole.sds_model import compact_radius

            rho = compact_radius(radius, model)
            actual_radius = areal_radius(rho, model)
            actual = compact_areal_velocity_profile(actual_radius, self.initial)
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)

    def test_support_must_fit_between_horizons(self) -> None:
        model = SdSParameters(cosmological_length=20.0)
        too_wide = ArealVelocityBumpInitialData(
            center_radius=12.0, support_half_width=8.0
        )
        with self.assertRaises(ValueError):
            scalar_areal_velocity_initial_data(
                np.linspace(0.0, 1.0, 64), model, too_wide, "minimal"
            )


if __name__ == "__main__":
    unittest.main()
