"""Unit tests for Schwarzschild-de Sitter scalar-wave coefficients."""

import unittest

import numpy as np

from black_hole.sds_model import (
    BRIDGE_CHOICES,
    ScalarInitialData,
    SdSParameters,
    bridge_boost,
    metric_f,
    propagation_coefficient,
    scalar_gaussian_initial_data,
    sds_horizons,
)


class SdSModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = SdSParameters(mass=1.0, cosmological_length=10.0, ell=2)

    def test_horizon_roots(self) -> None:
        horizons = sds_horizons(self.parameters)
        self.assertLess(horizons.black_hole, horizons.cosmological)
        self.assertLess(horizons.unphysical, 0.0)
        roots = np.array(
            [horizons.black_hole, horizons.cosmological, horizons.unphysical]
        )
        np.testing.assert_allclose(
            metric_f(roots, self.parameters),
            np.zeros(3),
            atol=5e-15,
        )

    def test_future_directed_bridge_limits(self) -> None:
        rho = np.linspace(0.0, 1.0, 5001)
        for bridge in BRIDGE_CHOICES:
            with self.subTest(bridge=bridge):
                boost = bridge_boost(rho, self.parameters, bridge)
                self.assertAlmostEqual(boost[0], 1.0)
                self.assertAlmostEqual(boost[-1], -1.0)
                self.assertLess(np.max(np.abs(boost[1:-1])), 1.0)

    def test_propagation_coefficient_is_regular(self) -> None:
        rho = np.linspace(0.0, 1.0, 5001)
        for bridge in BRIDGE_CHOICES:
            with self.subTest(bridge=bridge):
                coefficient = propagation_coefficient(
                    rho, self.parameters, bridge
                )
                self.assertTrue(np.all(np.isfinite(coefficient)))
                self.assertGreater(np.min(coefficient), 0.0)

    def test_time_symmetric_initial_data(self) -> None:
        rho = np.linspace(0.001, 0.999, 10001)
        initial = ScalarInitialData()
        u, psi, pi = scalar_gaussian_initial_data(
            rho, self.parameters, initial, "minimal"
        )
        numerical_derivative = np.gradient(u, rho, edge_order=2)
        interior = slice(20, -20)
        self.assertLess(
            np.max(np.abs(psi[interior] - numerical_derivative[interior])),
            5e-4,
        )
        boost = bridge_boost(rho, self.parameters, "minimal")
        np.testing.assert_allclose(pi, -boost * psi)

    def test_invalid_sds_parameters(self) -> None:
        with self.assertRaises(ValueError):
            SdSParameters(mass=1.0, cosmological_length=1.0)
        with self.assertRaises(ValueError):
            SdSParameters(ell=-1)
        with self.assertRaises(ValueError):
            ScalarInitialData(center_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
