"""Unit tests for the analytic model, independent of Dedalus."""

import unittest

import numpy as np

from black_hole.model import (
    InitialData,
    ModelParameters,
    boost,
    gaussian_initial_data,
    propagation_coefficient,
    rescaled_potential,
)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = ModelParameters(mass=1.0, ell=2)

    def test_boundary_coefficients(self) -> None:
        rho = np.array([0.0, 1.0])
        np.testing.assert_allclose(boost(rho), [1.0, -1.0])
        np.testing.assert_allclose(
            propagation_coefficient(rho, self.parameters), [1 / 16, 1 / 8]
        )
        np.testing.assert_allclose(
            rescaled_potential(rho, self.parameters), [1.5, 3.0]
        )

    def test_gaussian_constraint(self) -> None:
        rho = np.linspace(0, 1, 10001)
        initial = InitialData()
        u, psi, pi = gaussian_initial_data(rho, initial)
        numerical_derivative = np.gradient(u, rho, edge_order=2)
        interior = slice(10, -10)
        self.assertLess(
            np.max(np.abs(psi[interior] - numerical_derivative[interior])), 1e-4
        )
        np.testing.assert_array_equal(pi, np.zeros_like(pi))

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            ModelParameters(mass=0)
        with self.assertRaises(ValueError):
            ModelParameters(ell=1)
        with self.assertRaises(ValueError):
            InitialData(width=0)


if __name__ == "__main__":
    unittest.main()
