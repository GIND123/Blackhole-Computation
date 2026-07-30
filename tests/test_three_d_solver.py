"""Tests for the angular-spectral pure-mode 3D validation solver."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.three_d_solver import (
    PureMode,
    RealSphericalHarmonicBasis,
    ThreeDNumericalParameters,
    UniformFiniteDifference,
    run_pure_mode_simulation,
)


class AngularBasisTests(unittest.TestCase):
    def test_real_harmonics_are_discretely_orthonormal(self) -> None:
        basis = RealSphericalHarmonicBasis(ell_max=3)
        error = basis.gram_matrix() - np.eye(basis.mode_count)
        self.assertLess(np.max(np.abs(error)), 2e-14)

    def test_pure_modes_roundtrip_without_resolved_contamination(self) -> None:
        basis = RealSphericalHarmonicBasis(ell_max=2)
        for ell, m in ((0, 0), (1, 1), (2, 2), (2, -1)):
            diagnostics = basis.roundtrip_diagnostics(ell, m)
            self.assertGreater(diagnostics["purity"], 1.0 - 5e-15)
            self.assertLess(diagnostics["maximum_off_mode_amplitude"], 2e-14)


class RadialDiscretizationTests(unittest.TestCase):
    def test_eighth_order_stencil_is_exact_on_degree_eight_polynomial(self) -> None:
        derivative = UniformFiniteDifference(resolution=65, order=8)
        rho = np.linspace(0.0, 1.0, 65)
        measured = derivative.differentiate(rho**8)
        self.assertLess(np.max(np.abs(measured - 8.0 * rho**7)), 2e-11)

    def test_local_interpolation_is_exact_on_degree_eight_polynomial(self) -> None:
        derivative = UniformFiniteDifference(resolution=65, order=8)
        rho = np.linspace(0.0, 1.0, 65)
        target = 0.73125
        measured = derivative.interpolate(rho**8, target)
        self.assertAlmostEqual(float(measured), target**8, places=13)


class ShortEvolutionTests(unittest.TestCase):
    def test_pure_mode_preserves_constraint_and_mode_purity(self) -> None:
        result = run_pure_mode_simulation(
            background="sds",
            mode=PureMode(ell=1, m=1),
            numerical=ThreeDNumericalParameters(
                radial_resolution=96,
                angular_ell_max=2,
                timestep=0.0025,
                end_time=0.1,
                signal_dt=0.05,
                diagnostic_dt=0.05,
                snapshot_dt=0.1,
            ),
            cosmological_length=80.0,
        )
        self.assertEqual(result.modal_signals.shape[-1], 9)
        self.assertLess(np.max(result.constraint_linf), 2e-12)
        self.assertGreater(np.min(result.mode_purity), 1.0 - 5e-15)
        target = result.target_mode_index
        wrong = np.delete(result.modal_signals, target, axis=2)
        self.assertEqual(float(np.max(np.abs(wrong))), 0.0)


if __name__ == "__main__":
    unittest.main()
