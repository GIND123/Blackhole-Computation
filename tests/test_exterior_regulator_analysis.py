"""Tests for conservative exterior-versus-uniform comparison rules."""

import unittest

from black_hole.exterior_regulator_analysis import (
    conservative_refinement_scale,
    improvement_with_margins,
    leading_transfer_coefficient,
    schwarzschild_potential_integral,
)


class ExteriorRegulatorAnalysisTests(unittest.TestCase):
    def test_fit_free_transfer_coefficients_follow_background_integrals(self) -> None:
        self.assertAlmostEqual(schwarzschild_potential_integral(1.0, 2), 3.25)
        uniform = leading_transfer_coefficient("uniform_sds", 80.0)
        exterior = leading_transfer_coefficient("exterior_sds", 80.0)
        self.assertAlmostEqual(
            uniform["coefficient_times_cosmological_horizon"],
            4.0428004833,
            places=9,
        )
        self.assertAlmostEqual(
            exterior["coefficient_times_cosmological_horizon"],
            3.5912271716,
            places=9,
        )
        self.assertLess(
            exterior["leading_transfer_coefficient_M_inverse"],
            uniform["leading_transfer_coefficient_M_inverse"],
        )

    def test_refinement_scale_uses_larger_of_observed_and_richardson(self) -> None:
        refinement = conservative_refinement_scale(0.004, 0.001)
        self.assertGreater(refinement["observed_coupled_order"], 0.0)
        self.assertGreaterEqual(refinement["conservative_numerical_E2"], 0.001)
        self.assertGreaterEqual(
            refinement["conservative_numerical_E2"],
            refinement["estimated_fine_numerical_E2"],
        )

    def test_resolved_improvement_requires_disjoint_error_bands(self) -> None:
        resolved = improvement_with_margins(0.02, 0.001, 0.04, 0.002)
        self.assertTrue(resolved["resolved_improvement_with_numerical_margins"])
        self.assertTrue(resolved["resolved_reduction_at_least_25_percent"])

        unresolved = improvement_with_margins(0.038, 0.002, 0.04, 0.002)
        self.assertFalse(unresolved["resolved_improvement_with_numerical_margins"])
        self.assertFalse(unresolved["resolved_reduction_at_least_25_percent"])


if __name__ == "__main__":
    unittest.main()
