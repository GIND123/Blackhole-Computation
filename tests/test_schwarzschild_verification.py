"""Checks for the standalone archived Schwarzschild verification report."""

from __future__ import annotations

from pathlib import Path
import unittest

from black_hole.schwarzschild_verification import (
    angular_truncation_l2,
    sphere_time_relative_l2,
)
from black_hole.source_evolution import load_sourced_result


ROOT = Path("results/regulator_production_v3/raw/source")


@unittest.skipUnless(ROOT.exists(), "regulator source archives are absent")
class SchwarzschildVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coarse = load_sourced_result(ROOT / "coarse/schwarzschild.npz")
        cls.medium = load_sourced_result(ROOT / "medium/schwarzschild.npz")
        cls.fine = load_sourced_result(ROOT / "fine/schwarzschild.npz")

    def test_combined_refinement_improves(self) -> None:
        coarse = sphere_time_relative_l2(self.coarse, self.fine)
        medium = sphere_time_relative_l2(self.medium, self.fine)
        self.assertGreater(coarse, medium)
        self.assertLess(medium, 1e-4)

    def test_angular_omission_decreases_with_cutoff(self) -> None:
        self.assertGreater(
            angular_truncation_l2(self.fine, 42),
            angular_truncation_l2(self.fine, 46),
        )


if __name__ == "__main__":
    unittest.main()
