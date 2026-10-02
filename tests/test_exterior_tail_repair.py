"""Contract tests for the staged exterior-tail repair campaign."""

import unittest
from pathlib import Path

from black_hole.exterior_tail_repair import (
    END_U,
    RepairCase,
    archive_path,
    case_catalogue,
    cases,
)
from black_hole.run_exterior_tail_repair_campaign import primary_conformal_cases


class ExteriorTailRepairTests(unittest.TestCase):
    def test_diagnostic_matrix(self) -> None:
        self.assertEqual(END_U, 150.0)
        self.assertEqual(len(cases()), 28)
        self.assertEqual(len(case_catalogue()), 28)
        for coupling in (0.0, 1.0 / 6.0):
            selected = [case for case in cases() if case.curvature_coupling == coupling]
            self.assertEqual(len(selected), 14)

    def test_formulation_properties_and_isolated_paths(self) -> None:
        root = Path("repair-output")
        for case in cases():
            with self.subTest(case=case.name):
                self.assertEqual(
                    case.factored, case.formulation.startswith("factored_")
                )
                self.assertEqual(
                    case.conservative, case.formulation == "conservative"
                )
                self.assertEqual(
                    case.damping == 0.0,
                    case.formulation in {"standard", "conservative"},
                )
                path = archive_path(root, case)
                self.assertTrue(path.as_posix().startswith("repair-output/raw/"))
                self.assertIn(case.coupling_label, path.parts)

    def test_rejects_unsupported_case(self) -> None:
        with self.assertRaises(ValueError):
            RepairCase("factored_gamma0", 1.0 / 6.0, 2048)

    def test_primary_conformal_subset(self) -> None:
        selected = primary_conformal_cases()
        self.assertEqual(len(selected), 9)
        for name in selected:
            case = case_catalogue()[name]
            self.assertEqual(case.curvature_coupling, 1.0 / 6.0)
            self.assertIn(
                case.formulation,
                {"standard", "conservative", "factored_gamma1"},
            )
            self.assertEqual(case.timestep, 0.0025)


if __name__ == "__main__":
    unittest.main()
