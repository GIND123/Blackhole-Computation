"""Configuration and safety tests for the exterior-regulator sequence."""

import unittest

from black_hole.exterior_regulator_suite import (
    EXTERIOR_LENGTHS,
    LEVELS,
    archive_path,
    case_catalogue,
    contract_sha256,
    physical_contract,
)


class ExteriorRegulatorSuiteTests(unittest.TestCase):
    def test_catalogue_contains_only_three_new_exterior_cases(self) -> None:
        catalogue = case_catalogue()
        self.assertEqual(len(catalogue), len(LEVELS) * len(EXTERIOR_LENGTHS))
        self.assertTrue(all(name.startswith("exterior_sds_") for name in catalogue))
        self.assertFalse(any("schwarzschild" in name for name in catalogue))
        self.assertTrue(all("L80" in name for name in catalogue))

    def test_contract_records_far_horizon_transition(self) -> None:
        contract = physical_contract()
        self.assertIn("theta0=2*theta1", contract["transition"]["inner_compact_location"])
        self.assertIn("R1=0.9*r_c", contract["transition"]["outer_radius"])
        self.assertFalse(contract["control_archives_modified"])
        self.assertEqual(contract_sha256(), contract_sha256())

    def test_archives_are_isolated_from_frozen_controls(self) -> None:
        path = archive_path("new", "fine", 80.0)
        self.assertEqual(
            path.as_posix(), "new/raw/exterior/fine/sds_L80.npz"
        )
        with self.assertRaises(ValueError):
            archive_path("new", "fine", 160.0)


if __name__ == "__main__":
    unittest.main()
