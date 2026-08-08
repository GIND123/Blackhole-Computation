"""Configuration and safety tests for the frozen regulator simulations."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from black_hole.regulator_suite import (
    FLAT_LENGTHS,
    LEVELS,
    SOURCE_LENGTHS,
    archive_path,
    case_catalogue,
    contract_sha256,
    localized_source,
    physical_contract,
    _reserve_destination,
    source_numerical,
)


class RegulatorSuiteTests(unittest.TestCase):
    def test_catalogue_contains_three_levels_without_L1280(self) -> None:
        catalogue = case_catalogue()
        expected = len(LEVELS) * (
            1 + len(FLAT_LENGTHS) + 1 + len(SOURCE_LENGTHS)
        )
        self.assertEqual(len(catalogue), expected)
        self.assertTrue(any("L320" in name for name in catalogue))
        self.assertTrue(any("L640" in name for name in catalogue))
        self.assertFalse(any("L1280" in name for name in catalogue))

    def test_fixed_source_is_identical_at_every_source_case(self) -> None:
        source = localized_source()
        self.assertEqual(source.radial_half_width, 0.75)
        self.assertEqual(source.time_half_width, 2.0)
        self.assertEqual(source.angular_concentration, 64.0)
        self.assertEqual(
            contract_sha256("source"), contract_sha256("source")
        )

    def test_contract_records_required_geometric_choices(self) -> None:
        for study in ("flat", "source"):
            contract = physical_contract(study)
            self.assertEqual(contract["gauge"], "minimal")
            self.assertEqual(contract["height_reference_radius_over_M"], 4.0)
            self.assertIn("analytic", contract["retarded_time"])
            self.assertIn("rho=", contract["finite_L_compactification"])

    def test_source_refinement_is_strict(self) -> None:
        levels = [source_numerical(level) for level in LEVELS]
        self.assertEqual(
            [value.radial_resolution for value in levels], [1024, 1536, 2048]
        )
        self.assertEqual(
            [value.angular_ell_max for value in levels], [42, 46, 50]
        )
        self.assertTrue(
            levels[0].timestep > levels[1].timestep > levels[2].timestep
        )

    def test_archive_paths_separate_studies_and_levels(self) -> None:
        path = archive_path("out", "source", "fine", 320.0)
        self.assertEqual(path.as_posix(), "out/raw/source/fine/sds_L320.npz")

    def test_destination_reservation_is_atomic(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "case.npz"
            reservation = _reserve_destination(destination, "case")
            self.assertEqual(reservation.read_text(encoding="utf-8"), "case\n")
            with self.assertRaises(FileExistsError):
                _reserve_destination(destination, "case")


if __name__ == "__main__":
    unittest.main()
