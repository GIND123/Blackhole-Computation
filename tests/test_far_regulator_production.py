"""Safety and resolution tests for the far-transition production runner."""

import unittest

from black_hole.far_regulator_production import (
    END_TIME,
    LENGTHS,
    RESOLUTIONS,
    archive_path,
    case_catalogue,
    physical_contract,
    production_numerical,
    spectral_preflight,
)
from black_hole.regulator_suite import LEVELS, flat_numerical


class FarRegulatorProductionTests(unittest.TestCase):
    def test_catalogue_has_three_isolated_cases_per_length(self) -> None:
        catalogue = case_catalogue()
        self.assertEqual(len(catalogue), len(LENGTHS) * len(LEVELS))
        self.assertTrue(all(name.startswith("far_sds_L") for name in catalogue))
        self.assertFalse(any("schwarzschild" in name for name in catalogue))

    def test_endpoint_matches_frozen_flat_controls(self) -> None:
        self.assertEqual(END_TIME, 200.0)
        for level in LEVELS:
            self.assertEqual(flat_numerical(level).end_time, END_TIME)
        for length in LENGTHS:
            for level in LEVELS:
                self.assertEqual(production_numerical(length, level).end_time, END_TIME)

    def test_timesteps_match_frozen_flat_controls(self) -> None:
        for length in LENGTHS:
            for level in LEVELS:
                self.assertEqual(
                    production_numerical(length, level).timestep,
                    flat_numerical(level).timestep,
                )

    def test_resolution_ladder_increases_with_length(self) -> None:
        for length in LENGTHS:
            values = [RESOLUTIONS[length][level] for level in LEVELS]
            self.assertEqual(values, sorted(values))
            self.assertEqual(len(values), len(set(values)))
        self.assertEqual(RESOLUTIONS[640]["coarse"], 1792)

    def test_every_spectral_preflight_passes(self) -> None:
        for length in LENGTHS:
            for level in LEVELS:
                with self.subTest(length=length, level=level):
                    audit = spectral_preflight(length, level)
                    self.assertTrue(audit["passed"])
                    self.assertGreaterEqual(audit["transition_nodes"], 12)
                    self.assertGreaterEqual(audit["outer_cap_nodes"], 12)
                    self.assertGreater(audit["represented_minimum_Q"], 0.0)

    def test_contract_records_raw_background_independent_headline(self) -> None:
        contract = physical_contract()
        self.assertIn("raw unshifted", contract["headline_observable"])
        self.assertFalse(contract["background_transfer_correction_used_in_headline"])
        self.assertFalse(contract["control_archives_modified"])

    def test_archive_paths_are_separate_from_controls(self) -> None:
        path = archive_path("new", 320, "fine")
        self.assertEqual(
            path.as_posix(), "new/raw/exterior/L320/fine/sds_L320.npz"
        )
        with self.assertRaises(ValueError):
            archive_path("new", 40, "fine")


if __name__ == "__main__":
    unittest.main()
