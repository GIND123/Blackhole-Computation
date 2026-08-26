"""Safety and resolution tests for the width-floor production runner."""

import unittest

import numpy as np

from black_hole.far_regulator_production import (
    END_TIME,
    LENGTHS,
    RESOLUTIONS,
    archive_path,
    case_catalogue,
    evolution_stability_audit,
    physical_contract,
    production_numerical,
    spectral_preflight,
)
from black_hole.regulator_suite import LEVELS, flat_numerical
from black_hole.sds_result import SdSSimulationResult


class FarRegulatorProductionTests(unittest.TestCase):
    @staticmethod
    def _result(amplitude: float, constraint: float) -> SdSSimulationResult:
        return SdSSimulationResult(
            rho=np.linspace(0.0, 1.0, 8),
            areal_radius=np.linspace(2.0, 20.0, 8),
            signal_times=np.array([0.0, 1.0]),
            observer_rho=np.array([1.0]),
            observer_areal_radius=np.array([np.inf]),
            signals=np.array([[1.0], [amplitude]]),
            snapshot_times=np.array([0.0, 1.0]),
            u_snapshots=np.vstack((np.ones(8), amplitude * np.ones(8))),
            constraint_linf=np.array([0.0, constraint]),
            constraint_l2=np.array([0.0, constraint]),
            metadata={},
        )

    def test_catalogue_has_three_isolated_cases_per_length(self) -> None:
        catalogue = case_catalogue()
        self.assertEqual(len(catalogue), len(LENGTHS) * len(LEVELS))
        self.assertTrue(
            all(name.startswith("width_floor_sds_L") for name in catalogue)
        )
        self.assertFalse(any("schwarzschild" in name for name in catalogue))

    def test_qnm_endpoint_is_shorter_than_frozen_flat_controls(self) -> None:
        self.assertEqual(END_TIME, 100.0)
        for level in LEVELS:
            self.assertGreaterEqual(flat_numerical(level).end_time, END_TIME)
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
        self.assertEqual(RESOLUTIONS[640]["coarse"], 768)

    def test_every_spectral_preflight_passes(self) -> None:
        for length in LENGTHS:
            for level in LEVELS:
                with self.subTest(length=length, level=level):
                    audit = spectral_preflight(length, level)
                    self.assertTrue(audit["passed"])
                    self.assertGreaterEqual(audit["transition_nodes"], 12)
                    self.assertGreaterEqual(audit["outer_cap_nodes"], 12)
                    self.assertGreater(audit["represented_minimum_Q"], 0.0)
                    self.assertTrue(audit["transition_branch_verified"])
                    self.assertGreater(audit["transition_rho_width"], 0.0)
                    self.assertGreater(audit["outer_cap_rho_width"], 0.0)
        floored = spectral_preflight(640, "fine")
        reference = spectral_preflight(160, "fine")
        self.assertGreaterEqual(
            floored["transition_nodes"], reference["transition_nodes"]
        )
        self.assertTrue(floored["transition_width_floor_active"])
        self.assertEqual(floored["transition_nodes"], 37)
        self.assertEqual(floored["outer_cap_nodes"], 37)
        self.assertLess(
            floored["maximum_error_over_analytic_minimum"], 1.0e-3
        )

    def test_contract_records_raw_background_independent_headline(self) -> None:
        contract = physical_contract()
        self.assertIn("raw unshifted", contract["headline_observable"])
        self.assertFalse(contract["background_transfer_correction_used_in_headline"])
        self.assertFalse(contract["control_archives_modified"])
        self.assertEqual(
            contract["evolution_stability_acceptance"][
                "maximum_stored_solution_amplification"
            ],
            10.0,
        )

    def test_stability_audit_rejects_finite_blow_up(self) -> None:
        stable = evolution_stability_audit(self._result(1.1, 1.0e-3))
        unstable = evolution_stability_audit(self._result(1.0e20, 1.0e15))
        self.assertTrue(stable["passed"])
        self.assertFalse(unstable["passed"])
        self.assertGreater(
            unstable["maximum_stored_solution_amplification"], 10.0
        )

    def test_archive_paths_are_separate_from_controls(self) -> None:
        path = archive_path("new", 320, "fine")
        self.assertEqual(
            path.as_posix(), "new/raw/exterior/L320/fine/sds_L320.npz"
        )
        with self.assertRaises(ValueError):
            archive_path("new", 40, "fine")


if __name__ == "__main__":
    unittest.main()
