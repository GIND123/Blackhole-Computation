"""Tests for the raw far-transition production analysis contract."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from black_hole.far_regulator_production import CONTROL_ROOT, LENGTHS, RESOLUTIONS
from black_hole.far_regulator_production_analysis import (
    ANALYSIS_CONTRACT,
    ANALYSIS_WINDOWS,
    conservative_refinement_scale,
    improvement_with_margins,
    load_uniform_numerical_scales,
    analyze,
    threshold_with_reference_floor,
)
from black_hole.sds_result import SdSSimulationResult


class FarRegulatorProductionAnalysisTests(unittest.TestCase):
    @staticmethod
    def _result(times: np.ndarray, outer: np.ndarray, case: str) -> SdSSimulationResult:
        return SdSSimulationResult(
            rho=np.linspace(0.0, 1.0, 32),
            areal_radius=np.linspace(2.0, 20.0, 32),
            signal_times=times,
            observer_rho=np.array([0.0, 1.0]),
            observer_areal_radius=np.array([2.0, np.inf]),
            signals=np.column_stack((np.zeros_like(outer), outer)),
            snapshot_times=np.array([0.0, 200.0]),
            u_snapshots=np.zeros((2, 32)),
            constraint_linf=np.array([0.0, 1.0e-9]),
            constraint_l2=np.array([0.0, 1.0e-10]),
            metadata={
                "retarded_time_offset": {"q": 0.0},
                "spectral_preflight": {
                    "passed": True,
                    "transition_nodes": 20,
                    "outer_cap_nodes": 20,
                    "maximum_error_over_analytic_minimum": 0.01,
                },
                "simulation_provenance": {
                    "case": case,
                    "physical_contract_sha256": "test",
                },
            },
        )

    def test_contract_allows_no_waveform_fit_or_background_correction(self) -> None:
        self.assertIn("raw unshifted", ANALYSIS_CONTRACT["headline_observable"])
        self.assertFalse(ANALYSIS_CONTRACT["time_translation_fitted"])
        self.assertFalse(ANALYSIS_CONTRACT["amplitude_rescaling_fitted"])
        self.assertFalse(ANALYSIS_CONTRACT["time_dilation_fitted"])
        self.assertFalse(ANALYSIS_CONTRACT["background_transfer_correction_used"])

    def test_unequal_grid_order_uses_actual_resolutions(self) -> None:
        resolutions = tuple(RESOLUTIONS[320][level] for level in (
            "coarse", "medium", "fine"
        ))
        order = 4.0
        hc, hm, hf = (1.0 / resolution for resolution in resolutions)
        coarse_medium = hc**order - hm**order
        medium_fine = hm**order - hf**order
        result = conservative_refinement_scale(
            coarse_medium, medium_fine, resolutions
        )
        self.assertEqual(result["refinement_status"], "monotone_three_grid_sequence")
        self.assertAlmostEqual(result["observed_coupled_order"], order, places=9)
        self.assertGreaterEqual(
            result["conservative_numerical_E2"], medium_fine
        )

    def test_nonmonotone_sequence_is_unresolved_and_keeps_larger_change(self) -> None:
        result = conservative_refinement_scale(
            0.002, 0.003, (1792, 2048, 2304)
        )
        self.assertFalse(result["successive_changes_decrease"])
        self.assertEqual(
            result["refinement_status"], "unresolved_three_grid_sequence"
        )
        self.assertEqual(result["conservative_numerical_E2"], 0.003)
        self.assertTrue(np.isnan(result["richardson_fine_E2"]))

    def test_resolved_improvement_uses_disjoint_family_margins(self) -> None:
        resolved = improvement_with_margins(0.02, 0.001, 0.04, 0.002)
        self.assertTrue(resolved["resolved_improvement_with_numerical_margins"])
        self.assertTrue(resolved["resolved_reduction_at_least_25_percent"])

        unresolved = improvement_with_margins(0.038, 0.002, 0.04, 0.002)
        self.assertFalse(unresolved["resolved_improvement_with_numerical_margins"])
        self.assertFalse(unresolved["resolved_reduction_at_least_25_percent"])

    def test_direct_threshold_includes_schwarzschild_reference_floor(self) -> None:
        without_floor = threshold_with_reference_floor(0.009, 0.0005, 0.0, 0.01)
        with_floor = threshold_with_reference_floor(0.009, 0.0005, 0.0006, 0.01)
        self.assertTrue(without_floor["attained_with_numerical_margin"])
        self.assertFalse(with_floor["attained_with_numerical_margin"])

    def test_frozen_uniform_table_covers_every_requested_window(self) -> None:
        root = Path(CONTROL_ROOT)
        if not root.exists():
            self.skipTest("Frozen v3 control package is not available.")
        scales = load_uniform_numerical_scales(root, LENGTHS)
        self.assertEqual(len(scales), len(LENGTHS) * len(ANALYSIS_WINDOWS))
        self.assertTrue(all(value >= 0.0 for value in scales.values()))

    def test_analysis_keeps_candidate_ladder_separate_from_controls(self) -> None:
        times = np.linspace(0.0, 200.0, 2001)
        reference = 0.2 + np.sin(0.25 * times) * np.exp(-times / 90.0)
        perturbation = np.cos(0.17 * times) * np.exp(-times / 120.0)
        schwarzschild = {
            level: self._result(times, reference, f"schwarzschild_{level}")
            for level in ("coarse", "medium", "fine")
        }
        uniform = {
            level: self._result(times, reference + 0.04 * perturbation, f"uniform_{level}")
            for level in ("coarse", "medium", "fine")
        }
        candidate_amplitudes = {"coarse": 0.03, "medium": 0.02, "fine": 0.01}
        candidate = {
            level: self._result(
                times, reference + amplitude * perturbation, f"candidate_{level}"
            )
            for level, amplitude in candidate_amplitudes.items()
        }
        controls = {
            level: {
                "schwarzschild": schwarzschild[level],
                "uniform_sds": {80: uniform[level]},
            }
            for level in ("coarse", "medium", "fine")
        }
        candidates = {
            level: {80: candidate[level]}
            for level in ("coarse", "medium", "fine")
        }
        frozen_scales = {
            (80, window): 0.001 for window, _, _ in ANALYSIS_WINDOWS
        }

        with (
            patch(
                "black_hole.far_regulator_production_analysis.load_controls",
                return_value=controls,
            ),
            patch(
                "black_hole.far_regulator_production_analysis.load_candidates",
                return_value=candidates,
            ),
            patch(
                "black_hole.far_regulator_production_analysis.validate_archives"
            ),
            patch(
                "black_hole.far_regulator_production_analysis.load_uniform_numerical_scales",
                return_value=frozen_scales,
            ),
        ):
            result = analyze(Path("unused"), Path("unused"), (80,))

        self.assertEqual(len(result["direct"]), 3 * len(ANALYSIS_WINDOWS))
        self.assertEqual(len(result["numerical"]), len(ANALYSIS_WINDOWS))
        self.assertEqual(len(result["comparisons"]), len(ANALYSIS_WINDOWS))
        headline = next(
            row for row in result["comparisons"] if row["window"] == "radiative_signal"
        )
        self.assertTrue(headline["raw_unshifted"])
        self.assertTrue(headline["resolved_improvement_with_numerical_margins"])
        self.assertGreater(headline["uniform_sds_E2"], headline["exterior_sds_E2"])


if __name__ == "__main__":
    unittest.main()
