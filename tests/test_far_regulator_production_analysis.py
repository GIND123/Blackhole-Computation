"""Tests for the raw fixed-transition-width production analysis contract."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from black_hole.far_regulator_production import CONTROL_ROOT, LENGTHS, RESOLUTIONS
from black_hole.far_regulator_production_analysis import (
    ANALYSIS_CONTRACT,
    ANALYSIS_WINDOWS,
    ARCHIVED_NUMERICAL_WINDOWS,
    QNM_WINDOWS,
    _analysis_window_family,
    _candidate_archive_path,
    compute_uniform_numerical_scales,
    conservative_refinement_scale,
    improvement_with_margins,
    load_controls,
    load_uniform_numerical_scales,
    analyze,
    threshold_with_reference_floor,
)
from black_hole.exterior_sds_model import ExteriorSdSParameters
from black_hole.sds_result import SdSSimulationResult


class FarRegulatorProductionAnalysisTests(unittest.TestCase):
    @staticmethod
    def _result(
        times: np.ndarray,
        outer: np.ndarray,
        case: str,
        resolution: int = 768,
    ) -> SdSSimulationResult:
        model = ExteriorSdSParameters(cosmological_length=80.0).as_dict()
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
                "evolution_stability_audit": {
                    "passed": True,
                    "maximum_stored_solution_amplification": 1.0,
                },
                "simulation_provenance": {
                    "case": case,
                    "physical_contract_sha256": "test",
                },
                "numerical": {"resolution": resolution},
                "model": model,
                "background_audit": {
                    "minimum_A": 0.1,
                    "maximum_A": 0.2,
                    "minimum_P": 1.0,
                    "maximum_abs_P": 8.0,
                },
                "wall_seconds": 1.0,
                "iterations": 10,
            },
        )

    def test_candidate_path_prefers_new_then_unchanged_legacy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "current"
            legacy = root / "legacy"
            legacy_path = legacy / "raw/exterior/L160/fine/sds_L160.npz"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.touch()
            self.assertEqual(
                _candidate_archive_path(current, legacy, 160, "fine"),
                legacy_path,
            )
            current_path = current / "raw/exterior/L160/fine/sds_L160.npz"
            current_path.parent.mkdir(parents=True)
            current_path.touch()
            self.assertEqual(
                _candidate_archive_path(current, legacy, 160, "fine"),
                current_path,
            )
            self.assertEqual(
                _candidate_archive_path(current, legacy, 640, "fine"),
                current / "raw/exterior/L640/fine/sds_L640.npz",
            )

    def test_contract_allows_no_waveform_fit_or_background_correction(self) -> None:
        self.assertIn("raw unshifted", ANALYSIS_CONTRACT["headline_observable"])
        self.assertEqual(ANALYSIS_CONTRACT["qnm_window_sensitivity"], QNM_WINDOWS)
        self.assertFalse(ANALYSIS_CONTRACT["time_translation_fitted"])
        self.assertFalse(ANALYSIS_CONTRACT["amplitude_rescaling_fitted"])
        self.assertFalse(ANALYSIS_CONTRACT["time_dilation_fitted"])
        self.assertFalse(ANALYSIS_CONTRACT["background_transfer_correction_used"])

    def test_qnm_windows_are_fixed_and_classified_as_sensitivity_checks(self) -> None:
        self.assertEqual(
            QNM_WINDOWS,
            (
                ("qnm_early", 10.0, 40.0),
                ("qnm_central", 15.0, 45.0),
                ("qnm_late", 20.0, 50.0),
            ),
        )
        self.assertTrue(
            all(
                _analysis_window_family(window) == "qnm_window_sensitivity"
                for window, _, _ in QNM_WINDOWS
            )
        )
        self.assertEqual(_analysis_window_family("radiative_signal"), "cumulative")
        self.assertEqual(_analysis_window_family("early_ringdown"), "disjoint")

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

    def test_recomputed_legacy_uniform_scales_match_frozen_table(self) -> None:
        root = Path(CONTROL_ROOT)
        if not root.exists():
            self.skipTest("Frozen v3 control package is not available.")
        archived = load_uniform_numerical_scales(root, LENGTHS)
        controls = load_controls(root, LENGTHS)
        fine = controls["fine"]["schwarzschild"]
        reference_times = fine.signal_times - float(
            fine.metadata["retarded_time_offset"]["q"]
        )
        common = (reference_times >= 0.0) & (reference_times <= 80.0)
        times = reference_times[common]
        reference = fine.signals[common, -1]
        recomputed, _ = compute_uniform_numerical_scales(
            controls, times, reference, LENGTHS
        )

        expected_count = len(LENGTHS) * len(ARCHIVED_NUMERICAL_WINDOWS)
        self.assertEqual(len(archived), expected_count)
        for key, archived_scale in archived.items():
            self.assertAlmostEqual(recomputed[key], archived_scale, places=12)

    def test_uniform_qnm_scale_uses_paired_control_residuals(self) -> None:
        times = np.linspace(0.0, 80.0, 1601)
        reference = 0.3 + np.sin(0.3 * times) * np.exp(-times / 70.0)
        perturbation = np.cos(0.2 * times) * np.exp(-times / 100.0)
        levels = ("coarse", "medium", "fine")
        resolutions = {"coarse": 384, "medium": 512, "fine": 768}
        amplitudes = {"coarse": 0.06, "medium": 0.045, "fine": 0.04}
        controls = {
            level: {
                "schwarzschild": self._result(
                    times,
                    reference,
                    f"schwarzschild_{level}",
                    resolutions[level],
                ),
                "uniform_sds": {
                    80: self._result(
                        times,
                        reference + amplitudes[level] * perturbation,
                        f"uniform_{level}",
                        resolutions[level],
                    )
                },
            }
            for level in levels
        }

        scales, rows = compute_uniform_numerical_scales(
            controls, times, reference, (80,)
        )
        self.assertEqual(len(scales), len(ANALYSIS_WINDOWS))
        self.assertEqual(len(rows), len(ANALYSIS_WINDOWS))
        central = next(row for row in rows if row["window"] == "qnm_central")
        self.assertEqual(central["window_family"], "qnm_window_sensitivity")
        self.assertGreater(central["coarse_medium_paired_E2"], 0.0)
        self.assertGreater(central["medium_fine_paired_E2"], 0.0)
        self.assertEqual(
            scales[(80, "qnm_central")],
            central["conservative_numerical_E2"],
        )

    def test_analysis_keeps_candidate_ladder_separate_from_controls(self) -> None:
        times = np.linspace(0.0, 200.0, 2001)
        reference = 0.2 + np.sin(0.25 * times) * np.exp(-times / 90.0)
        perturbation = np.cos(0.17 * times) * np.exp(-times / 120.0)
        schwarzschild = {
            level: self._result(
                times,
                reference,
                f"schwarzschild_{level}",
                RESOLUTIONS[80][level],
            )
            for level in ("coarse", "medium", "fine")
        }
        uniform = {
            level: self._result(
                times,
                reference + 0.04 * perturbation,
                f"uniform_{level}",
                RESOLUTIONS[80][level],
            )
            for level in ("coarse", "medium", "fine")
        }
        candidate_amplitudes = {"coarse": 0.03, "medium": 0.02, "fine": 0.01}
        candidate = {
            level: self._result(
                times,
                reference + amplitude * perturbation,
                f"candidate_{level}",
                RESOLUTIONS[80][level],
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
        ):
            result = analyze(Path("unused"), Path("unused"), (80,))

        self.assertEqual(len(result["direct"]), 3 * len(ANALYSIS_WINDOWS))
        self.assertEqual(len(result["numerical"]), len(ANALYSIS_WINDOWS))
        self.assertEqual(len(result["uniform_numerical"]), len(ANALYSIS_WINDOWS))
        self.assertEqual(len(result["comparisons"]), len(ANALYSIS_WINDOWS))
        headline = next(
            row for row in result["comparisons"] if row["window"] == "qnm_central"
        )
        qnm_rows = {
            row["window"]: row
            for row in result["comparisons"]
            if row["window"].startswith("qnm_")
        }
        self.assertEqual(set(qnm_rows), {window for window, _, _ in QNM_WINDOWS})
        self.assertTrue(
            all(
                row["window_family"] == "qnm_window_sensitivity"
                for row in qnm_rows.values()
            )
        )
        self.assertTrue(headline["raw_unshifted"])
        self.assertTrue(headline["resolved_improvement_with_numerical_margins"])
        self.assertGreater(headline["uniform_sds_E2"], headline["exterior_sds_E2"])


if __name__ == "__main__":
    unittest.main()
