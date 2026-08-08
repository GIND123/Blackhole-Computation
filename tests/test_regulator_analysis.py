"""Unit tests for regulator metrics and exclusion rules."""

import unittest
from pathlib import Path

import numpy as np

from black_hole.regulator_analysis import (
    _effective_order,
    l12_phase_cleanup,
    waveform_metrics,
)


class RegulatorAnalysisTests(unittest.TestCase):
    def test_waveform_metric_recovers_complex_scale_without_time_shift(self) -> None:
        times = np.linspace(0.0, 40.0, 4001)
        reference = np.exp(-0.04 * times) * np.cos(1.7 * times)
        candidate = 1.02 * np.exp(-0.04 * times) * np.cos(1.7 * times + 0.03)
        measured = waveform_metrics(times, candidate, reference, 0.0, 40.0)
        self.assertAlmostEqual(measured["overlap_amplitude_ratio"], 1.02, delta=2e-3)
        self.assertAlmostEqual(abs(measured["phase_difference_radians"]), 0.03, delta=2e-3)
        self.assertFalse(measured["time_translation_fitted"])

    def test_effective_order_handles_unequal_resolution_ratios(self) -> None:
        resolutions = (384, 512, 768)
        order = 4.0
        h = [1.0 / value for value in resolutions]
        coarse_medium = h[0] ** order - h[1] ** order
        medium_fine = h[1] ** order - h[2] ** order
        measured = _effective_order(
            coarse_medium, medium_fine, *resolutions
        )
        self.assertAlmostEqual(measured, order, places=10)

    def test_every_L12_phase_pair_is_excluded(self) -> None:
        rows = l12_phase_cleanup(Path.cwd())
        self.assertEqual(len(rows), 6)
        self.assertTrue(
            all(not row["included_in_quantitative_phase_analysis"] for row in rows)
        )
        self.assertTrue(all(np.isnan(row["corrected_phase_radians"]) for row in rows))
        self.assertTrue(all(not row["pulse_arrivals_consistent"] for row in rows))


if __name__ == "__main__":
    unittest.main()
