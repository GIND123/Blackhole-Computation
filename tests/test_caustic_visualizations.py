"""Unit checks for modal reconstruction used by the visualization suite."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from black_hole.caustic_study import direction_waveform
from black_hole.caustic_visualizations import (
    angular_field,
    field_on_sphere,
    measured_pulse_times,
)
from black_hole.source_evolution import load_sourced_result


ARCHIVE = Path("results/regulator_production_v3/raw/source/fine/sds_L80.npz")


@unittest.skipUnless(ARCHIVE.exists(), "final localized-source archive is absent")
class CausticVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = load_sourced_result(ARCHIVE)

    def test_addition_theorem_reconstruction_matches_modal_path(self) -> None:
        sample = 50000
        observer = self.result.outer_index()
        response = self.result.response_signals[sample, observer]
        for phi in (0.0, np.pi / 3.0, np.pi):
            reconstructed = angular_field(
                self.result, response, np.asarray(np.pi / 2.0), np.asarray(phi)
            )
            _, established = direction_waveform(self.result, phi, observer)
            self.assertAlmostEqual(float(reconstructed), float(established[sample]), places=11)

    def test_measured_times_lie_in_declared_windows(self) -> None:
        measured = measured_pulse_times(self.result)
        for value, (start, end) in zip(measured, ((18, 35), (35, 53))):
            self.assertLessEqual(start, value)
            self.assertLessEqual(value, end)

    def test_common_time_interpolation_uses_modal_bracketing_values(self) -> None:
        left = 50000
        time = float(
            0.5
            * (
                self.result.retarded_time[left]
                + self.result.retarded_time[left + 1]
            )
        )
        actual, field = field_on_sphere(
            self.result,
            time,
            np.asarray(np.pi / 2.0),
            np.asarray(0.3),
            interpolate_time=True,
        )
        response = 0.5 * (
            self.result.response_signals[left, self.result.outer_index()]
            + self.result.response_signals[left + 1, self.result.outer_index()]
        )
        expected = angular_field(
            self.result,
            response,
            np.asarray(np.pi / 2.0),
            np.asarray(0.3),
        )
        self.assertEqual(actual, time)
        self.assertAlmostEqual(float(field), float(expected), places=13)


if __name__ == "__main__":
    unittest.main()
