"""Tests for local pulse estimators and uncertainty accounting."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.caustic_analysis import (
    analytic_signal_estimate,
    estimate_pulse,
    matched_template_estimate,
)


class LocalEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.arange(0.0, 30.0, 0.02)
        x = self.times - 15.0
        self.reference = np.exp(-(x / 1.8) ** 2) * np.cos(2.3 * x)

    def test_tapered_analytic_signal_finds_pulse_center_with_local_background(self) -> None:
        trace = self.reference + 0.07 - 0.003 * (self.times - 15.0)
        measured = analytic_signal_estimate(self.times, trace, (8.0, 22.0))
        self.assertAlmostEqual(measured["time"], 15.0, delta=0.03)
        self.assertAlmostEqual(measured["amplitude"], 1.0, delta=0.03)
        self.assertGreater(measured["integrated_flux_energy"], 0.0)

    def test_matched_fit_recovers_shift_amplitude_phase_and_background(self) -> None:
        shift = 0.37
        phase = -0.42
        x = self.times - shift - 15.0
        candidate = (
            1.7
            * np.exp(-(x / 1.8) ** 2)
            * np.cos(2.3 * x + phase)
            + 0.12
            + 0.004 * (self.times - 15.0)
        )
        measured = matched_template_estimate(
            self.times,
            candidate,
            self.times,
            self.reference,
            (8.0, 22.0),
        )
        self.assertAlmostEqual(measured["time_shift"], shift, delta=0.02)
        self.assertAlmostEqual(measured["amplitude"], 1.7, delta=0.04)
        self.assertAlmostEqual(measured["phase"], phase, delta=0.04)

    def test_reported_timing_uncertainty_includes_half_cadence(self) -> None:
        measured = estimate_pulse(
            pulse=0,
            phi=0.0,
            times=self.times,
            trace=self.reference,
            reference_times=self.times,
            reference_trace=self.reference,
            window=(8.0, 22.0),
        )
        self.assertGreaterEqual(
            measured.timing_uncertainty, measured.cadence_uncertainty
        )
        self.assertAlmostEqual(measured.cadence_uncertainty, 0.01)
        self.assertEqual(measured.time, measured.analytic_time)
        self.assertEqual(measured.amplitude, measured.analytic_amplitude)

    def test_lag_at_allowed_boundary_is_unresolved(self) -> None:
        shifted = np.interp(
            self.times - 4.0,
            self.times,
            self.reference,
            left=0.0,
            right=0.0,
        )
        measured = matched_template_estimate(
            self.times,
            shifted,
            self.times,
            self.reference,
            (8.0, 22.0),
            maximum_shift=2.0,
        )
        self.assertTrue(measured["lag_at_boundary"])
        self.assertFalse(measured["resolved"])


if __name__ == "__main__":
    unittest.main()
