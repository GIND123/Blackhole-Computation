"""Tests for tail fitting and trust-time diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.tail_analysis import (
    aligned_signals,
    asymptotic_constant,
    fit_exponential,
    fit_power_law,
    sliding_window_difference,
    select_stable_exponential_fit,
    trust_times,
)


class TailAnalysisTests(unittest.TestCase):
    def test_power_fit_recovers_exponent(self) -> None:
        times = np.linspace(10.0, 1000.0, 5000)
        signal = 2.5 * times**-4.0
        fit = fit_power_law(times, signal, 30.0, 800.0)
        self.assertAlmostEqual(fit.rate, 4.0, places=11)
        self.assertGreater(fit.r_squared, 1.0 - 1e-12)

    def test_exponential_fit_recovers_rate_and_offset(self) -> None:
        times = np.linspace(0.0, 200.0, 4001)
        offset = -0.03
        signal = offset + 1.2 * np.exp(-0.07 * times)
        fit = fit_exponential(times, signal, 10.0, 170.0, offset=offset)
        self.assertAlmostEqual(fit.rate, 0.07, places=11)

    def test_stable_exponential_selector_recovers_rate(self) -> None:
        times = np.linspace(0.0, 500.0, 10001)
        kappa = 0.01
        signal = 0.7 * np.exp(-1.1 * kappa * times)
        fit = select_stable_exponential_fit(times, signal, kappa)
        self.assertAlmostEqual(fit.rate / kappa, 1.1, places=10)

    def test_asymptotic_constant_reports_small_drift(self) -> None:
        times = np.linspace(0.0, 100.0, 1001)
        signal = 0.2 + np.exp(-0.2 * times)
        estimate = asymptotic_constant(times, signal)
        self.assertAlmostEqual(float(estimate["value"]), 0.2, places=7)
        self.assertLess(float(estimate["relative_half_interval_drift"]), 1e-6)

    def test_alignment_uses_common_interval(self) -> None:
        first_times = np.linspace(0.0, 10.0, 101)
        second_times = np.linspace(2.0, 12.0, 101)
        times, first, second = aligned_signals(
            first_times,
            first_times**2,
            second_times,
            second_times**2,
        )
        self.assertAlmostEqual(times[0], 2.0)
        self.assertLessEqual(times[-1], 10.0)
        np.testing.assert_allclose(first, second, rtol=1e-13, atol=1e-13)

    def test_sliding_difference_and_sustained_crossing(self) -> None:
        times = np.linspace(0.0, 200.0, 4001)
        reference = np.sin(0.2 * times) + 2.0
        relative_error = np.where(times < 80.0, 0.0, 0.08)
        candidate = reference * (1.0 + relative_error)
        out_times, difference = sliding_window_difference(
            times, reference, candidate, window_width=10.0
        )
        crossings = trust_times(out_times, difference, sustained_width=2.0)
        self.assertGreater(crossings[0.05], 70.0)
        self.assertLess(crossings[0.05], 85.0)
        self.assertTrue(np.isnan(crossings[0.10]))

    def test_trust_time_requires_prior_agreement(self) -> None:
        times = np.linspace(0.0, 100.0, 1001)
        difference = np.full_like(times, 0.2)
        crossings = trust_times(times, difference, start_time=10.0)
        self.assertTrue(all(np.isnan(value) for value in crossings.values()))


if __name__ == "__main__":
    unittest.main()
