"""Tests for the staged large cosmological length tail analysis."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.large_l_tail import (
    LocalFitSettings,
    effective_rates,
    final_cases,
    final_end_u,
    persistent_cosmological_entry,
    price_target,
    screening_cases,
)


class LargeLTailTests(unittest.TestCase):
    def test_local_fits_recover_power_and_exponential_rates(self) -> None:
        times = np.linspace(20.0, 1000.0, 19601)
        power_signal = times**-3
        settings = LocalFitSettings(
            envelope_width=1.0,
            price_window=40.0,
            exponential_scaled_window=0.25,
            floor_multiplier=1.0,
        )
        _, power, _ = effective_rates(times, power_signal, settings, kappa=0.001)
        self.assertAlmostEqual(float(np.nanmedian(power[4000:12000])), 3.0, places=3)

        exponential_signal = np.exp(-0.001 * times)
        _, _, normalized = effective_rates(
            times, exponential_signal, settings, kappa=0.001
        )
        self.assertAlmostEqual(
            float(np.nanmedian(normalized[6000:12000])), 1.0, places=3
        )

    def test_short_screen_can_measure_price_without_cosmological_window(self) -> None:
        times = np.linspace(20.0, 520.0, 10001)
        _, power, normalized = effective_rates(
            times,
            times**-3,
            LocalFitSettings(exponential_scaled_window=0.4),
            kappa=1.0e-4,
        )
        self.assertAlmostEqual(float(np.nanmedian(power)), 3.0, delta=0.005)
        self.assertTrue(np.all(np.isnan(normalized)))

    def test_case_ladders_match_requested_protocol(self) -> None:
        screen = screening_cases(640.0)
        self.assertEqual({case.resolution for case in screen}, {1536, 2048})
        self.assertTrue(all(case.timestep == 0.0025 for case in screen))
        self.assertEqual(
            screening_cases(640.0)[0].name,
            screening_cases(1280.0)[0].name,
        )
        final = final_cases(640.0)
        self.assertEqual(
            {case.resolution for case in final if case.timestep == 0.0025},
            {1536, 2048, 3072},
        )
        self.assertEqual(
            [case.resolution for case in final if case.timestep == 0.00125],
            [2048, 2048],
        )
        self.assertGreaterEqual(final_end_u(640.0), 2560.0)
        final_1280 = final_cases(1280.0)
        schwarzschild_640 = next(
            case for case in final if case.background == "schwarzschild"
        )
        schwarzschild_1280 = next(
            case for case in final_1280 if case.background == "schwarzschild"
        )
        self.assertNotEqual(schwarzschild_640.name, schwarzschild_1280.name)

    def test_price_targets_distinguish_fixed_and_asymptotic_observers(self) -> None:
        self.assertEqual(price_target(0), 5.0)
        self.assertEqual(price_target(1), 5.0)
        self.assertEqual(price_target(2), 3.0)

    def test_persistent_entry_requires_the_final_resolved_run(self) -> None:
        times = np.linspace(0.0, 1000.0, 1001)
        rate = np.full_like(times, 2.0)
        rate[300:401] = 1.0
        rate[600:901] = 1.0
        rate[901:] = np.nan
        entry = persistent_cosmological_entry(
            times, rate, 200.0, tolerance=0.05, kappa=0.002
        )
        self.assertEqual(entry, 600.0)


if __name__ == "__main__":
    unittest.main()
