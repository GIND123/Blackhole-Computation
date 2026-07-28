"""Tests for the transition-interval crossover criterion."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.crossover_final import (
    EnvelopeSettings,
    SweepGrid,
    SweepSummary,
    envelope_rate,
    transition_interval,
)


def synthetic_crossover(
    ell: int = 1,
    kappa: float = 0.0125,
    power: float = 5.0,
    switch: float = 220.0,
    end: float = 420.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a signal that changes from a power tail to an exponential tail.

    The amplitude is continuous at ``switch``, so the transition is carried by
    the local rate rather than by a jump in amplitude.  The carrier
    oscillation is fast compared with the envelope width, as it is for the
    quasinormal ringing of the evolved waveforms.
    """

    times = np.arange(20.0, end, 0.05)
    amplitude = np.where(
        times < switch,
        times**-power,
        switch**-power * np.exp(-ell * kappa * (times - switch)),
    )
    return times, amplitude * np.cos(6.0 * times)


class EnvelopeRateTests(unittest.TestCase):
    def test_recovers_a_pure_exponential_rate(self) -> None:
        times = np.arange(0.0, 400.0, 0.05)
        signal = np.exp(-0.02 * times) * np.cos(6.0 * times)
        rate, envelope = envelope_rate(times, signal, EnvelopeSettings(30.0, 0.5))
        interior = (times > 120.0) & (times < 280.0)
        self.assertTrue(np.all(np.isfinite(rate[interior])))
        self.assertTrue(np.allclose(rate[interior], 0.02, atol=1e-3))
        self.assertTrue(np.all(envelope[interior] > 0.0))

    def test_recovers_a_pure_power_index(self) -> None:
        times = np.arange(10.0, 900.0, 0.05)
        signal = times**-3.0 * np.cos(6.0 * times)
        rate, _ = envelope_rate(times, signal, EnvelopeSettings(30.0, 0.5))
        interior = (times > 300.0) & (times < 700.0)
        index = rate[interior] * times[interior]
        # The residual carrier ripple survives differentiation and is amplified
        # by the factor U of the index, so the scatter, not each sample, is the
        # meaningful bound.
        self.assertAlmostEqual(float(np.median(index)), 3.0, places=2)
        self.assertLess(float(np.max(np.abs(index - 3.0))), 0.3)

    def test_endpoints_and_the_amplitude_floor_are_masked(self) -> None:
        times = np.arange(0.0, 200.0, 0.05)
        signal = np.exp(-0.5 * times)
        rate, _ = envelope_rate(times, signal, EnvelopeSettings(20.0, 0.5))
        self.assertTrue(np.isnan(rate[0]))
        self.assertTrue(np.isnan(rate[-1]))
        self.assertTrue(np.isnan(rate[-len(rate) // 4]))

    def test_short_or_over_filtered_series_are_rejected(self) -> None:
        times = np.arange(0.0, 0.5, 0.05)
        with self.assertRaises(ValueError):
            envelope_rate(times, np.ones_like(times), EnvelopeSettings(30.0, 0.5))
        times = np.arange(0.0, 40.0, 0.05)
        with self.assertRaises(ValueError):
            envelope_rate(times, np.ones_like(times), EnvelopeSettings(400.0, 0.5))


class TransitionIntervalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kappa = 0.0125
        self.ell = 1
        times, signal = synthetic_crossover(kappa=self.kappa, switch=220.0)
        reference_amplitude = times**-5.0 * np.cos(6.0 * times)
        settings = EnvelopeSettings(30.0, 0.5)
        rate, _ = envelope_rate(times, signal, settings)
        reference_rate, _ = envelope_rate(times, reference_amplitude, settings)
        self.scaled = self.kappa * times
        self.normalized = rate / self.kappa
        self.reference = reference_rate / self.kappa

    def test_brackets_a_known_transition(self) -> None:
        interval = transition_interval(
            self.scaled,
            self.normalized,
            self.reference,
            self.ell,
            tolerance=0.10,
            persistence=0.25,
        )
        self.assertEqual(interval.status, "resolved")
        self.assertLess(interval.departure, self.kappa * 220.0)
        self.assertGreater(interval.entry, self.kappa * 220.0)
        self.assertAlmostEqual(interval.final_rate, 1.0, places=2)

    def test_departure_precedes_entry_for_every_tolerance(self) -> None:
        for tolerance in (0.05, 0.10, 0.20):
            interval = transition_interval(
                self.scaled,
                self.normalized,
                self.reference,
                self.ell,
                tolerance=tolerance,
                persistence=0.25,
            )
            self.assertEqual(interval.status, "resolved")
            self.assertLess(interval.departure, interval.entry)

    def test_a_tighter_tolerance_cannot_delay_the_entry(self) -> None:
        entries = [
            transition_interval(
                self.scaled,
                self.normalized,
                self.reference,
                self.ell,
                tolerance=tolerance,
                persistence=0.25,
            ).entry
            for tolerance in (0.05, 0.10, 0.20)
        ]
        self.assertTrue(entries[0] >= entries[1] >= entries[2])

    def test_a_signal_that_never_reaches_the_target_is_unresolved(self) -> None:
        # A power tail alone reaches gamma/kappa_c=1 only at kappa_c U=5, so the
        # window is stopped before the reference curves would meet by accident.
        times, _ = synthetic_crossover(end=300.0)
        signal = times**-5.0 * np.cos(6.0 * times)
        rate, _ = envelope_rate(times, signal, EnvelopeSettings(30.0, 0.5))
        interval = transition_interval(
            self.kappa * times,
            rate / self.kappa,
            rate / self.kappa,
            self.ell,
            tolerance=0.10,
            persistence=0.25,
        )
        self.assertEqual(interval.status, "no_cosmological_entry")
        self.assertTrue(np.isnan(interval.entry))

    def test_a_transient_visit_to_the_target_is_rejected(self) -> None:
        times = np.arange(20.0, 420.0, 0.05)
        rate = np.full_like(times, 3.0)
        rate[(times > 150.0) & (times < 190.0)] = 1.0
        interval = transition_interval(
            self.kappa * times,
            rate,
            np.full_like(times, 3.0),
            self.ell,
            tolerance=0.10,
            persistence=0.25,
        )
        self.assertEqual(interval.status, "no_cosmological_entry")

    def test_a_missing_reference_leaves_the_case_unresolved(self) -> None:
        interval = transition_interval(
            self.scaled,
            self.normalized,
            np.full_like(self.reference, np.nan),
            self.ell,
            tolerance=0.10,
            persistence=0.25,
        )
        self.assertEqual(interval.status, "no_schwarzschild_agreement")
        self.assertTrue(np.isfinite(interval.entry))

    def test_malformed_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            transition_interval(
                self.scaled[::-1],
                self.normalized,
                self.reference,
                self.ell,
            )
        with self.assertRaises(ValueError):
            transition_interval(
                self.scaled,
                self.normalized,
                self.reference,
                0,
            )
        with self.assertRaises(ValueError):
            transition_interval(
                self.scaled,
                self.normalized,
                self.reference,
                self.ell,
                tolerance=0.0,
            )


class SweepSummaryTests(unittest.TestCase):
    def test_grid_size_matches_the_enumerated_configurations(self) -> None:
        grid = SweepGrid()
        self.assertEqual(len(list(grid.configurations())), grid.size)

    def test_status_reports_the_dominant_failure(self) -> None:
        summary = SweepSummary(
            ell=1,
            length=20.0,
            observer=8.0,
            kappa=0.044,
            configurations=10,
            resolved=0,
            statuses={"no_cosmological_entry": 9, "no_schwarzschild_agreement": 1},
        )
        self.assertEqual(summary.status, "no_cosmological_entry")
        row = summary.as_row()
        self.assertTrue(np.isnan(row["kappa_c_U_entry_median"]))
        self.assertEqual(row["observer"], "r/M=8")

    def test_ranges_are_reported_in_both_time_variables(self) -> None:
        summary = SweepSummary(
            ell=1,
            length=80.0,
            observer=None,
            kappa=0.0125,
            configurations=4,
            resolved=4,
            departures=[1.0, 1.2, 1.4, 1.6],
            entries=[2.6, 2.8, 3.0, 3.2],
            statuses={"resolved": 4},
        )
        row = summary.as_row()
        self.assertEqual(row["status"], "resolved")
        self.assertAlmostEqual(row["kappa_c_U_entry_median"], 2.9)
        self.assertAlmostEqual(row["U_over_M_entry_median"], 2.9 / 0.0125)
        self.assertAlmostEqual(row["kappa_c_U_departure_minimum"], 1.0)
        self.assertEqual(row["observer"], "cosmological_horizon")


if __name__ == "__main__":
    unittest.main()
