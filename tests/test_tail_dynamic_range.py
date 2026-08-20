"""High dynamic range regression tests for the tail rate estimators.

A tail waveform carries a prompt pulse of order unity followed by a decay
that reaches the double-precision amplitude floor.  Squaring the signal
doubles that ratio in the exponent, so any estimator that differences one
global cumulative sum loses the tail entirely.  These tests drive the
estimators with signals whose decay laws are known in closed form and
require the measured indices to come back.
"""

from __future__ import annotations

import numpy as np
import pytest

from black_hole.large_l_tail import (
    LocalFitSettings,
    _centered_sum,
    effective_rates,
    local_log_fit_rate,
    rms_envelope,
)


CARRIER = 2.0 * np.pi / 2.0  # one oscillation every 2M, well inside the window


def _prompt(times: np.ndarray) -> np.ndarray:
    """Return an order unity prompt pulse that sets the dynamic range."""

    return np.exp(-((times - 20.0) ** 2) / (2.0 * 2.0**2))


def _power_law_case(index: float) -> tuple[np.ndarray, np.ndarray, float]:
    times = np.arange(0.0, 1400.0, 0.05)
    start = 200.0
    envelope = np.where(
        times >= start, 1.0e-3 * np.maximum(times / start, 1.0) ** -index, 0.0
    )
    signal = _prompt(times) + envelope * np.cos(CARRIER * times)
    return times, signal, start


def _deep_tail_case() -> tuple[np.ndarray, np.ndarray]:
    """Return a signal whose tail sits near the double-precision floor.

    The production waveforms decay until the envelope approaches the level
    where round-off in the evolution itself dominates.  This case reproduces
    that ratio so the summation can be tested where it actually fails.
    """

    times = np.arange(0.0, 800.0, 0.05)
    scaled = np.maximum(times / 60.0, 1.0)
    envelope = np.where(times >= 60.0, 1.0e-9 * scaled**-3.0, 0.0)
    return times, _prompt(times) + envelope * np.cos(CARRIER * times)


def _exponential_case(rate: float) -> tuple[np.ndarray, np.ndarray, float]:
    times = np.arange(0.0, 1400.0, 0.05)
    start = 200.0
    envelope = np.where(
        times >= start, 1.0e-3 * np.exp(-rate * (times - start)), 0.0
    )
    signal = _prompt(times) + envelope * np.cos(CARRIER * times)
    return times, signal, start


class TestDynamicRange:
    def test_signal_square_range_defeats_a_global_cumulative_sum(self) -> None:
        """The deep tail case really does exceed the reciprocal of eps."""

        _, signal = _deep_tail_case()
        squares = signal**2
        positive = squares[squares > 0.0]
        assert positive.max() / positive.min() > 1.0 / np.finfo(float).eps

    def test_window_sums_survive_the_prompt_pulse(self) -> None:
        times, signal = _deep_tail_case()
        squares = signal**2
        count = 201
        summed = _centered_sum(squares, count)
        deep = np.searchsorted(times, 600.0)
        exact = float(
            np.sum(
                squares[deep - count // 2 : deep + count // 2 + 1].astype(
                    np.longdouble
                )
            )
        )
        assert exact > 0.0
        assert summed[deep] == pytest.approx(exact, rel=1.0e-12)

    def test_global_cumulative_sum_loses_the_tail(self) -> None:
        """Guard the regression: differencing cumulative sums returns zero."""

        times, signal = _deep_tail_case()
        squares = signal**2
        count = 201
        cumulative = np.concatenate(([0.0], np.cumsum(squares)))
        differenced = cumulative[count:] - cumulative[:-count]
        deep = np.searchsorted(times, 600.0)
        assert differenced[deep - count // 2] == 0.0
        assert _centered_sum(squares, count)[deep] > 0.0


class TestKnownDecayLaws:
    @pytest.mark.parametrize("index", (3.0, 5.0))
    def test_power_law_index_is_recovered(self, index: float) -> None:
        times, signal, start = _power_law_case(index)
        amplitude = rms_envelope(times, signal, 10.0)
        measured = local_log_fit_rate(
            times, amplitude, 40.0, logarithmic_time=True
        )
        window = (times > start + 200.0) & (times < 1200.0)
        sampled = measured[window]
        sampled = sampled[np.isfinite(sampled)]
        assert sampled.size > 1000
        assert np.median(sampled) == pytest.approx(index, abs=2.0e-3)
        assert np.max(np.abs(sampled - index)) < 2.0e-2

    def test_exponential_rate_is_recovered(self) -> None:
        rate = 0.01
        times, signal, start = _exponential_case(rate)
        amplitude = rms_envelope(times, signal, 10.0)
        measured = local_log_fit_rate(
            times, amplitude, 0.25 / rate, logarithmic_time=False
        )
        window = (times > start + 100.0) & (times < 1100.0)
        sampled = measured[window]
        sampled = sampled[np.isfinite(sampled)]
        assert sampled.size > 1000
        assert np.median(sampled) == pytest.approx(rate, rel=2.0e-3)

    def test_effective_rates_reports_both_indices_together(self) -> None:
        rate = 0.01
        times, signal, start = _exponential_case(rate)
        settings = LocalFitSettings(exponential_scaled_window=0.25)
        amplitude, power, normalized = effective_rates(
            times, signal, settings, kappa=rate
        )
        window = (times > start + 100.0) & (times < 1100.0)
        finite = np.isfinite(normalized[window])
        assert np.median(normalized[window][finite]) == pytest.approx(
            1.0, rel=2.0e-3
        )
        assert np.all(amplitude[window][finite] > 0.0)

    def test_the_envelope_tracks_a_known_power_law(self) -> None:
        times, signal, start = _power_law_case(3.0)
        amplitude = rms_envelope(times, signal, 10.0)
        probe = np.searchsorted(times, 1000.0)
        expected = 1.0e-3 * (times[probe] / start) ** -3.0 / np.sqrt(2.0)
        assert amplitude[probe] == pytest.approx(expected, rel=5.0e-3)
