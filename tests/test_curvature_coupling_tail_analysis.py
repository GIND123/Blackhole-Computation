import numpy as np

from black_hole.curvature_coupling_tail_analysis import (
    FAMILIES,
    away_from_zero_crossings,
    exponential_row,
    longest_interval,
    power_law_fit_row,
)


def test_longest_interval_selects_longest_contiguous_run():
    times = np.arange(8.0)
    selected = np.array([False, True, True, False, True, True, True, False])

    assert longest_interval(times, selected) == (4.0, 6.0)


def test_zero_crossing_mask_removes_declared_neighborhood():
    times = np.arange(6.0)
    signal = np.array([1.0, 0.5, -0.5, -1.0, -0.5, -0.25])

    safe = away_from_zero_crossings(times, signal, half_width=1.0)

    assert np.array_equal(safe, [True, False, False, True, True, True])


def test_exponential_criterion_requires_scaled_duration():
    family = next(item for item in FAMILIES if item.key == "uniform_xi1o6")
    times = np.linspace(100.0, 400.0, 601)
    diagnostic = {
        "times": times,
        "gamma_over_kappa": np.full_like(times, 2.0),
        "gamma_supported": np.ones_like(times, dtype=bool),
    }

    row = exponential_row(family, diagnostic)

    assert row["passes_exponential_criterion"] is True
    assert row["candidate_duration_over_M"] == 300.0


def test_power_law_fit_recovers_synthetic_exponent():
    family = next(item for item in FAMILIES if item.key == "exterior_xi0")
    times = np.linspace(100.0, 300.0, 1001)
    diagnostic = {"times": times, "amplitude": 0.02 * times ** -1.25}

    row = power_law_fit_row(family, diagnostic, 120.0, 250.0)

    assert np.isclose(row["power_law_exponent"], 1.25)
    assert np.isclose(row["r_squared"], 1.0)
