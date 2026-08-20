"""Tests for the alternative large-L extrapolants.

Each estimator claims to annihilate a stated set of powers of ``1/L``.  The
tests build sequences that obey those models exactly and require the estimator
to return the planted limit, so a wrong coefficient is caught by arithmetic
rather than by inspection of a table.
"""

from __future__ import annotations

import numpy as np
import pytest

from black_hole.regulator_robustness import (
    NESTED_LINEAR_QUADRATIC,
    NESTED_QUADRATIC_QUARTIC,
    PAIR_LINEAR,
    PAIR_QUADRATIC,
    _dyadic_pairs,
    _dyadic_triples,
    _least_squares_limit,
    estimators,
)


LENGTHS = (20, 40, 80, 160, 320, 640)


def _sequence(limit: np.ndarray, terms: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Return ``W(L)`` for a planted limit and a set of ``1/L^power`` terms."""

    return {
        length: limit
        + sum(
            coefficient * float(length) ** -power
            for power, coefficient in terms.items()
        )
        for length in LENGTHS
    }


class TestLadderEnumeration:
    def test_every_dyadic_triple_in_the_sequence_is_used(self) -> None:
        assert _dyadic_triples(LENGTHS) == [
            (20, 40, 80),
            (40, 80, 160),
            (80, 160, 320),
            (160, 320, 640),
        ]

    def test_pairs_cover_the_whole_ladder(self) -> None:
        assert _dyadic_pairs(LENGTHS) == [
            (20, 40),
            (40, 80),
            (80, 160),
            (160, 320),
            (320, 640),
        ]


class TestCoefficientsAnnihilateTheirModel:
    def test_nested_triple_removes_linear_and_quadratic_terms(self) -> None:
        limit = np.asarray([3.0, -1.0])
        signals = _sequence(
            limit, {1: np.asarray([5.0, 2.0]), 2: np.asarray([-7.0, 11.0])}
        )
        first, second, third, divisor = NESTED_LINEAR_QUADRATIC
        for base, middle, top in _dyadic_triples(LENGTHS):
            recovered = (
                first * signals[base] + second * signals[middle] + third * signals[top]
            ) / divisor
            assert recovered == pytest.approx(limit, rel=1.0e-12)

    def test_even_triple_removes_quadratic_and_quartic_terms(self) -> None:
        limit = np.asarray([0.5])
        signals = _sequence(limit, {2: np.asarray([9.0]), 4: np.asarray([-4.0])})
        first, second, third, divisor = NESTED_QUADRATIC_QUARTIC
        for base, middle, top in _dyadic_triples(LENGTHS):
            recovered = (
                first * signals[base] + second * signals[middle] + third * signals[top]
            ) / divisor
            assert recovered == pytest.approx(limit, rel=1.0e-12)

    def test_linear_pair_removes_a_single_inverse_power(self) -> None:
        limit = np.asarray([2.0])
        signals = _sequence(limit, {1: np.asarray([13.0])})
        first, second, divisor = PAIR_LINEAR
        for base, top in _dyadic_pairs(LENGTHS):
            recovered = (first * signals[base] + second * signals[top]) / divisor
            assert recovered == pytest.approx(limit, rel=1.0e-12)

    def test_quadratic_pair_removes_a_single_squared_power(self) -> None:
        limit = np.asarray([-6.0])
        signals = _sequence(limit, {2: np.asarray([4.0])})
        first, second, divisor = PAIR_QUADRATIC
        for base, top in _dyadic_pairs(LENGTHS):
            recovered = (first * signals[base] + second * signals[top]) / divisor
            assert recovered == pytest.approx(limit, rel=1.0e-12)

    def test_a_wrong_model_does_not_return_the_limit(self) -> None:
        """The quadratic pair must fail on a sequence with a linear term."""

        limit = np.asarray([1.0])
        signals = _sequence(limit, {1: np.asarray([1.0])})
        first, second, divisor = PAIR_QUADRATIC
        recovered = (first * signals[80] + second * signals[160]) / divisor
        assert not np.isclose(recovered, limit, rtol=1.0e-6)


class TestLeastSquares:
    def test_fit_recovers_a_planted_limit(self) -> None:
        limit = np.asarray([7.0, -2.0, 0.25])
        signals = _sequence(
            limit, {1: np.asarray([1.0, 2.0, 3.0]), 2: np.asarray([-1.0, 0.5, 2.0])}
        )
        recovered = _least_squares_limit(signals, LENGTHS, (1, 2))
        assert recovered == pytest.approx(limit, rel=1.0e-10)

    def test_fit_uses_only_the_requested_subset(self) -> None:
        limit = np.asarray([1.5])
        signals = _sequence(limit, {1: np.asarray([4.0]), 2: np.asarray([-3.0])})
        signals[20] = signals[20] + 1000.0  # contaminate a member outside the subset
        subset = (80, 160, 320, 640)
        recovered = _least_squares_limit(signals, subset, (1, 2))
        assert recovered == pytest.approx(limit, rel=1.0e-10)


class TestEstimatorCatalogue:
    def test_catalogue_reports_distinct_models_for_shared_members(self) -> None:
        """Two estimators may share a member list; the model must separate them."""

        signals = _sequence(np.asarray([1.0]), {1: np.asarray([1.0])})
        catalogue = estimators(signals)
        keys = [
            (row["estimator"], row["model"], row["members"]) for row in catalogue
        ]
        assert len(keys) == len(set(keys))

    def test_every_triple_appears_under_both_truncations(self) -> None:
        signals = _sequence(np.asarray([1.0]), {1: np.asarray([1.0])})
        catalogue = estimators(signals)
        nested = {
            row["members"] for row in catalogue if row["estimator"] == "nested_triple"
        }
        even = {
            row["members"]
            for row in catalogue
            if row["estimator"] == "nested_triple_even"
        }
        assert nested == even
        assert len(nested) == 4
