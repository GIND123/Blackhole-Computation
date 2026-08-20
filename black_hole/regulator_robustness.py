"""Robustness of the large-L recovery to the assumed truncation.

The headline recovery uses one nested combination of four cosmological
lengths, built on the assumption that the finite-L waveform admits an
expansion in ``1/L`` whose first two terms can be cancelled.  Two questions
follow from that choice and neither is answered by the headline number: does
the conclusion survive a different assumed truncation, and does it survive
using different members of the archived sequence?

Nothing here runs a new simulation.  Every estimator below is a different
linear combination of the same frozen fine-level archives, compared against
the same independently evolved Schwarzschild waveform on the same windows,
with no relative time translation fitted anywhere.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .regulator_analysis import (
    CUMULATIVE_WINDOWS,
    _align_flat,
    _flat_signal,
    _l2,
    _retarded_times,
    _window_mask,
    load_flat_archives,
)
from .regulator_suite import FLAT_LENGTHS


OUTPUT_ROOT = Path("results/regulator_production_v3")

# Each estimator is a set of coefficients on a dyadic ladder ``(L, 2L, 4L)``
# or a pair ``(L, 2L)``.  The coefficients annihilate the stated powers of
# ``1/L`` exactly, so the label states the model rather than a preference.
NESTED_LINEAR_QUADRATIC = (1.0, -6.0, 8.0, 3.0)
NESTED_QUADRATIC_QUARTIC = (1.0, -20.0, 64.0, 45.0)
PAIR_LINEAR = (-1.0, 2.0, 1.0)
PAIR_QUADRATIC = (-1.0, 4.0, 3.0)


def _dyadic_triples(lengths: tuple[int, ...]) -> list[tuple[int, int, int]]:
    available = set(lengths)
    return [
        (base, 2 * base, 4 * base)
        for base in lengths
        if 2 * base in available and 4 * base in available
    ]


def _dyadic_pairs(lengths: tuple[int, ...]) -> list[tuple[int, int]]:
    available = set(lengths)
    return [(base, 2 * base) for base in lengths if 2 * base in available]


def _least_squares_limit(
    signals: dict[int, np.ndarray], lengths: tuple[int, ...], powers: tuple[int, ...]
) -> np.ndarray:
    """Return the fitted ``L -> infinity`` limit for one model and subset."""

    design = np.stack(
        [np.ones(len(lengths))]
        + [np.asarray([float(length) ** -power for length in lengths])
           for power in powers],
        axis=1,
    )
    stacked = np.stack([signals[length] for length in lengths])
    solution, *_ = np.linalg.lstsq(design, stacked, rcond=None)
    return solution[0]


def estimators(signals: dict[int, np.ndarray]) -> list[dict]:
    """Return every extrapolant that the archived ladder supports."""

    lengths = tuple(sorted(signals))
    rows: list[dict] = []
    for base, middle, top in _dyadic_triples(lengths):
        first, second, third, divisor = NESTED_LINEAR_QUADRATIC
        rows.append(
            {
                "estimator": "nested_triple",
                "model": "W(L) = W + a/L + b/L^2",
                "members": f"{base},{middle},{top}",
                "waveform": (
                    first * signals[base]
                    + second * signals[middle]
                    + third * signals[top]
                )
                / divisor,
            }
        )
        first, second, third, divisor = NESTED_QUADRATIC_QUARTIC
        rows.append(
            {
                "estimator": "nested_triple_even",
                "model": "W(L) = W + a/L^2 + b/L^4",
                "members": f"{base},{middle},{top}",
                "waveform": (
                    first * signals[base]
                    + second * signals[middle]
                    + third * signals[top]
                )
                / divisor,
            }
        )
    for base, top in _dyadic_pairs(lengths):
        for name, model, (first, second, divisor) in (
            ("pair_linear", "W(L) = W + a/L", PAIR_LINEAR),
            ("pair_quadratic", "W(L) = W + a/L^2", PAIR_QUADRATIC),
        ):
            rows.append(
                {
                    "estimator": name,
                    "model": model,
                    "members": f"{base},{top}",
                    "waveform": (first * signals[base] + second * signals[top])
                    / divisor,
                }
            )
    for powers, label in (((1, 2), "W + a/L + b/L^2"), ((2, 4), "W + a/L^2 + b/L^4")):
        rows.append(
            {
                "estimator": "least_squares_all",
                "model": label,
                "members": ",".join(str(value) for value in lengths),
                "waveform": _least_squares_limit(signals, lengths, powers),
            }
        )
        for dropped in lengths:
            subset = tuple(value for value in lengths if value != dropped)
            if len(subset) <= len(powers):
                continue
            rows.append(
                {
                    "estimator": "least_squares_drop_one",
                    "model": label,
                    "members": ",".join(str(value) for value in subset),
                    "waveform": _least_squares_limit(signals, subset, powers),
                }
            )
    return rows


def analyse(output_dir: Path = OUTPUT_ROOT) -> dict:
    """Compare every estimator with the independent Schwarzschild waveform."""

    archives = load_flat_archives(Path(output_dir))
    fine = archives["fine"]
    reference_result = fine[None]
    reference_times = _retarded_times(reference_result)
    reference_signal = _flat_signal(reference_result)
    common = (reference_times >= 0.0) & (reference_times <= 160.0)
    times = reference_times[common]
    reference = reference_signal[common]
    signals = {
        int(length): _align_flat(fine[float(length)], times)
        for length in FLAT_LENGTHS
    }

    rows: list[dict] = []
    for record in estimators(signals):
        for window, start, end in CUMULATIVE_WINDOWS:
            mask = _window_mask(times, start, end)
            denominator = _l2(reference[mask], times[mask])
            residual = _l2(record["waveform"][mask] - reference[mask], times[mask])
            rows.append(
                {
                    "estimator": record["estimator"],
                    "model": record["model"],
                    "members": record["members"],
                    "window": window,
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "E2_percent": 100.0 * residual / denominator,
                    "within_1_percent": bool(100.0 * residual / denominator < 1.0),
                }
            )

    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    detail = tables / "extrapolation_robustness.csv"
    with detail.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict] = []
    for record in estimators(signals):
        # The model belongs in the key: two estimators can share a name and a
        # member list while assuming different truncations.
        key = (record["estimator"], record["model"], record["members"])
        matching = [
            row
            for row in rows
            if (row["estimator"], row["model"], row["members"]) == key
        ]
        worst = max(matching, key=lambda row: row["E2_percent"])
        summary_rows.append(
            {
                "estimator": record["estimator"],
                "model": record["model"],
                "members": record["members"],
                "worst_cumulative_E2_percent": worst["E2_percent"],
                "worst_window": worst["window"],
                "within_1_percent": worst["E2_percent"] < 1.0,
            }
        )
    summary_rows.sort(key=lambda row: row["worst_cumulative_E2_percent"])
    compact = tables / "extrapolation_robustness_summary.csv"
    with compact.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    passing = [row for row in summary_rows if row["within_1_percent"]]
    anchored = [
        row
        for row in summary_rows
        if min(int(value) for value in row["members"].split(",")) >= 80
    ]
    summary = {
        "estimators_anchored_at_L80_or_above": len(anchored),
        "anchored_estimators_within_1_percent": sum(
            1 for row in anchored if row["within_1_percent"]
        ),
        "worst_anchored_E2_percent": (
            max(row["worst_cumulative_E2_percent"] for row in anchored)
            if anchored
            else None
        ),
        "estimators_tested": len(summary_rows),
        "estimators_within_1_percent": len(passing),
        "largest_worst_case_E2_percent": max(
            row["worst_cumulative_E2_percent"] for row in summary_rows
        ),
        "largest_worst_case_among_passing_percent": (
            max(row["worst_cumulative_E2_percent"] for row in passing)
            if passing
            else None
        ),
        "distinct_models": sorted({row["model"] for row in summary_rows}),
        "detail_table": str(detail),
        "summary_table": str(compact),
        "time_translation_fitted": False,
    }
    record_path = tables / "extrapolation_robustness.json"
    with record_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    print(json.dumps(analyse(arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
