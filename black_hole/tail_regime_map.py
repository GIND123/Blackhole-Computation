"""Where each cosmological length can and cannot resolve a decay regime.

The single length study answers whether one calculation sees the Price tail
and the cosmological decay in the same waveform.  It cannot say whether some
other length would, and the scaling argument that says none will is an
argument rather than a measurement.  This module makes it a measurement, by
placing every completed final ladder on one axis.

Two requirements pull against each other:

* the Schwarzschild power law needs the tail resolved for a few hundred ``M``
  after ringdown, which is easier the larger ``L`` is, since the cosmological
  horizon interferes later;
* the cosmological decay needs ``kappa_c U`` of order a few, that is ``U`` of
  order a few ``L``, by which time a ``U^-3`` tail has fallen by a further
  factor of order ``(3 L / U_P)^3``.

Raising ``L`` helps the first and hurts the second.  The question is whether
the two windows overlap anywhere, and that is decided by the measured floor of
each ladder rather than by an estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .large_l_tail import (  # noqa: E402
    FLOOR_SAFETY_FACTOR,
    LocalFitSettings,
    OUTPUT_ROOT,
    PRICE_DURATION,
    SCREEN_LENGTHS,
    _load_final_set,
    cosmological_rate,
    final_cases,
    archive_path,
    ladder_envelope_floor,
    measure_transition,
    price_target,
    trusted_interval_end,
)

OBSERVER_NAMES = {0: "r8M", 1: "r16M", 2: "outer"}
# The cosmological rate is only meaningfully separated from a power law once
# kappa_c U is of order the Price index; below this the two are numerically
# indistinguishable whatever the resolution.
COSMOLOGICAL_SCALED_TIME = 3.0


def completed_lengths(output_dir: Path = OUTPUT_ROOT) -> list[float]:
    """Return the lengths whose final ladder is complete on disk."""

    lengths = []
    for length in SCREEN_LENGTHS:
        if all(
            archive_path(Path(output_dir), case).exists()
            for case in final_cases(length)
        ):
            lengths.append(float(length))
    return lengths


def measure_length(length: float, output_dir: Path = OUTPUT_ROOT) -> list[dict]:
    """Return one row per observer describing what this ladder resolves."""

    results = _load_final_set(Path(output_dir), length)
    kappa = cosmological_rate(length)
    primary_sds = results[("sds", 3072, 0.0025)]
    primary_reference = results[("schwarzschild", 3072, 0.0025)]
    rows = []
    for observer in range(3):
        ladder = ladder_envelope_floor(results, observer, LocalFitSettings())
        reference_ladder = ladder_envelope_floor(
            results, observer, LocalFitSettings(), background="schwarzschild"
        )
        measurement = measure_transition(
            primary_sds,
            primary_reference,
            observer,
            LocalFitSettings(),
            measured_floor=ladder["floor"],
        )
        trusted = trusted_interval_end(
            ladder["times"], ladder["amplitude"], ladder["floor"]
        )
        reference_trusted = trusted_interval_end(
            reference_ladder["times"],
            reference_ladder["amplitude"],
            reference_ladder["floor"],
        )
        usable = np.isfinite(ladder["convergence_ratio"]) & (ladder["times"] > 200.0)
        departure = measurement["departure_U_over_M"]
        rows.append(
            {
                "length_over_M": length,
                "observer": OBSERVER_NAMES[observer],
                "price_target": price_target(observer),
                "kappa_c": kappa,
                "price_departure_U_over_M": departure,
                "price_departure_kappa_U": (
                    None if departure is None else kappa * departure
                ),
                "price_established": departure is not None,
                "cosmological_entry_U_over_M": measurement["entry_U_over_M"],
                "cosmological_entry_unanchored_U_over_M": measurement[
                    "cosmological_entry_without_price_anchor_U_over_M"
                ],
                "cosmological_entry_unanchored_kappa_U": measurement[
                    "cosmological_entry_without_price_anchor_kappa_U"
                ],
                "cosmological_regime_resolved": bool(
                    measurement["cosmological_entry_without_price_anchor_U_over_M"]
                    is not None
                ),
                "trusted_until_U_over_M": trusted,
                "trusted_until_kappa_U": None if trusted is None else kappa * trusted,
                "reference_trusted_until_U_over_M": reference_trusted,
                "cosmological_needs_kappa_U": COSMOLOGICAL_SCALED_TIME,
                "cosmological_scaled_time_reached": (
                    None if trusted is None else kappa * trusted
                ),
                "cosmological_reachable": (
                    None
                    if trusted is None
                    else bool(kappa * trusted >= COSMOLOGICAL_SCALED_TIME)
                ),
                "both_regimes_resolved": bool(
                    departure is not None
                    and measurement["entry_U_over_M"] is not None
                ),
                "median_convergence_ratio": (
                    float(np.median(ladder["convergence_ratio"][usable]))
                    if np.any(usable)
                    else None
                ),
                "floor_safety_factor": FLOOR_SAFETY_FACTOR,
                "required_price_duration_over_M": PRICE_DURATION,
                "status": measurement["status"],
            }
        )
    return rows


def _normalized_rate_curve(length: float, output_dir: Path) -> dict | None:
    """Return the trusted part of ``gamma_eff/kappa_c`` against ``kappa_c U``."""

    from .large_l_tail import local_log_fit_rate

    results = _load_final_set(Path(output_dir), length)
    kappa = cosmological_rate(length)
    settings = LocalFitSettings()
    ladder = ladder_envelope_floor(results, 2, settings)
    times = ladder["times"]
    trusted = (
        np.isfinite(ladder["amplitude"])
        & np.isfinite(ladder["floor"])
        & (ladder["floor"] > 0.0)
        & (ladder["amplitude"] > FLOOR_SAFETY_FACTOR * ladder["floor"])
    )
    amplitude = np.where(trusted, ladder["amplitude"], np.nan)
    gamma = local_log_fit_rate(
        times, amplitude, settings.exponential_scaled_window / kappa,
        logarithmic_time=False,
    ) / kappa
    keep = np.isfinite(gamma) & trusted & (gamma > 0.0)
    if not keep.any():
        return None
    return {
        "scaled_time": kappa * times[keep],
        "normalized_gamma": gamma[keep],
    }


def build(output_dir: Path = OUTPUT_ROOT) -> dict:
    """Measure every completed ladder and draw the regime map."""

    lengths = completed_lengths(output_dir)
    if not lengths:
        raise FileNotFoundError("No final ladder is complete.")
    rows: list[dict] = []
    for length in lengths:
        rows.extend(measure_length(length, output_dir))

    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table_path = tables / "tail_regime_map.csv"
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    outer = [row for row in rows if row["observer"] == "outer"]
    outer.sort(key=lambda row: row["length_over_M"])

    figure, axes = plt.subplots(3, 1, figsize=(8.2, 11.4))
    values = [row["length_over_M"] for row in outer]

    # Upper panel: everything in cosmological time, where the two requirements
    # can be compared directly.
    trusted = [row["trusted_until_kappa_U"] for row in outer]
    departure = [row["price_departure_kappa_U"] for row in outer]
    axes[0].axhspan(
        COSMOLOGICAL_SCALED_TIME,
        max([value for value in trusted if value] + [COSMOLOGICAL_SCALED_TIME]) * 3.0,
        color="#0072B2",
        alpha=0.10,
    )
    axes[0].axhline(
        COSMOLOGICAL_SCALED_TIME, color="#0072B2", linestyle="--", linewidth=1.2,
        label=r"cosmological decay needs $\kappa_c U\gtrsim%.0f$"
        % COSMOLOGICAL_SCALED_TIME,
    )
    axes[0].plot(values, trusted, "o-", color="#D55E00", linewidth=1.8,
                 label="trusted to (ladder floor)")
    marked = [
        (value, point)
        for value, point in zip(values, departure)
        if point is not None
    ]
    if marked:
        axes[0].plot(
            [item[0] for item in marked], [item[1] for item in marked],
            "s--", color="#009E73", linewidth=1.5,
            label=r"Price departure $\kappa_c U_{\rm P}$",
        )
    axes[0].set(
        xscale="log", yscale="log",
        ylabel=r"cosmological time $\kappa_c U$",
        title="What each cosmological length can reach before its ladder floor",
    )
    axes[0].set_xticks(values)
    axes[0].set_xticklabels([f"{value:g}" for value in values])
    axes[0].xaxis.set_minor_locator(plt.NullLocator())
    axes[0].legend(fontsize=8.5, loc="lower left")
    axes[0].grid(alpha=0.25, which="both")

    # Lower panel: the same thing in units of M, where the Price requirement
    # lives, with the established interval drawn as a bar.
    for row in outer:
        end = row["price_departure_U_over_M"]
        if end is None:
            continue
        start = max(end - PRICE_DURATION, 0.0)
        axes[1].plot(
            [row["length_over_M"]] * 2, [start, end],
            color="#009E73", linewidth=7.0, solid_capstyle="butt", alpha=0.85,
            label="_",
        )
    axes[1].plot(
        values, [row["trusted_until_U_over_M"] for row in outer],
        "o-", color="#D55E00", linewidth=1.8, label="trusted to (ladder floor)",
    )
    axes[1].plot(
        values, [COSMOLOGICAL_SCALED_TIME / row["kappa_c"] for row in outer],
        "^--", color="#0072B2", linewidth=1.5,
        label=r"$U$ at which $\kappa_c U=%.0f$" % COSMOLOGICAL_SCALED_TIME,
    )
    axes[1].plot([], [], color="#009E73", linewidth=7.0, alpha=0.85,
                 label="established Price interval")
    axes[1].set(
        xscale="log", yscale="log",
        xlabel=r"cosmological length $L/M$",
        ylabel=r"retarded time $U/M$",
    )
    axes[1].set_xticks(values)
    axes[1].set_xticklabels([f"{value:g}" for value in values])
    axes[1].xaxis.set_minor_locator(plt.NullLocator())
    axes[1].legend(fontsize=8.5, loc="upper left")
    axes[1].grid(alpha=0.25, which="both")

    # Third panel: the measured rate itself, in cosmological time.  If the
    # obstruction were a matter of choosing L better, these curves would reach
    # different depths; the question is whether any of them reaches unity.
    for length, colour in zip(lengths, plt.cm.viridis(
            np.linspace(0.1, 0.85, len(lengths)))):
        curve = _normalized_rate_curve(length, output_dir)
        if curve is None:
            continue
        axes[2].plot(curve["scaled_time"], curve["normalized_gamma"],
                     color=colour, linewidth=1.8, label=r"$L/M=%g$" % length)
    axes[2].axhspan(0.9, 1.1, color="0.85")
    axes[2].axhline(1.0, color="0.35", linestyle="--", linewidth=1.1)
    axes[2].axvline(COSMOLOGICAL_SCALED_TIME, color="#0072B2", linestyle=":",
                    linewidth=1.2)
    axes[2].text(COSMOLOGICAL_SCALED_TIME, 0.97,
                 r"  $\kappa_c U=%.0f$" % COSMOLOGICAL_SCALED_TIME,
                 transform=axes[2].get_xaxis_transform(), fontsize=8,
                 va="top", color="#0072B2")
    axes[2].text(0.99, 1.15, r"cosmological target $\gamma_{\rm eff}/\kappa_c=1$",
                 transform=axes[2].get_yaxis_transform(), ha="right",
                 va="bottom", fontsize=8)
    axes[2].set(
        xlabel=r"cosmological time $\kappa_c U$",
        ylabel=r"$\gamma_{\rm eff}/\kappa_c$",
        yscale="log",
        title=(
            "The power law collapses in cosmological time; only the shortest\n"
            "length reaches the exponential before its floor"
        ),
    )
    axes[2].legend(fontsize=8.5)
    axes[2].grid(alpha=0.25, which="both")

    figure.tight_layout()
    png = Path(output_dir) / "tail_regime_map.png"
    pdf = Path(output_dir) / "tail_regime_map.pdf"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    overlap = [row for row in rows if row["both_regimes_resolved"]]
    summary = {
        "lengths_measured": lengths,
        "observers": sorted(OBSERVER_NAMES.values()),
        "cosmological_scaled_time_required": COSMOLOGICAL_SCALED_TIME,
        "floor_safety_factor": FLOOR_SAFETY_FACTOR,
        "any_length_resolves_both_regimes": bool(overlap),
        "lengths_resolving_both": sorted(
            {row["length_over_M"] for row in overlap}
        ),
        "lengths_establishing_price_at_the_outer_boundary": sorted(
            {row["length_over_M"] for row in outer if row["price_established"]}
        ),
        "lengths_reaching_the_cosmological_regime": sorted(
            {
                row["length_over_M"]
                for row in outer
                if row["cosmological_regime_resolved"]
            }
        ),
        "lengths_whose_record_reaches_the_required_scaled_time": sorted(
            {
                row["length_over_M"]
                for row in outer
                if row["cosmological_reachable"]
            }
        ),
        "table": str(table_path),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "time_translation_fitted": False,
    }
    record = tables / "tail_regime_map.json"
    with record.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=float)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output_dir), indent=2, default=float))


if __name__ == "__main__":
    main()
