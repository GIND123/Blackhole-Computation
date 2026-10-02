"""Regenerate the manuscript figures from their archived numerical inputs."""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from black_hole.regulator_analysis import (
    FLAT_LENGTHS,
    create_plots,
    flat_analysis,
    l12_phase_cleanup,
    source_analysis,
)


def create_flat_column_figure(flat: dict, destination: Path) -> None:
    """Render the frozen waveform sequence at its one-column print size."""
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True)
        times, reference = flat["times"], flat["reference"]
        axes[0].plot(times, reference, color="black", linewidth=1.2,
                     label="Schwarzschild")
        for length in FLAT_LENGTHS:
            signal = flat["fine_signals"][length]
            axes[0].plot(times, signal, linewidth=0.8, label=f"$L/M={length}$")
            axes[1].plot(times, signal - reference, linewidth=0.8)
        axes[0].set_ylabel("boundary waveform")
        axes[1].set(xlabel=r"retarded time $U/M$", ylabel=r"$W_L-W_{\rm Schw}$")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.set_xlim(0, 80)
            axis.tick_params(labelsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
                   frameon=False, columnspacing=0.8, handlelength=1.5,
                   handletextpad=0.4)
        fig.tight_layout(rect=(0, 0, 1, 0.82), pad=0.5)
        fig.savefig(destination)
        plt.close(fig)


def create_localized_column_figure(source: dict, destination: Path) -> None:
    """Show localized-source recovery without redundant directional diagnostics."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter

    waveform = source["waveform"]
    metrics = waveform["metrics"]
    times = waveform["times"]
    norms = waveform["sphere_plot"]
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(
            2, 1, figsize=(3.4, 4.4),
            gridspec_kw={"height_ratios": (1, 1.35)},
        )
        axes[0].loglog(
            [row["cosmological_length_over_M"] for row in metrics],
            [100 * row["E2"] for row in metrics],
            "o-", color="#16697a", linewidth=1.1, markersize=4,
        )
        axes[0].axhline(1, color="0.45", linestyle="--", linewidth=0.8)
        axes[0].text(82, 1.12, "1%", color="0.35", fontsize=8)
        axes[0].set(
            xlabel=r"$L/M$", ylabel=r"$E_{2,S^2}$ (%)",
            xlim=(72, 710), ylim=(0.8, 27),
            xticks=[80, 160, 320, 640], yticks=[1, 3, 10, 20],
        )
        axes[0].xaxis.set_major_formatter(ScalarFormatter())
        axes[0].yaxis.set_major_formatter(ScalarFormatter())
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[0].yaxis.set_minor_locator(NullLocator())
        axes[0].set_title("(a) Direct sphere-integrated error", loc="left", fontsize=8.5)

        for key, color, label in (
            ("reference", "black", "Schwarzschild norm"),
            ("L320_residual", "#0072B2", r"$L/M=320$ residual"),
            ("L640_residual", "#D55E00", r"$L/M=640$ residual"),
            ("extrapolant_difference", "#009E73", "extrapolant difference"),
        ):
            axes[1].semilogy(times, norms[key], color=color, linewidth=1.0,
                             label=label)
        axes[1].set(
            xlabel=r"retarded time $U/M$", ylabel=r"$L^2(S^2)$ norm",
            xlim=(23, times[-1]), ylim=(1e-8, 0.12),
            xticks=[25, 35, 45, 55], yticks=[1e-1, 1e-3, 1e-5, 1e-7],
        )
        axes[1].set_title("(b) Time-resolved modal norms", loc="left", fontsize=8.5)
        axes[1].legend(loc="lower right", fontsize=7.5, frameon=False,
                       handlelength=1.5, handletextpad=0.5, labelspacing=0.25)
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.19, right=0.97, bottom=0.10, top=0.94,
                            hspace=0.57)
        fig.savefig(destination)
        plt.close(fig)


def create_qnm_column_figure(destination: Path) -> None:
    """Plot the archived, aligned residuals at one-column print size."""
    import matplotlib.pyplot as plt
    import numpy as np

    table = REPOSITORY_ROOT / "results/exterior_regulator_width_floor_qnm_v5/tables/width_floor_aligned_waveforms.csv"
    data = np.genfromtxt(table, delimiter=",", names=True)
    times = data["U_over_M"]
    mask = (times >= 0) & (times <= 60)
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True, sharey=True)
        for axis, family, title in zip(
            axes, ("uniform", "exterior"), ("Uniform SdS", "Exterior-supported SdS")
        ):
            axis.axvspan(15, 45, color="0.92", linewidth=0)
            axis.axhline(0, color="0.45", linewidth=0.6)
            for length, color in zip((80, 160, 320, 640), ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):
                residual = data[f"{family}_sds_L{length}"] - data["schwarzschild_fine"]
                axis.plot(times[mask], residual[mask], color=color, linewidth=0.9,
                          label=f"$L/M={length}$")
            axis.set(xlim=(0, 60), ylim=(-0.045, 0.045),
                     ylabel=r"$W_L-W_{\rm Schw}$")
            axis.set_title(title, fontsize=9)
            axis.grid(axis="y", color="0.88", linewidth=0.5)
        axes[0].text(30, -0.0405, "ringdown window", ha="center", fontsize=7,
                     color="0.35")
        axes[1].set_xlabel(r"$U/M$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8,
                   frameon=False, handlelength=2)
        fig.tight_layout(rect=(0, 0, 1, 0.87), pad=0.5)
        fig.savefig(destination)
        plt.close(fig)


def create_price_column_figure(destination: Path) -> None:
    """Lead with the resolved Price interval and retain masked late-time context."""
    import json
    import matplotlib.pyplot as plt
    from black_hole.large_l_tail import (
        LocalFitSettings, TIMESTEP, _load_final_set, cosmological_rate,
        effective_rates, ladder_envelope_floor, retarded_series,
        trusted_interval_end,
    )

    root = REPOSITORY_ROOT / "results/large_l_tail"
    summary = json.loads((root / "tables/final_L3072_summary.json").read_text())
    interval = summary["price_fit_interval_U_over_M"]
    results = _load_final_set(root, 3072)
    settings = LocalFitSettings()
    kappa = cosmological_rate(3072)
    curves = {}
    limits = []
    for background in ("sds", "schwarzschild"):
        ladder = ladder_envelope_floor(results, 2, settings, background=background)
        times, signal = retarded_series(results[(background, 3072, TIMESTEP)], 2)
        amplitude, power, gamma = effective_rates(
            times, signal, settings, kappa=kappa, measured_floor=ladder["floor"]
        )
        curves[background] = (times, amplitude, power, gamma)
        limits.append(trusted_interval_end(times, amplitude, ladder["floor"]))
    common_limit = min(limits)

    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42,
                         "lines.linewidth": 1.0}):
        fig, axes = plt.subplots(
            2, 1, figsize=(3.4, 4.2), gridspec_kw={"height_ratios": (1.2, 1)}
        )
        styles = (
            ("sds", "#D55E00", "-", r"SdS at $\mathcal{H}_c^+$"),
            ("schwarzschild", "0.2", "--", r"Schwarzschild at $\mathscr{I}^+$"),
        )
        for background, color, linestyle, label in styles:
            times, amplitude, power, _ = curves[background]
            axes[0].plot(times, power, color=color, linestyle=linestyle,
                         label=label)
            axes[1].semilogy(times, amplitude, color=color, linestyle=linestyle)
        for axis in axes:
            axis.axvspan(*interval, color="#F0E442", alpha=0.24, linewidth=0)
            for endpoint in interval:
                axis.axvline(endpoint, color="0.4", linestyle=":", linewidth=0.7)
            axis.grid(alpha=0.2)
            axis.set_xlabel(r"retarded time $U/M$")
        axes[0].set(
            xlim=(150, 500), ylim=(2.8, 3.4), ylabel=r"local power index $p_{\rm eff}$",
            xticks=[150, 250, 350, 450], yticks=[2.8, 3.0, 3.2, 3.4],
        )
        axes[0].set_title("(a) Intermediate Price decay", loc="left", fontsize=8.5)
        axes[0].axhspan(2.85, 3.15, color="0.5", alpha=0.12, linewidth=0)
        axes[0].axhline(3, color="0.5", linestyle=":", linewidth=0.7)
        axes[0].text(490, 2.87, r"$3\pm5\%$", ha="right", va="bottom", color="0.35")
        axes[0].text(sum(interval) / 2, 3.35, r"$\Delta U=150M$", ha="center",
                     va="center", fontsize=8)
        axes[1].set(
            xlim=(0, 1.06 * common_limit), ylim=(1e-10, 1),
            ylabel=r"RMS envelope $A$", xticks=[0, 500, 1000, 1500, 2000],
            yticks=[1e-1, 1e-3, 1e-5, 1e-7, 1e-9],
        )
        axes[1].set_title("(b) Longer-time envelope", loc="left", fontsize=8.5)
        axes[1].axvspan(common_limit, 1.06 * common_limit, color="0.9", linewidth=0)
        axes[1].axvline(common_limit, color="#7f2704", linewidth=0.8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=1, fontsize=8,
                   frameon=False, handlelength=2.2)
        fig.subplots_adjust(left=0.19, right=0.97, bottom=0.10, top=0.84,
                            hspace=0.58)
        fig.savefig(destination)
        plt.close(fig)


def create_matched_tail_column_figure(destination: Path) -> None:
    """Recompute the five-signal tail figure with its measured numerical masks."""
    from black_hole.sbp_matched_tail_figure import (
        REVISION_ROOT, create_figure, diagnose, load_members,
    )

    members = load_members(REVISION_ROOT)
    if destination.name != "matched_tail_comparison.pdf":
        raise ValueError("The matched-tail plotter uses a fixed artifact name")
    with (REVISION_ROOT / "analysis/matched_tail_intervals.csv").open() as stream:
        intervals = list(csv.DictReader(stream))
    create_figure({name: diagnose(ladder) for name, ladder in members.items()},
                  destination.parent, intervals)


def main() -> None:
    production = REPOSITORY_ROOT / "results" / "regulator_production_v3"
    destination = Path(__file__).resolve().parent / "figs"

    flat = flat_analysis(production)
    source = source_analysis(production, REPOSITORY_ROOT)
    phase_rows = l12_phase_cleanup(REPOSITORY_ROOT)

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sds-paper-figures-") as temporary:
        generated = create_plots(Path(temporary), flat, source, phase_rows)
        for path in generated:
            if path.suffix.lower() == ".pdf":
                target = destination / path.name
                shutil.copy2(path, target)
                print(target)

    create_flat_column_figure(flat, destination / "flat_waveform_sequence.pdf")
    create_localized_column_figure(source, destination / "localized_source_regulator.pdf")
    create_matched_tail_column_figure(destination / "matched_tail_comparison.pdf")
    create_price_column_figure(destination / "large_L3072_tail_transition.pdf")


if __name__ == "__main__":
    main()
