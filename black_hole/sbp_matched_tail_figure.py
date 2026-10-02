"""Matched five-signal tail comparison using the independent SBP--Radau runs.

The physical experiment and the archived 640M acceptance rule are unchanged.
This module adapts that rule to the new two-degree and fixed-degree timestep
controls; it does not silently apply the stricter large-L reference-matching
rule used by a different experiment.  It writes derived files only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .curvature_coupling_tail_analysis import (
    ANALYSIS_START, EXPONENTIAL_SCALED_WINDOW, FLOOR_SAFETY_FACTOR,
    PRICE_MINIMUM_DURATION, PRICE_TARGET, PRICE_TOLERANCE,
    RATE_REFINEMENT_TOLERANCE, away_from_zero_crossings, longest_interval,
)
from .large_l_tail import LocalFitSettings, cosmological_rate, effective_rates, rms_envelope
from .sbp_flagship_summary import Flagship
from .sbp_pilot_analysis import (
    DEFAULT_ROOT, REPOSITORY, Run, exact_difference, interpolate,
    interpolation_sensitivity, metrics, number, read_exclusions, read_run,
    uniform_grid, write_csv,
)


@dataclass(frozen=True)
class Family:
    stem: str
    label: str
    color: str
    linestyle: str
    exterior: bool = False

    def name(self, degree: int, dt: str) -> str:
        return Flagship(self.stem, self.label, self.color, self.exterior).name(degree, dt)


FAMILIES = (
    Family("schwarzschild", "Schwarzschild", "#222222", "-"),
    Family("uniform640_minimal", r"Uniform, $\xi=0$", "#D55E00", "--"),
    Family("uniform640_conformal", r"Uniform, $\xi=1/6$", "#CC9900", "-"),
    Family("exterior640_minimal", r"Exterior, $\xi=0$", "#0072B2", "--", True),
    Family("exterior640_conformal", r"Exterior, $\xi=1/6$", "#009E73", "-", True),
)
WINDOWS = ((150., 300.), (300., 500.), (500., 750.), (750., 950.))
REVISION_ROOT = REPOSITORY / "results/sbp_matched_tail_revision_v1"


def load_members(root: Path, baseline: Path = DEFAULT_ROOT) -> dict[str, dict[str, Run]]:
    """Read only completed matched runs with no mixed physical controls."""
    exclusions = {**read_exclusions(baseline), **read_exclusions(root)}
    members = {}
    for family in FAMILIES:
        ladder = {}
        for role, degree, dt in (("coarse", 24, "0p1"), ("medium", 32, "0p1"), ("fine", 40, "0p1"),
                                 ("half_dt", 40, "0p05")):
            name = family.name(degree, dt)
            if role == "coarse" and family.stem in ("schwarzschild", "uniform640_conformal"):
                name += "_verified"
            if name in exclusions:
                raise ValueError(f"Quarantined input {name}: {exclusions[name]}")
            path = root / "runs" / name
            if not path.exists():
                path = baseline / "runs" / name
            run = read_run(path)
            if run is None:
                raise FileNotFoundError(f"Required complete tail run: {name}")
            config = run.configuration
            if (run.integrator != "radau3" or run.precision != "double-double"
                    or run.degree != degree or run.dt != float(dt.replace("p", "."))):
                raise ValueError(f"Run is not the named double-double Radau control: {name}")
            if (int(config["ell"]) != 1 or run.times[-1] < 999.99
                    or config["layout"] != ("wave-resolved" if family.exterior else "standard")):
                raise ValueError(f"Wrong experiment or incomplete duration: {name}")
            if family.stem != "schwarzschild" and number(config["L_over_M"]) != 640:
                raise ValueError(f"Only the matched L/M=640 experiment belongs here: {name}")
            expected_background = ("schwarzschild" if family.stem == "schwarzschild"
                                   else "exterior" if family.exterior else "uniform")
            expected_xi = 1/6 if "conformal" in family.stem else 0.
            velocity_labels = {"u=0; v=areal bump center6 halfwidth3 amplitude1",
                               "u=0, d_tau u=C-infinity areal bump centered at 6M with half-width 3M"}
            if (config["background"] != expected_background
                    or abs(number(config["xi"])-expected_xi) > 1e-15
                    or config["initial_data"] not in velocity_labels
                    or config.get("initial_data_kind", "velocity") != "velocity"):
                raise ValueError(f"Wrong matched initial data, geometry, or coupling: {name}")
            ladder[role] = run
        if len({r.key for r in ladder.values()}) != 1:
            raise ValueError(f"Mixed physical families: {family.stem}")
        if len({r.configuration["q"] for r in ladder.values()}) != 1:
            raise ValueError(f"Mixed geometric clocks: {family.stem}")
        members[family.stem] = ladder
    return members


def diagnose(ladder: dict[str, Run], envelope_width=10., rate_width=40., step=.05):
    """Apply the original matched-tail rate/floor/zero-crossing criteria.

    Spatial changes compare p=32 and p=40 at dt=.1M; time changes compare
    .1M and .05M at p=40.  The plotted fine signal is p=40, dt=.05M.
    A common sampling grid changes no geometric clock or physical profile.
    """
    end = min(r.times[-1] for r in ladder.values())
    times = uniform_grid(0., min(1000., end), step)
    signals = {key: interpolate(r.times, r.u, times) for key, r in ladder.items()}
    envelopes = {key: rms_envelope(times, values, envelope_width, floor_multiplier=0.)
                 for key, values in signals.items()}
    spatial_floor = np.abs(envelopes["medium"] - envelopes["fine"])
    temporal_floor = np.abs(envelopes["fine"] - envelopes["half_dt"])
    floor = np.maximum(spatial_floor, temporal_floor)
    settings = LocalFitSettings(envelope_width=envelope_width, price_window=rate_width,
                                exponential_scaled_window=EXPONENTIAL_SCALED_WINDOW,
                                floor_multiplier=FLOOR_SAFETY_FACTOR)
    kappa = cosmological_rate(640.)
    rates = {key: effective_rates(times, values, settings, kappa=kappa)
             for key, values in signals.items()}
    amplitude, power, gamma = effective_rates(
        times, signals["half_dt"], settings, kappa=kappa, measured_floor=floor)
    spatial_power = np.abs(rates["medium"][1] - rates["fine"][1])
    temporal_power = np.abs(rates["fine"][1] - rates["half_dt"][1])
    supported = (np.isfinite(amplitude) & np.isfinite(power)
                 & np.isfinite(spatial_power) & np.isfinite(temporal_power)
                 & (spatial_power <= RATE_REFINEMENT_TOLERANCE)
                 & (temporal_power <= RATE_REFINEMENT_TOLERANCE)
                 & away_from_zero_crossings(times, signals["half_dt"], rate_width/2))
    spatial_gamma = np.abs(rates["medium"][2] - rates["fine"][2])
    temporal_gamma = np.abs(rates["fine"][2] - rates["half_dt"][2])
    gamma_supported = (np.isfinite(amplitude) & np.isfinite(gamma)
                       & (spatial_gamma <= RATE_REFINEMENT_TOLERANCE)
                       & (temporal_gamma <= RATE_REFINEMENT_TOLERANCE)
                       & away_from_zero_crossings(times, signals["half_dt"],
                                                  EXPONENTIAL_SCALED_WINDOW/(2*kappa)))
    selected = (supported & (times >= ANALYSIS_START)
                & (np.abs(power-PRICE_TARGET) <= PRICE_TOLERANCE))
    coarse_floor = np.abs(envelopes["coarse"]-envelopes["medium"]) if "coarse" in envelopes else np.full_like(times,np.nan)
    return dict(times=times, signal=signals["half_dt"], amplitude=amplitude, power=power,
                power_supported=supported, price_interval=longest_interval(times, selected),
                floor=floor, spatial_floor=spatial_floor, spatial_coarse=coarse_floor, temporal_floor=temporal_floor,
                spatial_power_change=spatial_power, temporal_power_change=temporal_power,
                gamma_over_kappa=gamma, gamma_supported=gamma_supported)


def interval_row(stem: str, diagnostic: dict, envelope_width=10., rate_width=40.):
    interval = diagnostic["price_interval"]
    start, end = interval if interval is not None else (np.nan, np.nan)
    duration = end-start
    return dict(family=stem, envelope_width_over_M=envelope_width,
                rate_fit_width_over_M=rate_width, start_U_over_M=start,
                end_U_over_M=end, duration_over_M=duration,
                passes_price_criterion=bool(duration >= PRICE_MINIMUM_DURATION))


def create_figure(diagnostics: dict, output: Path, intervals: list[dict]):
    """Keep the replaced ringdown figure's 3.4 by 3.5 inch footprint."""
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42,
                         "axes.labelsize": 8, "legend.fontsize": 7.8}):
        figure, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True)
        handles = []
        for family in FAMILIES:
            data = diagnostics[family.stem]
            times = data["times"]
            selection = (times >= 100.) & (times <= 975.)
            amplitude = np.where(selection, data["amplitude"], np.nan)
            power = np.where(selection & data["power_supported"], data["power"], np.nan)
            line, = axes[0].semilogy(times, amplitude, color=family.color,
                                    ls=family.linestyle, lw=1., label=family.label)
            axes[1].plot(times, power, color=family.color, ls=family.linestyle, lw=1.)
            handles.append(line)
        axes[1].axhspan(2.7, 3.3, color="0.9", zorder=-5)
        axes[1].axhline(3., color="0.5", lw=.6, zorder=-4)
        # Use the same interval row as the manuscript table, not plot constants.
        accepted = next(row for row in intervals if row["family"] == "uniform640_conformal")
        duration = float(accepted["duration_over_M"])
        if np.isfinite(duration) and duration >= PRICE_MINIMUM_DURATION:
            start, end = (float(accepted[key]) for key in ("start_U_over_M", "end_U_over_M"))
            for axis in axes:
                axis.axvspan(start, end, facecolor=FAMILIES[2].color, alpha=.13,
                             edgecolor="none", zorder=-6)
            axes[0].text(.08, .075, rf"Uniform conformal: ${duration:.0f}M$",
                         transform=axes[0].transAxes, color="#755800", fontsize=8,
                         ha="left", va="bottom")
        axes[0].set_ylabel(r"RMS amplitude $A$")
        axes[0].set_ylim(1e-10, 5e-4)
        axes[0].set_yticks([1e-4, 1e-6, 1e-8, 1e-10])
        axes[1].set(xlabel=r"retarded time $U/M$", ylabel=r"local index $p_{\rm eff}$",
                    ylim=(-4, 12), yticks=[-4, 0, 3, 6, 9, 12])
        for index, axis in enumerate(axes):
            axis.set_xlim(100, 975)
            axis.set_xticks([200, 400, 600, 800])
            axis.grid(alpha=.2, lw=.5)
            axis.tick_params(labelsize=8, pad=2)
            axis.text(.98 if index == 0 else .02, .92, f"({chr(97+index)})",
                      transform=axis.transAxes, fontsize=8, va="top",
                      ha="right" if index == 0 else "left")
        # Column-major ordering pairs each geometry's two couplings.
        legend_order = (0, 1, 3, None, 2, 4)
        figure.legend([handles[index] if index is not None else Line2D([], [], visible=False)
                       for index in legend_order],
                      [FAMILIES[index].label if index is not None else ""
                       for index in legend_order], loc="upper center",
                      ncol=2, frameon=False, handlelength=1.5, columnspacing=.8,
                      handletextpad=.4, labelspacing=.3)
        figure.subplots_adjust(left=.19, right=.98, bottom=.12, top=.78, hspace=.12)
        figure.savefig(output / "matched_tail_comparison.pdf")
        figure.savefig(output / "matched_tail_comparison.png", dpi=220)
        plt.close(figure)


def build(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    if root not in output.parents or any(output == root/n or root/n in output.parents
                                         for n in ("runs", "operators")):
        raise ValueError("Output must be a derived directory inside the pilot root")
    members = load_members(root)
    diagnostics = {stem: diagnose(ladder) for stem, ladder in members.items()}
    output.mkdir(parents=True, exist_ok=True)
    rows, sweeps, refinement, fits, exponential = [], [], [], [], []
    for family in FAMILIES:
        stem = family.stem
        ladder, data = members[stem], diagnostics[stem]
        rows.append(interval_row(stem, data))
        for envelope in (5., 10., 20.):
            for fit in (30., 40., 60.):
                diagnostic = data if (envelope, fit) == (10., 40.) else diagnose(ladder, envelope, fit)
                sweeps.append(interval_row(stem, diagnostic, envelope, fit))
        for kind, first, second in (("spatial_coarse", "coarse", "medium"),
                                     ("spatial", "medium", "fine"),
                                     ("timestep", "fine", "half_dt")):
            times, difference, reference = exact_difference(ladder[first], ladder[second])
            for start, end in WINDOWS:
                measured = metrics(times, difference, reference, start, end)
                refinement.append(dict(family=stem, comparison=kind, start_U_over_M=start,
                                       end_U_over_M=end, **measured))
        finest = ladder["half_dt"]
        interpolation_delta = interpolation_sensitivity(finest.times, finest.u, data["times"])
        for start, end in WINDOWS:
            measured = metrics(data["times"], interpolation_delta, data["signal"], start, end)
            refinement.append(dict(family=stem, comparison="output_cadence", start_U_over_M=start,
                                   end_U_over_M=end, **measured))
        if family.exterior:
            start, end = (120., 250.) if "minimal" in stem else (160., 400.)
            selected = (data["times"] >= start) & (data["times"] <= end) & np.isfinite(data["amplitude"])
            x, y = np.log(data["times"][selected]), np.log(data["amplitude"][selected])
            slope, intercept = np.polyfit(x, y, 1)
            r2 = 1-np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)
            fits.append(dict(family=stem, start_U_over_M=start, end_U_over_M=end,
                             exponent=-slope, r_squared=r2))
        elif stem != "schwarzschild":
            target = 2. if "conformal" in stem else 1.
            selected = (data["gamma_supported"] & (data["times"] >= ANALYSIS_START)
                        & (np.abs(data["gamma_over_kappa"]-target) <= .1*target))
            interval = longest_interval(data["times"], selected)
            start,end = interval if interval is not None else (np.nan,np.nan)
            minimum = EXPONENTIAL_SCALED_WINDOW/cosmological_rate(640.)
            exponential.append(dict(family=stem, gamma_over_kappa_target=target,
                                    start_U_over_M=start,end_U_over_M=end,
                                    duration_over_M=end-start,
                                    required_duration_over_M=minimum,
                                    passes_exponential_criterion=bool(end-start>=minimum)))
    write_csv(output / "matched_tail_intervals.csv", rows)
    write_csv(output / "matched_tail_estimator_sweep.csv", sweeps)
    write_csv(output / "matched_tail_refinement.csv", refinement)
    write_csv(output / "matched_tail_power_fits.csv", fits)
    write_csv(output / "matched_tail_exponential_candidates.csv", exponential)
    np.savez_compressed(output / "matched_tail_curves.npz",
                        **{f"{stem}_{key}": value for stem, data in diagnostics.items()
                           for key, value in data.items() if isinstance(value, np.ndarray)})
    create_figure(diagnostics, output, rows)
    inputs = {}
    for ladder in members.values():
        for run in ladder.values():
            for name in ("configuration.json", "backend.json", "history.csv"):
                path = run.path / name
                inputs[str(path.relative_to(REPOSITORY))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(
        inputs_sha256=inputs,
        outputs={str((output/name).relative_to(REPOSITORY)):
                 hashlib.sha256((output/name).read_bytes()).hexdigest()
                 for name in ("matched_tail_intervals.csv", "matched_tail_estimator_sweep.csv",
                              "matched_tail_refinement.csv", "matched_tail_power_fits.csv",
                              "matched_tail_exponential_candidates.csv", "matched_tail_curves.npz",
                              "matched_tail_comparison.pdf", "matched_tail_comparison.png")},
        source_sha256={str(path.relative_to(REPOSITORY)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), Path(__file__).with_name("sbp_pilot_analysis.py"),
                                    Path(__file__).with_name("large_l_tail.py"),
                                    Path(__file__).with_name("curvature_coupling_tail_analysis.py"))},
        physical_experiment="Matched ell=1 velocity bump, minimal foliation, U=tau-q, no fit alignment",
        numerical_method="LGL continuous SBP spectral elements, double-double Radau IIA order 3",
        controls="p=24,32,40 at dt=0.1M; p=40 dt=0.05M; final exterior wave-resolved layout",
        estimator=dict(envelope_width=10., power_fit_width=40., fit_width_sweep=[30.,40.,60.],
                       envelope_width_sweep=[5.,10.,20.], floor_factor=10.,
                       index_refinement_tolerance=.1, zero_exclusion_half_fit_width=True,
                       price_tolerance=.3, price_duration=40., analysis_start=80.),
        distinctions="This is the matched L640 criterion, not the stricter L3072 criterion. It does not require reference-index matching, consistent with the archived matched-tail pipeline. Observed refinement changes are not rigorous error bounds.",
        figure="All five signals; common Schwarzschild reference; invalid power samples remain NaN and are not connected; one-column 3.4 x 3.5 inch footprint.",
        intervals=rows,
    )
    (output / "matched_tail_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output_dir or args.root / "analysis"), indent=2))


if __name__ == "__main__":
    main()
