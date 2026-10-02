"""Compact, independently refined summary of the SBP Radau tail pilot.

This deliberately selects only the final double-double Radau3 members and
their matched p/time controls.  Exterior controls use the same selected
layout (wave-resolved by default).  Quarantined runs and unsuccessful Hermite diagnostics cannot
enter the summary.  No frozen archive or manuscript asset is modified.
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
import numpy as np

from .large_l_tail import cosmological_rate, effective_rates
from .sbp_pilot_analysis import (
    DEFAULT_ROOT, REPOSITORY, SETTINGS, Run, archived_path, exact_difference,
    interpolate, interpolation_sensitivity, longest_interval, metrics, number,
    raw_index, read_archive, read_exclusions, read_run, uniform_grid, write_csv,
)


@dataclass(frozen=True)
class Flagship:
    stem: str
    label: str
    color: str
    exterior: bool = False

    def name(self, degree: int, dt: str, exterior_layout: str = "wave-resolved"):
        suffix = {"split-layer": "split", "wave-resolved": "wave"}[exterior_layout]
        layout = f"_{suffix}" if self.exterior else ""
        return f"{self.stem}_p{degree}{layout}_radau3_dt{dt}_U1000"


FAMILIES = (
    Flagship("schwarzschild", "Schwarzschild", "#333333"),
    Flagship("uniform3072_minimal", r"Uniform $L/M=3072$, $\xi=0$", "#0072B2"),
    Flagship("uniform640_conformal", r"Uniform $L/M=640$, $\xi=1/6$", "#E69F00"),
    Flagship("exterior640_minimal", r"Exterior $L/M=640$, $\xi=0$", "#882255", True),
    Flagship("exterior640_conformal", r"Exterior $L/M=640$, $\xi=1/6$", "#009E73", True),
)
WINDOWS = ((150., 300.), (300., 500.), (500., 950.))


def load_members(root: Path, exterior_layout: str = "wave-resolved"):
    exclusions = read_exclusions(root)
    members, missing = {}, []
    for family in FAMILIES:
        ladder = {}
        for role, degree, dt in (("medium", 32, "0p1"),
                                 ("fine", 40, "0p1"),
                                 ("half_dt", 40, "0p05")):
            name = family.name(degree, dt, exterior_layout)
            if name in exclusions:
                raise ValueError(f"Required flagship is quarantined: {name}: {exclusions[name]}")
            run = read_run(root / "runs" / name)
            if run is None:
                missing.append(name)
                continue
            if run.integrator != "radau3" or run.precision != "double-double":
                raise ValueError(f"Flagship must use double-double Radau3: {name}")
            if run.degree != degree or not np.isclose(run.dt, float(dt.replace("p", "."))):
                raise ValueError(f"Flagship name and metadata disagree: {name}")
            if run.times[-1] < 999.9 or int(run.configuration["ell"]) != 1:
                raise ValueError(f"Flagship must be a completed ell=1 tail to U=1000: {name}")
            expected_layout = exterior_layout if family.exterior else "standard"
            if run.configuration["layout"] != expected_layout:
                raise ValueError(f"Wrong layout for {name}: expected {expected_layout}")
            ladder[role] = run
        if len(ladder) == 3:
            if len({run.configuration["layout"] for run in ladder.values()}) != 1:
                raise ValueError(f"Mixed layouts in a nominal p/time control: {family.stem}")
            if len({run.key for run in ladder.values()}) != 1:
                raise ValueError(f"Mixed physical families in control: {family.stem}")
            members[family.stem] = ladder
    if missing:
        raise FileNotFoundError("Required flagship runs are not yet complete:\n" + "\n".join(missing))
    return members


def fitted(run: Run, times: np.ndarray):
    signal = interpolate(run.times, run.u, times)
    config = run.configuration
    kappa = (1/640 if config["background"] == "schwarzschild"
             else cosmological_rate(number(config["L_over_M"])))
    amplitude, power, _ = effective_rates(times, signal, SETTINGS, kappa=kappa)
    return dict(u=signal, amplitude=amplitude, power=power)


def refinement_diagnostics(ladder: dict, times: np.ndarray):
    fitted_members = {name: fitted(run, times) for name, run in ladder.items()}
    finest = fitted_members["half_dt"]
    space = np.abs(fitted_members["medium"]["amplitude"] - fitted_members["fine"]["amplitude"])
    temporal = np.abs(fitted_members["fine"]["amplitude"] - finest["amplitude"])
    amplitude_floor = np.maximum(space, temporal)
    index_change = np.maximum(
        np.abs(fitted_members["medium"]["power"] - fitted_members["fine"]["power"]),
        np.abs(fitted_members["fine"]["power"] - finest["power"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_envelope_change = amplitude_floor / finest["amplitude"]
    resolved = (np.isfinite(finest["power"])
                & (relative_envelope_change <= .01)
                & (index_change <= .1)
                & (finest["amplitude"] > 10*amplitude_floor))
    return dict(**finest, envelope_change=relative_envelope_change,
                index_change=index_change, resolved=resolved)


def price_interval(times, candidate, reference, tolerance: float):
    """A pilot consistency gate, not a replacement of the archived protocol."""
    admissible = ((times >= 100.) & (times <= 975.)
                  & (np.abs(candidate["power"]-3) <= tolerance)
                  & (np.abs(reference["power"]-3) <= tolerance)
                  & (np.abs(candidate["power"]-reference["power"]) <= tolerance)
                  & candidate["resolved"] & reference["resolved"])
    return longest_interval(times, admissible)


def finite_sample(times, values, time):
    index = int(np.argmin(np.abs(times-time)))
    value = values[index]
    return float(value) if np.isfinite(value) else None


def build(root: Path, output: Path, step: float = .05, *, exterior_layout: str = "wave-resolved"):
    root, output = root.resolve(), output.resolve()
    if root not in output.parents or any(output == root/name or root/name in output.parents for name in ("runs", "operators")):
        raise ValueError("Flagship output must be a derived directory inside the pilot root")
    members = load_members(root, exterior_layout)
    output.mkdir(parents=True, exist_ok=True)
    common_end = min(run.times[-1] for ladder in members.values() for run in ladder.values())
    times = uniform_grid(0., min(1000., common_end), step)
    diagnostic = {name: refinement_diagnostics(ladder, times) for name, ladder in members.items()}
    refinement_window = (times >= 150.) & (times <= 950.)
    fully_resolved = {name: bool(np.all(data["resolved"][refinement_window]))
                      for name, data in diagnostic.items()}
    reference = diagnostic["schwarzschild"]
    headline, intervals, samples, plot_differences = [], [], [], {}
    archive_inputs = set()
    for family in FAMILIES:
        ladder, data = members[family.stem], diagnostic[family.stem]
        controls = {}
        for kind, first, second in (("spatial", "medium", "fine"), ("timestep", "fine", "half_dt")):
            control_times, delta, control_reference = exact_difference(ladder[first], ladder[second])
            controls[kind] = (control_times, delta, control_reference)
        finest = ladder["half_dt"]
        raw = raw_index(finest)
        archive = archived_path(finest)
        if archive is None:
            raise FileNotFoundError(f"No matched frozen archive for {family.stem}")
        archive_inputs.add(archive)
        old_t, old_u, _ = read_archive(archive)
        old = interpolate(old_t, old_u, times)
        config = finest.configuration
        kappa = (1/640 if config["background"] == "schwarzschild"
                 else cosmological_rate(number(config["L_over_M"])))
        old_amplitude, old_power, _ = effective_rates(times, old, SETTINGS, kappa=kappa)
        difference = data["u"]-old
        interpolation_change = (interpolation_sensitivity(old_t, old_u, times)
                                + interpolation_sensitivity(finest.times, finest.u, times))
        for start, end in WINDOWS:
            row = dict(family=family.stem, window_start_M=start, window_end_M=end,
                       selected_run=finest.name, nodes=finest.configuration["nodes"])
            for kind, (t, delta, ref) in controls.items():
                result = metrics(t, delta, ref, start, end)
                if result is None:
                    raise ValueError(f"Missing full {kind} comparison window for {family.stem}")
                row[f"{kind}_relative_L2_percent"] = 100*result["relative_L2"]
            row["archived_relative_L2_percent"] = 100*metrics(times, difference, old, start, end)["relative_L2"]
            row["archived_rms_envelope_relative_L2_percent"] = 100*metrics(
                times, data["amplitude"]-old_amplitude, old_amplitude, start, end)["relative_L2"]
            row["interpolation_sensitivity_relative_L2_percent"] = 100*metrics(times, interpolation_change, old, start, end)["relative_L2"]
            window = (times >= start) & (times <= end)
            row["max_envelope_refinement_percent"] = 100*float(np.nanmax(data["envelope_change"][window]))
            row["max_index_refinement"] = float(np.nanmax(data["index_change"][window]))
            row["archived_fitted_index_max_difference"] = float(np.nanmax(
                np.abs(data["power"][window]-old_power[window])))
            native_window = (finest.times >= start) & (finest.times <= end) & np.isfinite(raw)
            row["minimum_raw_index"] = float(np.min(raw[native_window]))
            row["maximum_raw_index"] = float(np.max(raw[native_window]))
            headline.append(row)
        for tolerance, duration in ((.15, 150.), (.3, 40.)):
            start, end, length = price_interval(times, data, reference, tolerance)
            intervals.append(dict(family=family.stem, absolute_index_tolerance=tolerance,
                                  minimum_duration_M=duration, start_U_M=start, end_U_M=end,
                                  continuous_duration_M=length, meets_pilot_gate=length >= duration,
                                  refinement_resolved_through_U150_950=fully_resolved[family.stem],
                                  status="pilot consistency with archived claim; not replacement of full production protocol"))
        for time in (200., 300., 500., 900.):
            samples.append(dict(family=family.stem, U_M=time,
                                raw_power_index=finite_sample(finest.times, raw, time),
                                fitted_power_index=finite_sample(times, data["power"], time),
                                u=finite_sample(finest.times, finest.u, time)))
        spacetime = []
        for kind, (t, delta, _) in controls.items():
            # Positive difference curves only; interpolating a signed, tiny
            # difference is avoided.  Decimal subtraction happened upstream.
            valid = (t >= 90.) & (t <= 960.)
            values = np.interp(times, t[valid], np.abs(delta[valid]), left=np.nan, right=np.nan)
            spacetime.append(values / data["amplitude"])
        plot_differences[family.stem] = np.maximum(*spacetime)
    write_csv(output / "flagship_window_errors.csv", headline)
    write_csv(output / "flagship_price_intervals.csv", intervals)
    write_csv(output / "flagship_power_indices.csv", samples)

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5}):
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), constrained_layout=True)
        for family in FAMILIES:
            finest = members[family.stem]["half_dt"]
            panel = axes[1] if family.exterior else axes[0]
            label = family.label + (" (refinement unresolved)" if not fully_resolved[family.stem] else "")
            panel.plot(finest.times, raw_index(finest), color=family.color, lw=1.35, label=label)
            selection = (times >= 100.) & (times <= 950.)
            axes[2].semilogy(times[selection], plot_differences[family.stem][selection],
                             color=family.color, lw=1.05, label=family.label)
        schwarzschild = members["schwarzschild"]["half_dt"]
        axes[1].plot(schwarzschild.times, raw_index(schwarzschild), color="#555555", ls="--", lw=1., label="Schwarzschild reference")
        for axis in axes[:2]:
            axis.axhspan(2.7, 3.3, color="0.9", zorder=-5)
            axis.axhline(3., color="0.5", lw=.7, zorder=-4)
            axis.set_ylim(0., 8.)
            axis.set_ylabel(r"Instantaneous $p=-U\,\partial_\tau u/u$")
            axis.legend(loc="upper left")
        axes[0].set_title("(a) Uniform artificial cosmology: recovery and departure")
        exterior_unresolved = any(not fully_resolved[family.stem] for family in FAMILIES if family.exterior)
        axes[1].set_title("(b) Exterior-supported cosmology: " +
                          ("refinement remains unresolved" if exterior_unresolved else "resolved pilot comparison"))
        axes[2].set_title("(c) Independent spatial and timestep differences")
        axes[2].set_ylabel("Larger observed difference / 10M RMS")
        axes[2].legend(ncol=2, loc="best")
        for axis in axes:
            axis.set_xlim(100., 1000.)
            axis.grid(alpha=.2)
            axis.set_xlabel(r"Fixed retarded time $U/M$")
        fig.savefig(output / "flagship_tail_overview.pdf")
        fig.savefig(output / "flagship_tail_overview.png", dpi=180)
        plt.close(fig)

    files = {}
    for ladder in members.values():
        for run in ladder.values():
            for filename in ("configuration.json", "backend.json", "history.csv"):
                path = run.path / filename
                files[path.relative_to(REPOSITORY).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in archive_inputs:
        files[path.relative_to(REPOSITORY).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = [Path(__file__), Path(__file__).with_name("sbp_pilot_analysis.py"), Path(__file__).with_name("large_l_tail.py")]
    manifest = dict(
        scope="independent pilot; no manuscript or frozen archive modified",
        exterior_layout=exterior_layout,
        exterior_layout_description=(
            "Transition-layer subdivisions plus areal-radius boundaries r/M=80,120,160,200 to resolve the incoming wave before the transition; physical profile unchanged."
            if exterior_layout == "wave-resolved" else
            "Transition-layer subdivisions only; no additional pretransition wave-resolution blocks."),
        selected={name: {role: run.name for role, run in ladder.items()} for name, ladder in members.items()},
        excluded_runs=read_exclusions(root), input_sha256=files,
        refinement_resolved_through_U150_950=fully_resolved,
        source_sha256={path.relative_to(REPOSITORY).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        estimator=dict(uniform_U_cadence=step, rms_width_M=10., fit_width_M=40.,
                       envelope_relative_refinement_limit=.01, index_refinement_limit=.1,
                       floor_safety_factor=10., earliest_U_M=100., latest_U_M=975.),
        figure="Raw instantaneous indices; no temporal averaging in panels (a) and (b). Near-zero samples masked. Axis range 0 to 8; excursions outside that range are not claims of missing data.",
        caveat="Band durations are pilot consistency gates, not repetitions of every original production sensitivity test. Observed numerical differences are not absolute error bounds. No fitted alignment or amplitude rescaling.",
    )
    (output / "flagship_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    notes = ["# Refined SBP–Radau tail pilot", "",
             f"The overview uses p=40, dt=0.05M double-double Radau3 waveforms. Exterior calculations use the {exterior_layout} layout. Spatial controls compare p=32 and p=40 at dt=0.1M on that same layout; timestep controls compare dt=0.1M and 0.05M at p=40.", "",
             manifest["exterior_layout_description"], "",
             "The upper two panels show the signed instantaneous index -U v/u, not an RMS fit or a smoothed derivative. Zeros and immediately adjacent samples are masked. Excursions outside the displayed 0–8 range are clipped by the axes, not used to infer tail intervals. The shaded band is 3 ± 10%.", "",
             "The bottom panel takes the larger observed spatial or timestep waveform difference and divides by the finest 10M RMS amplitude. This is an observed comparison, not a proved error bound. The CSV lists the independent spatial and temporal relative L2 differences separately.", "",
             "The legacy comparison table reports both the signed-waveform L2 difference and the 10M RMS-envelope L2 difference from the actual finest frozen archive. It also reports the maximum difference between their 40M fitted indices. The old waveform is not replaced with a theoretical tail or a refitted curve.", "",
             "Interval tests retain the archived 10M RMS / 40M logarithmic-fit estimator. A candidate and the independently computed Schwarzschild control must both lie within the specified band around 3 and agree with each other to the same absolute tolerance. Both require ≤1% envelope refinement change, ≤0.1 index refinement change, and amplitude exceeding ten times the observed envelope-refinement floor. Durations are measured on a common U grid through U=975M. These are pilot consistency checks, not replacements for the complete archived protocol or its parameter sweeps.", "",
             "Curves that fail the numerical refinement gate anywhere over U=150–950M are explicitly labelled unresolved. A failed Price-interval test for such a curve is not by itself independent evidence of physical failure. The exterior comparison must not be promoted to a replacement production result while this sensitivity remains.", "",
             "Quarantined preliminary backend outputs and Hermite stiff-mode diagnostics are excluded. No time translation or amplitude rescaling is fitted. High arithmetic precision alone does not establish spatial or temporal accuracy.", ""]
    (output / "flagship_notes.md").write_text("\n".join(notes))
    return {"families": len(members), "completed_runs": sum(map(len, members.values())),
            "window_comparisons": len(headline), "price_gates": intervals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exterior-layout", choices=("wave-resolved", "split-layer"), default="wave-resolved")
    args = parser.parse_args()
    default_directory = "flagship" if args.exterior_layout == "wave-resolved" else "flagship_split_layer"
    report = build(args.root, args.output_dir or args.root / default_directory,
                   exterior_layout=args.exterior_layout)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
