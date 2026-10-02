"""Read-only comparison of independent SBP--Hermite pilot calculations.

The frozen archives and pilot run directories are inputs only.  Derived tables,
plots and a provenance record are written under ``--output-dir``.  This is an
assessment of a new discretization, not a new production result: arithmetic
precision is never used as an estimate of discretization accuracy.

New-run differences use Decimal subtraction on shared nominal tau samples
(printed times are matched to 1e-24 M, below their double-double drift).
Archive comparisons use the prescribed U=tau-q clock, with no fitted time or
amplitude alignment.  Cubic interpolation is accompanied by a half-cadence
sensitivity estimate; that estimate is not a rigorous interpolation bound.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.ndimage import maximum_filter1d

from .large_l_tail import LocalFitSettings, cosmological_rate, effective_rates, rms_envelope


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY / "results/sbp_hermite_pilot_v1"
WINDOWS = ((15., 45.), (150., 300.), (300., 500.), (500., 950.))
PROBES = (200., 300., 500., 900.)
SETTINGS = LocalFitSettings(envelope_width=10., price_window=40.)


def number(value) -> float:
    """Accept both decimal strings and the exact coupling spelling ``1/6``."""
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator)
    return float(value)


def family_key(configuration: dict) -> str:
    background = configuration["background"]
    ell = int(configuration["ell"])
    if background == "schwarzschild":
        return f"schwarzschild_ell{ell}"
    length = number(configuration["L_over_M"])
    coupling = number(configuration["xi"])
    xi = "0" if coupling == 0 else "1o6" if np.isclose(coupling, 1/6) else f"{coupling:g}"
    return f"{background}_L{length:g}_xi{xi}_ell{ell}"


@dataclass
class Run:
    name: str
    path: Path
    configuration: dict
    backend: dict
    tau_decimal: list[Decimal]
    outer_decimal: list[Decimal]
    tau: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @property
    def key(self):
        return family_key(self.configuration)

    @property
    def q(self):
        return number(self.configuration["q"])

    @property
    def times(self):
        return self.tau - self.q

    @property
    def degree(self):
        return int(self.configuration["degree"])

    @property
    def dt(self):
        return number(self.configuration["dt"])

    @property
    def precision(self):
        return self.backend.get("precision", "unknown")

    @property
    def integrator(self):
        description = self.backend.get("method", self.configuration.get("integrator", "unknown")).lower()
        if "radau" in description:
            return "radau3"
        if "hermite" in description:
            return "hermite4"
        return re.sub(r"[^a-z0-9]+", "_", description).strip("_")


def read_run(path: Path) -> Run | None:
    config_path, backend_path = path / "configuration.json", path / "backend.json"
    if not config_path.exists() or not backend_path.exists():
        return None
    try:
        configuration = json.loads(config_path.read_text())
        backend = json.loads(backend_path.read_text())
    except json.JSONDecodeError:
        # A producer can be finishing its metadata write while we scan.
        return None
    if backend.get("completed") is not True or configuration.get("returncode", 0) != 0:
        return None
    with (path / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    times = [Decimal(row["tau"]) for row in rows]
    outer = [Decimal(row["u_right"]) for row in rows]
    tau = np.array([float(value) for value in times])
    u = np.array([float(value) for value in outer])
    v = np.array([float(row["v_right"]) for row in rows])
    if tau.size < 4 or np.any(np.diff(tau) <= 0) or not np.all(np.isfinite([tau, u, v])):
        raise ValueError(f"Nonfinite or nonmonotone completed history: {path}")
    expected = number(configuration["end_tau"])
    if abs(tau[-1] - expected) > 1e-10 * max(1., abs(expected)):
        raise ValueError(f"Completed history does not reach requested endpoint: {path}")
    return Run(path.name, path, configuration, backend, times, outer, tau, u, v)


def archived_path(run: Run) -> Path | None:
    """Select the finest frozen member of the same physical tail experiment."""
    config = run.configuration
    if int(config["ell"]) != 1:
        return None
    background, coupling = config["background"], number(config["xi"])
    if background == "schwarzschild":
        if run.times[-1] > 1000.01:
            relative = "results/large_l_tail/raw/final_schwarzschild_for_L3072_N3072_dt0p0025.npz"
        else:
            relative = "results/curvature_coupling_production_v2/raw/tail/schwarzschild/xi0/schwarzschild/N3072_dt0p0025.npz"
    elif number(config["L_over_M"]) == 3072 and background == "uniform" and coupling == 0:
        relative = "results/large_l_tail/raw/final_sds_L3072_N3072_dt0p0025.npz"
    elif number(config["L_over_M"]) == 640 and (coupling == 0 or np.isclose(coupling, 1/6)):
        xi = "xi0" if coupling == 0 else "xi1o6"
        relative = f"results/curvature_coupling_production_v2/raw/tail/{background}/{xi}/L640/N3072_dt0p0025.npz"
    else:
        return None
    path = REPOSITORY / relative
    return path if path.exists() else None


def read_archive(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        q = float(metadata["retarded_time_offset"]["q"])
        observer = int(np.argmax(archive["observer_rho"]))
        return archive["signal_times"].copy() - q, archive["signals"][:, observer].copy(), metadata


def uniform_grid(start: float, stop: float, step: float):
    return np.arange(np.ceil(start/step), np.floor(stop/step) + 1) * step


def interpolate(times, values, target):
    return CubicSpline(times, values, extrapolate=False)(target)


def interpolation_sensitivity(times, values, target):
    """Full- versus half-cadence spline difference, not a rigorous bound."""
    keep = np.unique(np.r_[np.arange(0, len(times), 2), len(times)-1])
    return np.abs(interpolate(times, values, target) - interpolate(times[keep], values[keep], target))


def metrics(times, difference, reference, start, stop):
    if len(times) < 3:
        return None
    cadence = float(np.median(np.diff(times)))
    if times[0] > start + cadence or times[-1] < stop - cadence:
        return None
    selection = (times >= start) & (times <= stop)
    if np.count_nonzero(selection) < 8:
        return None
    selection &= np.isfinite(difference) & np.isfinite(reference)
    if np.count_nonzero(selection) < 8:
        return None
    t, delta, ref = times[selection], difference[selection], reference[selection]
    if t[0] > start + cadence or t[-1] < stop - cadence:
        return None
    denominator = trapezoid(ref*ref, t)
    if denominator <= 0:
        return None
    return dict(window_start=start, window_end=stop, samples=int(t.size),
                relative_L2=float(np.sqrt(trapezoid(delta*delta, t) / denominator)),
                absolute_Linf=float(np.max(np.abs(delta))),
                relative_Linf=float(np.max(np.abs(delta)) / np.max(np.abs(ref))))


def control_type(first: Run, second: Run) -> str | None:
    if first.key != second.key:
        return None
    attributes = (("spatial", first.degree, second.degree),
                  ("timestep", first.dt, second.dt),
                  ("layout", first.configuration["layout"], second.configuration["layout"]),
                  ("arithmetic", first.precision, second.precision),
                  ("integrator", first.integrator, second.integrator))
    changed = [name for name, a, b in attributes if a != b]
    return changed[0] if len(changed) == 1 else None


def exact_difference(first: Run, second: Run):
    """Subtract 32-digit histories on shared nominal times, without interpolation.

    Repeated additions at different dt can print times differing at 1e-29 M.
    Quantization only matches those samples; it never rounds the field values.
    """
    if abs(first.q-second.q) > 1e-12:
        raise ValueError(f"Same-family clocks differ: {first.name}, {second.name}")
    with localcontext() as context:
        context.prec = 50
        quantum = Decimal("1e-24")
        second_map = {t.quantize(quantum): (t, u) for t, u in zip(second.tau_decimal, second.outer_decimal)}
        shared = []
        for t, u in zip(first.tau_decimal, first.outer_decimal):
            counterpart = second_map.get(t.quantize(quantum))
            if counterpart is not None:
                other_t, other_u = counterpart
                if abs(t-other_t) > quantum:
                    raise ValueError("Nominal sample clocks do not match")
                shared.append((t, u, other_u))
        times = np.array([float(t) - first.q for t, _, _ in shared])
        delta = np.array([float(a-b) for _, a, b in shared])
        reference = np.array([float(b) for _, _, b in shared])
    return times, delta, reference


def raw_index(run: Run):
    """Signed instantaneous index; zeros and their neighbouring samples masked."""
    scale = maximum_filter1d(np.abs(run.u), size=11, mode="nearest")
    valid = (run.times > 0) & (np.abs(run.u) > 1e-10*scale)
    crossing = np.flatnonzero(run.u[:-1] * run.u[1:] <= 0)
    valid[np.unique(np.r_[crossing, crossing+1])] = False
    value = np.full_like(run.u, np.nan)
    value[valid] = -run.times[valid] * run.v[valid] / run.u[valid]
    return value


def rates(run: Run, step: float):
    times = uniform_grid(max(0., run.times[0]), run.times[-1], step)
    signal = interpolate(run.times, run.u, times)
    if len(times)*step < 65:
        missing = np.full_like(times, np.nan)
        return times, signal, missing, missing
    length = run.configuration.get("L_over_M")
    kappa = cosmological_rate(number(length)) if run.configuration["background"] != "schwarzschild" else 1/640
    amplitude, power, _ = effective_rates(times, signal, SETTINGS, kappa=kappa)
    return times, signal, amplitude, power


def longest_interval(times, mask):
    positions = np.flatnonzero(mask)
    if not positions.size:
        return None, None, 0.
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1)+1)
    selected = max(groups, key=lambda group: times[group[-1]]-times[group[0]])
    start, stop = times[selected[0]], times[selected[-1]]
    return float(start), float(stop), float(stop-start)


def finite_value(value):
    return float(value) if np.isfinite(value) else None


def summary_rows(run: Run, rate_data):
    times, _, amplitude, power = rate_data
    raw = raw_index(run)
    rows = []
    for probe in PROBES:
        if probe > run.times[-1]:
            continue
        nearest = int(np.argmin(np.abs(run.times-probe)))
        rate_nearest = int(np.argmin(np.abs(times-probe)))
        rows.append(dict(run=run.name, family=run.key, integrator=run.integrator, U=probe,
                         raw_sample_U=float(run.times[nearest]),
                         u=float(run.u[nearest]), v=float(run.v[nearest]),
                         raw_power_index=finite_value(raw[nearest]),
                         rms_amplitude=finite_value(amplitude[rate_nearest]),
                         rms_fitted_power_index=finite_value(power[rate_nearest]),
                         status="diagnostic; independent refinement must establish accuracy"))
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank(run: Run):
    layout_rank = {"shifted":0, "standard":1, "split-layer":2, "wave-resolved":3}
    return ("double-double" in run.precision, run.degree, -run.dt,
            layout_rank[run.configuration["layout"]], run.times[-1])


def read_exclusions(root: Path) -> dict[str, str]:
    """Read an explicit quarantine list without changing any raw run."""
    path = root / "excluded_runs.json"
    if not path.exists():
        return {}
    exclusions = json.loads(path.read_text())
    if not isinstance(exclusions, dict) or any(
        not isinstance(name, str) or Path(name).name != name
        or not isinstance(reason, str) or not reason.strip()
        for name, reason in exclusions.items()
    ):
        raise ValueError("excluded_runs.json must map run-directory names to nonempty reasons")
    return exclusions


def assess(root: Path, output_dir: Path, step: float = .1):
    root, output_dir = root.resolve(), output_dir.resolve()
    if root not in output_dir.parents or root / "runs" in output_dir.parents or root / "operators" in output_dir.parents or output_dir.name in ("runs", "operators"):
        raise ValueError("Derived output must not overwrite an input directory")
    exclusions = read_exclusions(root)
    runs = [run for path in sorted((root / "runs").glob("*"))
            if path.is_dir() and path.name not in exclusions
            if (run := read_run(path)) is not None]
    if not runs:
        raise ValueError(f"No completed pilot runs under {root / 'runs'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_data = {run.name: rates(run, step) for run in runs}
    comparisons, archive_comparisons, probes, intervals = [], [], [], []
    archived = {}
    for run in runs:
        probes.extend(summary_rows(run, rate_data[run.name]))
        times, _, _, power = rate_data[run.name]
        for tolerance, minimum in ((.3, 40.), (.15, 150.)):
            start, stop, duration = longest_interval(times, (times >= 100) & (np.abs(power-3) <= tolerance))
            intervals.append(dict(run=run.name, family=run.key, integrator=run.integrator, tolerance=tolerance,
                                  minimum_duration=minimum, start=start, end=stop,
                                  duration=duration, duration_only_pass=duration >= minimum,
                                  status="diagnostic only; not the full archived refinement/reference acceptance test"))
        path = archived_path(run)
        if path is None:
            continue
        if path not in archived:
            archived[path] = read_archive(path)
        old_times, old_u, _ = archived[path]
        common = uniform_grid(max(0., run.times[0], old_times[0]), min(run.times[-1], old_times[-1]), step)
        old = interpolate(old_times, old_u, common)
        new = interpolate(run.times, run.u, common)
        new_envelope = rms_envelope(common, new, 10., floor_multiplier=0.)
        old_envelope = rms_envelope(common, old, 10., floor_multiplier=0.)
        interpolation_floor = (interpolation_sensitivity(old_times, old_u, common)
                               + interpolation_sensitivity(run.times, run.u, common))
        for start, stop in WINDOWS:
            result = metrics(common, new-old, old, start, stop)
            sensitivity = metrics(common, interpolation_floor, old, start, stop)
            envelope_comparison = metrics(common, new_envelope-old_envelope, old_envelope, start, stop)
            if result:
                archive_comparisons.append(dict(run=run.name, family=run.key, integrator=run.integrator,
                    archive=path.relative_to(REPOSITORY).as_posix(), **result,
                    interpolation_sensitivity_relative_L2=sensitivity["relative_L2"],
                    rms_envelope_relative_L2=envelope_comparison["relative_L2"] if envelope_comparison else None,
                    float64_subtraction_scale=8*np.finfo(float).eps,
                    interpretation="independent-code difference, not an absolute error estimate"))
    for i, first in enumerate(runs):
        for second in runs[i+1:]:
            kind = control_type(first, second)
            if kind is None:
                continue
            if rank(first) > rank(second):
                coarse, fine = second, first
            else:
                coarse, fine = first, second
            times, delta, reference = exact_difference(coarse, fine)
            if len(times) < 120 or times[-1]-times[0] < 15:
                continue
            fine_envelope = rms_envelope(times, reference, 10., floor_multiplier=0.)
            coarse_envelope = rms_envelope(times, reference+delta, 10., floor_multiplier=0.)
            for start, stop in WINDOWS:
                result = metrics(times, delta, reference, start, stop)
                if result:
                    envelope_comparison = metrics(times, coarse_envelope-fine_envelope, fine_envelope, start, stop)
                    comparisons.append(dict(family=first.key, comparison=kind,
                        candidate_integrator=coarse.integrator, reference_integrator=fine.integrator,
                        candidate=coarse.name, reference=fine.name, **result,
                        rms_envelope_relative_L2=envelope_comparison["relative_L2"] if envelope_comparison else None,
                        subtraction="50-digit Decimal on shared nominal tau within 1e-24 M; no interpolation"))
    write_csv(output_dir / "independent_controls.csv", comparisons)
    write_csv(output_dir / "archived_waveform_differences.csv", archive_comparisons)
    write_csv(output_dir / "power_index_samples.csv", probes)
    write_csv(output_dir / "diagnostic_price_intervals.csv", intervals)
    families = sorted({(run.key, run.integrator) for run in runs})
    selected = [max((run for run in runs if (run.key, run.integrator) == key), key=rank) for key in families]
    for run in selected:
        times, new, envelope, power = rate_data[run.name]
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True, constrained_layout=True)
        axes[0].semilogy(times, np.abs(new), label=f"SBP {run.integrator} p={run.degree}, dt={run.dt:g}")
        axes[1].plot(run.times, raw_index(run), lw=.7, alpha=.4, label=r"SBP signed $-Uv/u$")
        axes[1].plot(times, power, lw=1.5, label="SBP 10M RMS / 40M fit")
        axes[1].axhspan(2.7, 3.3, color="0.9", zorder=-5, label="3 ± 10%")
        axes[1].axhline(3, color="0.4", lw=.7)
        axes[1].set_ylim(-.5, 8)
        path = archived_path(run)
        if path:
            old_times, old_u, _ = archived[path]
            old = interpolate(old_times, old_u, times)
            valid = np.isfinite(old)
            old_amplitude, old_power = np.full_like(times, np.nan), np.full_like(times, np.nan)
            if np.count_nonzero(valid)*step >= 65:
                length = run.configuration.get("L_over_M")
                kappa = cosmological_rate(number(length)) if run.configuration["background"] != "schwarzschild" else 1/640
                old_amplitude[valid], old_power[valid], _ = effective_rates(times[valid], old[valid], SETTINGS, kappa=kappa)
            axes[0].semilogy(times, np.abs(old), "--", lw=1., label="Frozen Chebyshev N=3072")
            axes[1].plot(times, old_power, "--", label="Frozen 10M RMS / 40M fit")
            scale = old_amplitude.copy()
            scale[scale <= 0] = np.nan
            sensitivity = (interpolation_sensitivity(old_times, old_u, times)
                           + interpolation_sensitivity(run.times, run.u, times))
            axes[2].semilogy(times, np.abs(new-old)/scale, label="|SBP − frozen| / frozen RMS")
            axes[2].semilogy(times, sensitivity/scale, ":", label="Interpolation sensitivity")
        for companion in sorted(runs, key=rank, reverse=True):
            if control_type(companion, run) == "spatial" and companion.degree < run.degree:
                t, delta, _ = exact_difference(companion, run)
                denominator = np.interp(t, times, envelope)
                axes[2].semilogy(t, np.abs(delta)/denominator, alpha=.7,
                                label=f"SBP |p{companion.degree} − p{run.degree}| / RMS")
                break
        axes[0].set_ylabel(r"Outer-boundary $|u|$")
        axes[1].set_ylabel("Local power index")
        axes[2].set_ylabel("Relative difference")
        axes[2].set_xlabel(r"Fixed retarded time $U/M$")
        axes[2].set_xlim(100 if times[-1] > 200 else 0, times[-1])
        for axis in axes:
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7, loc="best")
        figure_stem = f"{run.key}_{run.integrator}"
        fig.suptitle(figure_stem.replace("_", " ") + "\nPilot comparison: diagnostic, not a production replacement", fontsize=10)
        fig.savefig(output_dir / f"{figure_stem}.pdf")
        fig.savefig(output_dir / f"{figure_stem}.png", dpi=170)
        plt.close(fig)
    inventory = [dict(run=run.name, family=run.key, degree=run.degree,
                      nodes=run.configuration["nodes"], dt=run.dt,
                      layout=run.configuration["layout"], precision=run.precision, integrator=run.integrator,
                      end_U=float(run.times[-1]), endpoint_u=float(run.u[-1]),
                      endpoint_v=float(run.v[-1]),
                      elapsed_seconds=run.backend.get("total_seconds")) for run in runs]
    write_csv(output_dir / "run_inventory.csv", inventory)
    provenance = dict(
        status="independent pilot analysis; frozen archives unchanged",
        estimator=dict(envelope_width=10., power_fit_width=40.,
                       roundoff_floor_multiplier=100., uniform_U_step=step),
        numerical_warning="Raw indices and diagnostic Price durations do not establish convergence. Decimal differences avoid float64 cancellation but do not prove 31-digit accuracy.",
        interpolation_warning="Half-cadence cubic spline differences estimate interpolation sensitivity, not a rigorous bound.",
        inputs={str(run.path.relative_to(REPOSITORY) if run.path.is_relative_to(REPOSITORY) else run.path):
                    {name: hashlib.sha256((run.path/name).read_bytes()).hexdigest()
                     for name in ("configuration.json", "backend.json", "history.csv")} for run in runs},
        frozen_inputs={str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest() for path in archived},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        excluded_runs=exclusions,
        exclusion_list_sha256=(hashlib.sha256((root / "excluded_runs.json").read_bytes()).hexdigest()
                               if exclusions else None),
        figure_files=[f"{run.key}_{run.integrator}.{extension}"
                      for run in selected for extension in ("pdf", "png")],
        completed_runs=len(runs), independent_control_rows=len(comparisons),
        archived_comparison_rows=len(archive_comparisons))
    (output_dir / "analysis_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    lines = ["# Independent SBP–Hermite pilot assessment", "",
             "No production archive or manuscript figure was changed.", "",
             "The new solver is tested against independent spatial, timestep, layout, and arithmetic controls where available. Missing controls are not inferred from high precision.", "",
             "Waveforms use U=tau-q with no fitted shift or amplitude rescaling. New-run differences are subtracted as Decimal values on shared nominal tau samples (printed times matched within 1e-24 M). Archive comparisons use cubic interpolation and report the full/half-cadence sensitivity separately.", "",
             "The 10M RMS / 40M logarithmic-fit estimator is reused unchanged. The signed instantaneous index -Uv/u is additional diagnostic information. Diagnostic band durations are not the full archived acceptance criterion: they do not include a reference comparison and measured refinement floor.", "",
             "## Selected finest pilot members", ""]
    for run in selected:
        kinds = sorted({row["comparison"] for row in comparisons if row["family"] == run.key})
        lines.append(f"- {run.name}: p={run.degree}, {run.configuration['nodes']} nodes, dt={run.dt:g}, {run.precision}, {run.integrator}; U ends at {run.times[-1]:g}. Available control types: {', '.join(kinds) or 'none'}. See `{run.key}_{run.integrator}.pdf`.")
    if exclusions:
        lines.extend(["", "## Explicitly excluded pilot runs", "",
                      "These raw histories remain untouched but are excluded from all tables, comparisons, and figure selection. Only figures listed in this report and its manifest belong to this assessment; older unlisted derived files are not evidence.", ""])
        lines.extend(f"- `{name}`: {reason}" for name, reason in sorted(exclusions.items()))
    lines.extend(["", "## Reading the tables", "",
                  "- `independent_controls.csv`: one factor changed at a time; observed differences, not Richardson error estimates.",
                  "- `archived_waveform_differences.csv`: relative L2 differences to the finest frozen matched waveform. Values are fractions, not percentages.",
                  "- `power_index_samples.csv`: instantaneous and archived-estimator indices at U=200, 300, 500, and 900 where available.",
                  "- `diagnostic_price_intervals.csv`: the longest unrefined fit interval in each band, starting after U=100. These are not publication acceptance intervals.",
                  "- `run_inventory.csv`: completed runs, endpoint amplitudes, precision, and timing.",
                  "", "Replacing a figure requires resolved spatial, temporal, and interface/layout accuracy, not just a smoother curve or agreement of two high-precision outputs.", ""])
    (output_dir / "README.md").write_text("\n".join(lines))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-step", type=float, default=.1)
    args = parser.parse_args()
    if args.sample_step <= 0:
        parser.error("--sample-step must be positive")
    output = args.output_dir or args.root / "analysis"
    report = assess(args.root, output, args.sample_step)
    print(json.dumps({key: report[key] for key in
                     ("completed_runs", "independent_control_rows", "archived_comparison_rows")}, indent=2))


if __name__ == "__main__":
    main()
