"""Final Schwarzschild-to-de Sitter tail crossover analysis.

This module replaces the earlier "closer to" crossover classification with a
*transition interval*: the crossover is bracketed by

* a departure time, the end of the last interval over which the finite-``L``
  local decay rate still agrees with the Schwarzschild rate measured in an
  independent evolution with identical physical initial data, and
* an entry time, the start of the first interval over which the local rate
  enters and stays inside a tolerance of the cosmological rate
  ``gamma/kappa_c = ell``.

Both times are reported as ranges obtained by sweeping the amplitude-envelope
width, the persistence width, and the tolerance, so that the quoted crossover
carries a systematic uncertainty instead of a single high-precision number.
The primary time variable is ``kappa_c U``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from .sds_model import SdSParameters, sds_horizons
from .sds_result import load_sds_result
from .tail_analysis import json_safe, numerical_amplitude_floor


ELL1_LENGTHS = (20.0, 40.0, 80.0, 160.0)
SDS_OBSERVERS: tuple[float | None, ...] = (4.0, 8.0, 16.0, None)
PRIMARY_RADIUS = 8.0

# The initial pulse is centered at r=6M and supported on 3M<r<9M, so no
# observer carries a physically meaningful tail before this retarded time.
# The cut is imposed in M rather than in kappa_c U so that the same physical
# transient is excluded from every cosmological length.
MINIMUM_RETARDED_TIME = 30.0

OBSERVER_COLORS = {
    4.0: "#6a3d9a",
    8.0: "#1f78b4",
    16.0: "#33a02c",
    None: "#000000",
}
RESOLUTION_WEIGHTS = {
    1024: (3.0, 0.40),
    1536: (2.1, 0.60),
    2048: (1.4, 0.85),
    3072: (0.9, 1.00),
}
RESOLUTION_COLORS = {
    1024: "#e08214",
    1536: "#8073ac",
    2048: "#1f78b4",
    3072: "#08306b",
    4096: "#1f78b4",
}


def observer_label(radius: float | None) -> str:
    """Return the plotting label of one observer."""

    if radius is None:
        return r"$\mathcal{H}_c^{+}$"
    return rf"$r/M={radius:g}$"


def observer_key(radius: float | None) -> str:
    """Return the tabular key of one observer."""

    return "cosmological_horizon" if radius is None else f"r/M={radius:g}"


# --------------------------------------------------------------------------
# Local decay rates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeSettings:
    """Widths, in units of ``M``, of the local rate estimator."""

    smoothing_width: float = 30.0
    rms_fraction: float = 0.5

    @property
    def rms_width(self) -> float:
        return self.rms_fraction * self.smoothing_width


def _odd_window(width: float, step: float) -> int:
    samples = max(7, int(round(width / step)))
    if samples % 2 == 0:
        samples += 1
    return samples


def envelope_rate(
    times: np.ndarray,
    signal: np.ndarray,
    settings: EnvelopeSettings,
    *,
    floor_multiplier: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``gamma_eff = -d ln A/dU`` of a sliding RMS envelope ``A``.

    Differentiating ``log|u|`` directly is singular at every quasinormal-mode
    zero crossing.  The centered root-mean-square envelope is phase
    insensitive and preserves a slowly varying power-law or exponential rate.
    Samples whose envelope has reached the double-precision amplitude floor,
    and samples inside the half-widths of the two filters, are returned as
    ``nan``.
    """

    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if times.shape != signal.shape or times.ndim != 1:
        raise ValueError("Times and signal must be one-dimensional peers.")
    if times.size < 16:
        raise ValueError("The time series is too short for a local rate.")
    step = float(np.median(np.diff(times)))
    rms_samples = _odd_window(settings.rms_width, step)
    smoothing_samples = _odd_window(settings.smoothing_width, step)
    if rms_samples + smoothing_samples >= times.size:
        raise ValueError("The requested filters are longer than the signal.")

    kernel = np.full(rms_samples, 1.0 / rms_samples)
    envelope = np.sqrt(np.convolve(signal**2, kernel, mode="same"))
    floor = numerical_amplitude_floor(signal)
    smooth = savgol_filter(
        np.log(np.maximum(envelope, floor)),
        smoothing_samples,
        3,
        mode="interp",
    )
    rate = -np.gradient(smooth, times)

    resolved = envelope > floor_multiplier * floor
    edge = rms_samples // 2 + smoothing_samples // 2
    resolved[:edge] = False
    resolved[-edge:] = False
    rate[~resolved] = np.nan
    envelope[~resolved] = np.nan
    return rate, envelope


def observer_index(result, radius: float | None) -> int:
    """Return the recorded observer index closest to one areal radius."""

    if radius is None:
        return int(np.argmax(result.observer_rho))
    finite = np.flatnonzero(np.isfinite(result.observer_areal_radius))
    offsets = np.abs(result.observer_areal_radius[finite] - radius)
    return int(finite[int(np.argmin(offsets))])


def retarded_series(result, radius: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the geometric retarded time ``U`` and signal of one observer."""

    offset = float(result.metadata["retarded_time_offset"]["q"])
    times = np.asarray(result.signal_times, dtype=float) - offset
    signal = np.asarray(result.signals[:, observer_index(result, radius)], dtype=float)
    return times, signal


# --------------------------------------------------------------------------
# Transition interval
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionInterval:
    """Bracketing times of one Schwarzschild-to-SdS rate transition."""

    status: str
    departure: float
    entry: float
    final_rate: float
    resolved_end: float

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved"


def _sustained_start_indices(flag: np.ndarray, count: int) -> np.ndarray:
    """Return indices at which ``count`` consecutive samples are all true."""

    if count <= 1:
        return np.flatnonzero(flag)
    padded = np.concatenate(([0], np.cumsum(flag.astype(int))))
    totals = padded[count:] - padded[:-count]
    return np.flatnonzero(totals == count)


def transition_interval(
    scaled_time: np.ndarray,
    normalized_rate: np.ndarray,
    reference_rate: np.ndarray,
    ell: int,
    *,
    tolerance: float = 0.10,
    persistence: float = 0.25,
    minimum_scaled_time: float = 0.35,
    tail_fraction: float = 0.80,
) -> TransitionInterval:
    r"""Bracket the crossover between two persistently identified regimes.

    ``scaled_time`` is ``kappa_c U``, ``normalized_rate`` is
    ``gamma_eff/kappa_c`` of the finite-``L`` evolution, and
    ``reference_rate`` is the same quantity measured at the same observer in
    the Schwarzschild evolution with identical physical initial data.

    The entry time is the beginning of the first interval of width
    ``persistence`` inside which ``|gamma_eff/kappa_c - ell| <= tolerance*ell``,
    provided the same tolerance also holds over at least ``tail_fraction`` of
    the remaining resolved samples.  The departure time is the end of the last
    interval of width ``persistence``, before that entry, over which the
    finite-``L`` rate still agrees with the Schwarzschild rate to within
    ``tolerance`` of ``max(|reference|, ell)``.  Both must exist for the
    transition to be called resolved.
    """

    scaled_time = np.asarray(scaled_time, dtype=float)
    normalized_rate = np.asarray(normalized_rate, dtype=float)
    reference_rate = np.asarray(reference_rate, dtype=float)
    if not (scaled_time.shape == normalized_rate.shape == reference_rate.shape):
        raise ValueError("Times, rates, and reference rates must have one shape.")
    if scaled_time.ndim != 1 or scaled_time.size < 3:
        raise ValueError("At least three one-dimensional samples are required.")
    if np.any(np.diff(scaled_time) <= 0.0):
        raise ValueError("Scaled time must be strictly increasing.")
    if ell <= 0:
        raise ValueError("The cosmological rate target requires ell > 0.")
    if tolerance <= 0.0 or persistence <= 0.0:
        raise ValueError("The tolerance and persistence width must be positive.")

    valid = np.isfinite(normalized_rate) & (scaled_time >= minimum_scaled_time)
    if not np.any(valid):
        return TransitionInterval("no_resolved_rate", np.nan, np.nan, np.nan, np.nan)

    scale = np.maximum(np.abs(reference_rate), float(ell))
    schwarzschild_like = (
        valid
        & np.isfinite(reference_rate)
        & (np.abs(normalized_rate - reference_rate) <= tolerance * scale)
    )
    sds_like = valid & (np.abs(normalized_rate - ell) <= tolerance * ell)

    step = float(np.median(np.diff(scaled_time)))
    count = max(2, int(round(persistence / step)))
    resolved_end = float(scaled_time[valid][-1])
    final_rate = float(np.median(normalized_rate[valid][-count:]))

    entry = np.nan
    for index in _sustained_start_indices(sds_like, count):
        remaining = valid & (scaled_time >= scaled_time[index])
        if np.mean(sds_like[remaining]) >= tail_fraction:
            entry = float(scaled_time[index])
            break

    starts = _sustained_start_indices(schwarzschild_like, count)
    ends = scaled_time[starts + count - 1] if starts.size else np.empty(0)
    if np.isfinite(entry):
        ends = ends[ends <= entry]
    departure = float(ends[-1]) if ends.size else np.nan

    if not np.isfinite(entry):
        status = "no_cosmological_entry"
    elif not np.isfinite(departure):
        status = "no_schwarzschild_agreement"
    else:
        status = "resolved"
    return TransitionInterval(status, departure, entry, final_rate, resolved_end)


# --------------------------------------------------------------------------
# Systematic sweep
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepGrid:
    """Estimator and criterion parameters varied for the systematic range."""

    smoothing_widths: tuple[float, ...] = (20.0, 30.0, 45.0)
    rms_fractions: tuple[float, ...] = (0.4, 0.6)
    persistences: tuple[float, ...] = (0.15, 0.25, 0.40)
    tolerances: tuple[float, ...] = (0.05, 0.10, 0.20)

    def configurations(self):
        for width in self.smoothing_widths:
            for fraction in self.rms_fractions:
                for persistence in self.persistences:
                    for tolerance in self.tolerances:
                        yield EnvelopeSettings(width, fraction), persistence, tolerance

    @property
    def size(self) -> int:
        return (
            len(self.smoothing_widths)
            * len(self.rms_fractions)
            * len(self.persistences)
            * len(self.tolerances)
        )


@dataclass
class SweepSummary:
    """Systematic range of one transition interval over the sweep."""

    ell: int
    length: float
    observer: float | None
    kappa: float
    configurations: int
    resolved: int
    departures: list[float] = field(default_factory=list)
    entries: list[float] = field(default_factory=list)
    final_rates: list[float] = field(default_factory=list)
    statuses: dict[str, int] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.configurations == 0:
            return "no_data"
        if self.resolved == 0:
            return max(self.statuses, key=self.statuses.get)
        if self.resolved < 0.5 * self.configurations:
            return "marginal"
        return "resolved"

    def _statistics(self, values: list[float], prefix: str) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return {
                f"{prefix}_median": float("nan"),
                f"{prefix}_minimum": float("nan"),
                f"{prefix}_maximum": float("nan"),
            }
        return {
            f"{prefix}_median": float(np.median(array)),
            f"{prefix}_minimum": float(np.min(array)),
            f"{prefix}_maximum": float(np.max(array)),
        }

    def as_row(self) -> dict:
        row: dict = {
            "ell": self.ell,
            "L_over_M": self.length,
            "observer": observer_key(self.observer),
            "kappa_c": self.kappa,
            "status": self.status,
            "configurations": self.configurations,
            "resolved_configurations": self.resolved,
        }
        row.update(self._statistics(self.departures, "kappa_c_U_departure"))
        row.update(self._statistics(self.entries, "kappa_c_U_entry"))
        for key in (
            "kappa_c_U_departure_median",
            "kappa_c_U_departure_minimum",
            "kappa_c_U_departure_maximum",
            "kappa_c_U_entry_median",
            "kappa_c_U_entry_minimum",
            "kappa_c_U_entry_maximum",
        ):
            row[key.replace("kappa_c_U", "U_over_M")] = row[key] / self.kappa
        row["final_rate_median"] = (
            float(np.median(self.final_rates)) if self.final_rates else float("nan")
        )
        row["status_counts"] = ";".join(
            f"{name}={count}" for name, count in sorted(self.statuses.items())
        )
        return row


def _interpolated_reference(
    times: np.ndarray,
    reference_times: np.ndarray,
    reference_rate: np.ndarray,
) -> np.ndarray:
    """Interpolate a reference rate onto another time grid without extending it."""

    finite = np.isfinite(reference_rate)
    if not np.any(finite):
        return np.full_like(times, np.nan)
    values = np.interp(
        times,
        reference_times[finite],
        reference_rate[finite],
        left=np.nan,
        right=np.nan,
    )
    inside = (times >= reference_times[finite][0]) & (
        times <= reference_times[finite][-1]
    )
    values[~inside] = np.nan
    return values


def rate_pair(
    result,
    reference,
    radius: float | None,
    kappa: float,
    settings: EnvelopeSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``kappa_c U`` with the finite-``L`` and Schwarzschild rates."""

    times, signal = retarded_series(result, radius)
    rate, _ = envelope_rate(times, signal, settings)
    reference_times, reference_signal = retarded_series(reference, radius)
    reference_rate, _ = envelope_rate(reference_times, reference_signal, settings)
    aligned = _interpolated_reference(times, reference_times, reference_rate)
    return kappa * times, rate / kappa, aligned / kappa


def sweep_transition(
    result,
    reference,
    radius: float | None,
    *,
    ell: int,
    length: float,
    kappa: float,
    grid: SweepGrid | None = None,
) -> tuple[SweepSummary, list[dict]]:
    """Repeat the transition-interval measurement over the parameter grid."""

    grid = grid or SweepGrid()
    summary = SweepSummary(
        ell=ell,
        length=length,
        observer=radius,
        kappa=kappa,
        configurations=0,
        resolved=0,
    )
    rows: list[dict] = []
    for settings, persistence, tolerance in grid.configurations():
        try:
            scaled, normalized, reference_rate = rate_pair(
                result, reference, radius, kappa, settings
            )
            interval = transition_interval(
                scaled,
                normalized,
                reference_rate,
                ell,
                tolerance=tolerance,
                persistence=persistence,
                minimum_scaled_time=kappa * MINIMUM_RETARDED_TIME,
            )
        except ValueError:
            continue
        summary.configurations += 1
        summary.statuses[interval.status] = summary.statuses.get(interval.status, 0) + 1
        summary.final_rates.append(interval.final_rate)
        if interval.is_resolved:
            summary.resolved += 1
            summary.departures.append(interval.departure)
            summary.entries.append(interval.entry)
        rows.append(
            {
                "ell": ell,
                "L_over_M": length,
                "observer": observer_key(radius),
                "smoothing_width_over_M": settings.smoothing_width,
                "rms_fraction": settings.rms_fraction,
                "persistence_kappa_c_U": persistence,
                "tolerance": tolerance,
                "status": interval.status,
                "kappa_c_U_departure": interval.departure,
                "kappa_c_U_entry": interval.entry,
                "U_departure_over_M": interval.departure / kappa,
                "U_entry_over_M": interval.entry / kappa,
                "final_normalized_rate": interval.final_rate,
                "resolved_end_kappa_c_U": interval.resolved_end,
            }
        )
    return summary, rows


# --------------------------------------------------------------------------
# Schwarzschild power-law onset
# --------------------------------------------------------------------------


def power_law_onset(
    result,
    radius: float | None,
    power: float,
    *,
    settings: EnvelopeSettings = EnvelopeSettings(),
    tolerance: float = 0.05,
    persistence: float = 40.0,
    horizon: float = 150.0,
) -> float:
    """Return the retarded time at which the Price index first settles.

    The returned time is the start of the first interval of width
    ``persistence`` over which the measured local index ``gamma_eff U`` stays
    within ``tolerance`` of ``power``, and over which it also holds for most of
    the following ``horizon``.  The forward horizon is finite because every
    finite-radius signal eventually reaches the amplitude floor of the
    evolution, where the measured index stops being meaningful.
    """

    times, signal = retarded_series(result, radius)
    rate, _ = envelope_rate(times, signal, settings)
    index = rate * times
    valid = np.isfinite(index) & (times > 0.0)
    close = valid & (np.abs(index - power) <= tolerance * power)
    step = float(np.median(np.diff(times)))
    count = max(2, int(round(persistence / step)))
    for start in _sustained_start_indices(close, count):
        ahead = valid & (times >= times[start]) & (times <= times[start] + horizon)
        if np.any(ahead) and np.mean(close[ahead]) >= 0.8:
            return float(times[start])
    return float("nan")


# --------------------------------------------------------------------------
# Input archives
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Archives:
    """Directories holding the evolutions used by the final analysis.

    ``refined`` is searched first, so the extended Schwarzschild reference and
    the new convergence ladder take precedence over the earlier archives of
    the same name.
    """

    refined: Path = Path("results/sds_scalar/tails/crossover_final/raw")
    baseline: Path = Path("results/sds_scalar/tails/high_resolution_rates/raw")
    timestep: Path = Path("results/sds_scalar/tails/crossover_final/raw_timestep")

    def locate(self, name: str) -> Path:
        for directory in (self.refined, self.baseline):
            candidate = Path(directory) / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{name} is present in neither {self.refined} nor {self.baseline}."
        )

    def exists(self, name: str) -> bool:
        return any(
            (Path(directory) / name).exists()
            for directory in (self.refined, self.baseline)
        )

    def load(self, name: str):
        return load_sds_result(self.locate(name))


def sds_name(ell: int, length: float, resolution: int) -> str:
    return f"sds_ell{ell}_L{length:g}_N{resolution}.npz"


def schwarzschild_name(ell: int, resolution: int) -> str:
    return f"schwarzschild_ell{ell}_N{resolution}.npz"


def cosmological_rate(length: float) -> float:
    """Return the cosmological surface gravity of one SdS background."""

    return sds_horizons(
        SdSParameters(mass=1.0, cosmological_length=length, ell=0)
    ).kappa_cosmological


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _clipped(x: np.ndarray, y: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Return a copy of ``y`` with out-of-range values blanked, not clamped."""

    values = np.array(y, dtype=float)
    values[(~np.isfinite(values)) | (values < limits[0]) | (values > limits[1])] = np.nan
    return values


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------


REFERENCE_SETTINGS = EnvelopeSettings(30.0, 0.5)
# The quadrupole carrier is faster, so its envelope needs a wider window.
REFERENCE_SETTINGS_ELL2 = EnvelopeSettings(45.0, 0.5)
REFERENCE_TOLERANCE = 0.10
RATE_LIMITS = (-0.2, 4.4)


def measure_transitions(
    archives: Archives,
    *,
    grid: SweepGrid | None = None,
) -> tuple[list[SweepSummary], list[dict]]:
    """Measure every ell=1 transition interval, plus the ell=2 check."""

    grid = grid or SweepGrid()
    summaries: list[SweepSummary] = []
    sweep_rows: list[dict] = []

    reference = archives.load(schwarzschild_name(1, 2048))
    for length in ELL1_LENGTHS:
        result = archives.load(sds_name(1, length, 2048))
        kappa = cosmological_rate(length)
        for radius in SDS_OBSERVERS:
            summary, rows = sweep_transition(
                result,
                reference,
                radius,
                ell=1,
                length=length,
                kappa=kappa,
                grid=grid,
            )
            summaries.append(summary)
            sweep_rows.extend(rows)

    reference_ell2 = archives.load(schwarzschild_name(2, 4096))
    result_ell2 = archives.load(sds_name(2, 80.0, 4096))
    kappa = cosmological_rate(80.0)
    for radius in SDS_OBSERVERS:
        summary, rows = sweep_transition(
            result_ell2,
            reference_ell2,
            radius,
            ell=2,
            length=80.0,
            kappa=kappa,
            grid=grid,
        )
        summaries.append(summary)
        sweep_rows.extend(rows)
    return summaries, sweep_rows


def _transition_span(summaries: list[SweepSummary], ell: int, length: float, radius):
    for summary in summaries:
        if (
            summary.ell == ell
            and summary.length == length
            and summary.observer == radius
        ):
            return summary
    return None


def plot_transition_intervals(
    archives: Archives,
    summaries: list[SweepSummary],
    output_dir: Path,
) -> Path:
    """Plot the dipole rate transition against the Schwarzschild reference."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = archives.load(schwarzschild_name(1, 2048))
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharey=True)
    for axis, length in zip(axes.ravel(), ELL1_LENGTHS):
        result = archives.load(sds_name(1, length, 2048))
        kappa = cosmological_rate(length)
        limit = kappa * float(result.signal_times[-1])
        axis.axhspan(
            1.0 - REFERENCE_TOLERANCE,
            1.0 + REFERENCE_TOLERANCE,
            color="0.85",
            zorder=0,
        )
        axis.axhline(1.0, color="black", linestyle="-.", linewidth=1.0, zorder=1)
        for radius in SDS_OBSERVERS:
            scaled, normalized, reference_rate = rate_pair(
                result, reference, radius, kappa, REFERENCE_SETTINGS
            )
            emphasized = radius in (PRIMARY_RADIUS, None)
            axis.plot(
                scaled,
                _clipped(scaled, normalized, RATE_LIMITS),
                color=OBSERVER_COLORS[radius],
                linewidth=1.6 if emphasized else 0.9,
                alpha=1.0 if emphasized else 0.45,
                label=observer_label(radius),
                zorder=3 if emphasized else 2,
            )
            if emphasized:
                # Only the two compared observers carry their Schwarzschild
                # reference; drawing all four makes the panel unreadable.
                axis.plot(
                    scaled,
                    _clipped(scaled, reference_rate, RATE_LIMITS),
                    color=OBSERVER_COLORS[radius],
                    linewidth=1.0,
                    linestyle=(0, (4, 2)),
                    alpha=0.6,
                    zorder=2,
                )
        summary = _transition_span(summaries, 1, length, PRIMARY_RADIUS)
        if summary is not None and summary.departures and summary.entries:
            axis.axvspan(
                float(np.median(summary.departures)),
                float(np.median(summary.entries)),
                color=OBSERVER_COLORS[PRIMARY_RADIUS],
                alpha=0.13,
                zorder=0,
            )
        horizon = _transition_span(summaries, 1, length, None)
        if horizon is not None and horizon.departures and horizon.entries:
            for value, style in (
                (float(np.median(horizon.departures)), (0, (1, 1))),
                (float(np.median(horizon.entries)), "-"),
            ):
                axis.axvline(
                    value,
                    color="black",
                    linewidth=0.9,
                    linestyle=style,
                    alpha=0.5,
                    zorder=1,
                )
        status = summary.status if summary is not None else "no_data"
        note = "" if status == "resolved" else f"  ({status.replace('_', ' ')})"
        axis.set_title(rf"$L/M={length:g}$, $\kappa_c^{{-1}}={1.0 / kappa:.0f}M$" + note)
        axis.set_xlim(0.35, min(4.6, limit))
        axis.set_ylim(RATE_LIMITS)
        axis.grid(alpha=0.22)
        if summary is not None and summary.departures:
            axis.text(
                0.985,
                0.94,
                rf"$r=8M$: $\kappa_c U={np.median(summary.departures):.2f}"
                rf"\rightarrow{np.median(summary.entries):.2f}$"
                f"\n{summary.resolved}/{summary.configurations} configurations",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
            )
    for axis in axes[-1]:
        axis.set_xlabel(r"$\kappa_c U$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.4", linestyle=(0, (4, 2))))
    labels.append("Schwarzschild reference")
    handles.append(plt.Line2D([], [], color="black", linestyle="-."))
    labels.append(r"SdS target $\gamma/\kappa_c=\ell$")
    fig.legend(handles, labels, ncol=6, loc="lower center", fontsize=8.5)
    fig.suptitle(
        r"Dipole rate transition: shading marks the $r=8M$ transition interval"
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    path = output_dir / "sds_ell1_transition_intervals.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_transition_uncertainty(
    summaries: list[SweepSummary], output_dir: Path
) -> Path:
    """Plot departure and entry ranges obtained from the systematic sweep."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    offsets = {4.0: -0.18, 8.0: -0.06, 16.0: 0.06, None: 0.18}
    positions = {length: index for index, length in enumerate(ELL1_LENGTHS)}
    unresolved: list[tuple[float, str]] = []
    for radius in SDS_OBSERVERS:
        color = OBSERVER_COLORS[radius]
        for scaled_axis, axis in zip((True, False), axes):
            drawn = False
            for length in ELL1_LENGTHS:
                summary = _transition_span(summaries, 1, length, radius)
                if summary is None:
                    continue
                position = positions[length] + offsets[radius]
                if not summary.departures:
                    if scaled_axis:
                        unresolved.append((position, color))
                    continue
                factor = 1.0 if scaled_axis else 1.0 / summary.kappa
                departures = np.asarray(summary.departures) * factor
                entries = np.asarray(summary.entries) * factor
                axis.vlines(
                    position,
                    np.median(departures),
                    np.median(entries),
                    color=color,
                    linewidth=6.0,
                    alpha=0.30,
                )
                for values, marker in ((departures, "v"), (entries, "^")):
                    axis.vlines(
                        position,
                        np.min(values),
                        np.max(values),
                        color=color,
                        linewidth=1.2,
                    )
                    axis.plot(
                        position,
                        np.median(values),
                        marker=marker,
                        color=color,
                        markersize=6.5,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        label=observer_label(radius) if not drawn else None,
                    )
                    drawn = True
    for axis, label, title in zip(
        axes,
        (r"$\kappa_c U$", r"$U/M$"),
        ("Cosmological units", "Geometric units"),
    ):
        axis.set_xticks(list(positions.values()))
        axis.set_xticklabels([rf"${value:g}$" for value in ELL1_LENGTHS])
        axis.set_xlabel(r"$L/M$")
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.grid(alpha=0.22, axis="y")
    axes[1].set_yscale("log")
    for position, color in unresolved:
        # Blended coordinates keep the marker on the axis floor whatever the
        # data limits turn out to be.
        axes[0].plot(
            position,
            0.02,
            transform=axes[0].get_xaxis_transform(),
            marker="x",
            color=color,
            markersize=5,
            clip_on=False,
        )
    axes[0].text(
        0.5,
        0.055,
        "crosses mark observers with no resolved transition",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=8,
        color="0.35",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, name in zip(handles, labels):
        unique.setdefault(name, handle)
    handles = [
        *unique.values(),
        plt.Line2D([], [], color="0.3", marker="v", linestyle="none"),
        plt.Line2D([], [], color="0.3", marker="^", linestyle="none"),
    ]
    labels = [*unique.keys(), "departure", "entry"]
    fig.legend(handles, labels, ncol=6, loc="lower center", fontsize=8.5)
    fig.suptitle(
        "Transition intervals with systematic ranges from the estimator sweep"
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    path = output_dir / "sds_ell1_transition_uncertainty.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_scaled_universality(
    archives: Archives,
    summaries: list[SweepSummary],
    output_dir: Path,
) -> Path:
    """Compare the transition of every ``L`` at ``r=8M`` in cosmological units."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = archives.load(schwarzschild_name(1, 2048))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.82, len(ELL1_LENGTHS)))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    for color, length in zip(colors, ELL1_LENGTHS):
        result = archives.load(sds_name(1, length, 2048))
        kappa = cosmological_rate(length)
        scaled, normalized, _ = rate_pair(
            result, reference, PRIMARY_RADIUS, kappa, REFERENCE_SETTINGS
        )
        axes[0].plot(
            scaled,
            _clipped(scaled, normalized, (0.0, 3.2)),
            color=color,
            linewidth=1.5,
            label=rf"$L/M={length:g}$",
        )
        summary = _transition_span(summaries, 1, length, PRIMARY_RADIUS)
        if summary is not None and summary.status == "resolved":
            axes[0].plot(
                np.median(summary.entries),
                1.0,
                marker="o",
                color=color,
                markersize=7,
                markeredgecolor="white",
            )
    axes[0].axhspan(
        1.0 - REFERENCE_TOLERANCE, 1.0 + REFERENCE_TOLERANCE, color="0.85", zorder=0
    )
    axes[0].axhline(1.0, color="black", linestyle="-.", linewidth=1.0)
    axes[0].set(
        xlim=(0.8, 4.6),
        ylim=(0.0, 3.2),
        xlabel=r"$\kappa_c U$",
        ylabel=r"$\gamma_{\rm eff}/\kappa_c$",
        title=r"Dipole rate at $r=8M$, all cosmological lengths",
    )
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.22)

    lengths, medians, lows, highs, physical = [], [], [], [], []
    marginal_lengths, marginal_entries = [], []
    for length in ELL1_LENGTHS:
        summary = _transition_span(summaries, 1, length, PRIMARY_RADIUS)
        if summary is None or not summary.entries:
            continue
        values = np.asarray(summary.entries)
        if summary.status != "resolved":
            marginal_lengths.append(length)
            marginal_entries.append(float(np.median(values)))
            continue
        lengths.append(length)
        medians.append(float(np.median(values)))
        lows.append(float(np.min(values)))
        highs.append(float(np.max(values)))
        physical.append(float(np.median(values)) / summary.kappa)
    if marginal_lengths:
        axes[1].plot(
            marginal_lengths,
            marginal_entries,
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            color=OBSERVER_COLORS[PRIMARY_RADIUS],
            markersize=7,
            label="marginal",
        )
    if lengths:
        axes[1].errorbar(
            lengths,
            medians,
            yerr=[
                np.asarray(medians) - np.asarray(lows),
                np.asarray(highs) - np.asarray(medians),
            ],
            marker="o",
            color=OBSERVER_COLORS[PRIMARY_RADIUS],
            capsize=4,
            linewidth=1.4,
            label=r"$\kappa_c U_{\rm entry}$",
        )
        twin = axes[1].twinx()
        twin.plot(
            lengths,
            physical,
            marker="s",
            color="0.45",
            linestyle="--",
            linewidth=1.2,
            label=r"$U_{\rm entry}/M$",
        )
        twin.set_ylabel(r"$U_{\rm entry}/M$", color="0.35")
        twin.tick_params(axis="y", colors="0.35")
        handles, labels = axes[1].get_legend_handles_labels()
        extra, extra_labels = twin.get_legend_handles_labels()
        axes[1].legend(
            handles + extra,
            labels + extra_labels,
            fontsize=8.5,
            loc="center left",
        )
    axes[1].set(
        xscale="log",
        xlabel=r"$L/M$",
        ylabel=r"$\kappa_c U_{\rm entry}$",
        title=r"Entry into the cosmological rate at $r=8M$",
    )
    axes[1].set_xticks(list(ELL1_LENGTHS))
    axes[1].set_xticklabels([rf"${value:g}$" for value in ELL1_LENGTHS])
    axes[1].grid(alpha=0.22)
    fig.suptitle(
        r"Entry time is nearly $L$-independent in units of $\kappa_c^{-1}$"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = output_dir / "sds_ell1_scaled_entry.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


# The L/M=160 tail at r=8M falls to about 1e-9 of the peak amplitude, so it
# needs one refinement level beyond the L/M=80 ladder before the finest run can
# be treated as the reference.
CONVERGENCE_LADDER: dict[float, tuple[int, ...]] = {
    80.0: (1024, 1536, 2048),
    160.0: (1024, 1536, 2048, 3072),
}
CONVERGENCE_LENGTHS = tuple(CONVERGENCE_LADDER)
MATCHED_TIMESTEP = 0.0025
TIMESTEP_CHECK = (80.0, 1024, 0.00125)


def _timestep_check(
    archives: Archives,
    results: dict[int, object],
    axis,
    kappa: float,
) -> dict:
    """Compare one matched-resolution pair of timesteps at ``r=8M``.

    The spatial differences are only a measure of spatial error if the shared
    timestep contributes less, so the halved-timestep evolution at the coarsest
    resolution is drawn on the same axis.
    """

    length, resolution, timestep = TIMESTEP_CHECK
    row = {
        "ell": 1,
        "L_over_M": length,
        "observer": observer_key(PRIMARY_RADIUS),
        "resolution": resolution,
        "timestep": timestep,
        "comparison": f"timestep {timestep:g} against {MATCHED_TIMESTEP:g}",
    }
    path = Path(archives.timestep) / sds_name(1, length, resolution)
    if not path.exists():
        row["status"] = "missing_timestep_archive"
        return row
    halved = load_sds_result(path)
    times, signal = retarded_series(results[resolution], PRIMARY_RADIUS)
    scale = float(np.max(np.abs(signal)))
    halved_times, halved_signal = retarded_series(halved, PRIMARY_RADIUS)
    difference = np.abs(np.interp(times, halved_times, halved_signal) - signal)
    _, envelope = envelope_rate(times, signal, REFERENCE_SETTINGS)
    local = difference / np.where(np.isfinite(envelope), envelope, np.nan)
    window = (times > 0.5 / kappa) & (times < 4.0 / kappa)
    axis.semilogy(
        kappa * times,
        np.maximum(local, 1e-18),
        color="0.45",
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        label=rf"$\Delta\tau/2$ against $\Delta\tau$, $N={resolution}$",
    )
    finite = window & np.isfinite(local)
    row["status"] = "measured"
    row["maximum_constraint_linf"] = float(np.max(halved.constraint_linf))
    row["maximum_relative_difference"] = float(np.max(difference[window]) / scale)
    row["maximum_local_relative_difference"] = float(np.max(local[finite]))
    row["median_local_relative_difference"] = float(np.median(local[finite]))
    return row


def analyze_convergence(
    archives: Archives,
    output_dir: Path,
    *,
    grid: SweepGrid | None = None,
) -> tuple[Path, list[dict]]:
    """Show finite-radius spatial convergence at ``r=8M`` for ``L/M=80,160``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = grid or SweepGrid()
    reference = archives.load(schwarzschild_name(1, 2048))
    rows: list[dict] = []
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))

    for row_index, length in enumerate(CONVERGENCE_LENGTHS):
        kappa = cosmological_rate(length)
        ladder = tuple(
            resolution
            for resolution in CONVERGENCE_LADDER[length]
            if archives.exists(sds_name(1, length, resolution))
        )
        if len(ladder) < 2:
            raise FileNotFoundError(
                f"L/M={length:g} needs at least two archived resolutions."
            )
        results = {
            resolution: archives.load(sds_name(1, length, resolution))
            for resolution in ladder
        }
        finest = results[ladder[-1]]
        amplitude_axis, difference_axis, rate_axis = axes[row_index]

        # The cosmological horizon is included in the table because its late
        # amplitude is orders of magnitude larger than the r=8M amplitude, so
        # the two observers do not need the same resolution.
        for radius in (PRIMARY_RADIUS, None):
            plotted = radius == PRIMARY_RADIUS
            finest_times, finest_signal = retarded_series(finest, radius)
            scale = float(np.max(np.abs(finest_signal)))
            _, finest_envelope = envelope_rate(
                finest_times, finest_signal, REFERENCE_SETTINGS
            )
            window = (finest_times > 0.5 / kappa) & (finest_times < 4.0 / kappa)
            for resolution in ladder:
                result = results[resolution]
                times, signal = retarded_series(result, radius)
                rate, envelope = envelope_rate(times, signal, REFERENCE_SETTINGS)
                color = RESOLUTION_COLORS[resolution]
                if plotted:
                    # The amplitudes coincide wherever the run is converged, so
                    # they are drawn from thick and pale to thin and dark.
                    width, opacity = RESOLUTION_WEIGHTS[resolution]
                    amplitude_axis.semilogy(
                        kappa * times,
                        envelope,
                        color=color,
                        linewidth=width,
                        alpha=opacity,
                        label=rf"$N={resolution}$",
                    )
                    rate_axis.plot(
                        kappa * times,
                        _clipped(kappa * times, rate / kappa, RATE_LIMITS),
                        color=color,
                        linewidth=1.4,
                        label=rf"$N={resolution}$",
                    )
                summary, _ = sweep_transition(
                    result,
                    reference,
                    radius,
                    ell=1,
                    length=length,
                    kappa=kappa,
                    grid=grid,
                )
                row = {
                    "ell": 1,
                    "L_over_M": length,
                    "observer": observer_key(radius),
                    "resolution": resolution,
                    "timestep": float(result.metadata["numerical"]["timestep"]),
                    "comparison": f"resolution {resolution} against {ladder[-1]}",
                    "maximum_constraint_linf": float(np.max(result.constraint_linf)),
                }
                row.update(
                    {
                        key: value
                        for key, value in summary.as_row().items()
                        if key
                        in (
                            "status",
                            "resolved_configurations",
                            "kappa_c_U_departure_median",
                            "kappa_c_U_entry_median",
                            "U_over_M_departure_median",
                            "U_over_M_entry_median",
                        )
                    }
                )
                if resolution != ladder[-1]:
                    interpolated = np.interp(finest_times, times, signal)
                    difference = np.abs(interpolated - finest_signal)
                    # The tail amplitude falls by many orders of magnitude, so
                    # the error that matters is measured against the local
                    # amplitude rather than against the peak.
                    local = difference / np.where(
                        np.isfinite(finest_envelope), finest_envelope, np.nan
                    )
                    if plotted:
                        difference_axis.semilogy(
                            kappa * finest_times,
                            np.maximum(local, 1e-18),
                            color=color,
                            linewidth=1.1,
                            label=rf"$N={resolution}$ against $N={ladder[-1]}$",
                        )
                    finite = window & np.isfinite(local)
                    row["maximum_relative_difference"] = float(
                        np.max(difference[window]) / scale
                    )
                    row["maximum_local_relative_difference"] = float(
                        np.max(local[finite])
                    )
                    row["median_local_relative_difference"] = float(
                        np.median(local[finite])
                    )
                rows.append(row)

        if length == TIMESTEP_CHECK[0]:
            rows.append(
                _timestep_check(archives, results, difference_axis, kappa)
            )

        amplitude_axis.set(
            ylabel=r"RMS envelope at $r=8M$",
            title=rf"$L/M={length:g}$: amplitude",
        )
        difference_axis.axhline(0.01, color="0.3", linewidth=0.8, linestyle=":")
        difference_axis.set(
            ylim=(1e-16, 1.0),
            ylabel=r"$|u_N-u_{\rm ref}|/A_{\rm ref}(U)$",
            title=rf"$L/M={length:g}$: difference over local amplitude",
        )
        rate_axis.axhline(1.0, color="black", linestyle="-.", linewidth=1.0)
        rate_axis.set(
            ylim=RATE_LIMITS,
            ylabel=r"$\gamma_{\rm eff}/\kappa_c$",
            title=rf"$L/M={length:g}$: local rate",
        )
        for axis in axes[row_index]:
            axis.set_xlim(0.35, 4.6)
            axis.set_xlabel(r"$\kappa_c U$")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)
    fig.suptitle(
        r"Spatial convergence at $r=8M$ with matched timestep $\Delta\tau=0.0025M$"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = output_dir / "sds_ell1_r8_convergence.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    _write_rows(output_dir / "sds_ell1_r8_convergence.csv", rows)
    return path, rows


def plot_ell2_check(
    archives: Archives,
    summaries: list[SweepSummary],
    output_dir: Path,
) -> Path:
    """Plot the quadrupole transition at ``L/M=80`` as an independent check."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = archives.load(schwarzschild_name(2, 4096))
    result = archives.load(sds_name(2, 80.0, 4096))
    kappa = cosmological_rate(80.0)
    limits = (-0.5, 9.0)
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.axhspan(
        2.0 * (1.0 - REFERENCE_TOLERANCE),
        2.0 * (1.0 + REFERENCE_TOLERANCE),
        color="0.85",
        zorder=0,
    )
    axis.axhline(2.0, color="black", linestyle="-.", linewidth=1.0)
    for radius in SDS_OBSERVERS:
        scaled, normalized, reference_rate = rate_pair(
            result, reference, radius, kappa, REFERENCE_SETTINGS_ELL2
        )
        axis.plot(
            scaled,
            _clipped(scaled, normalized, limits),
            color=OBSERVER_COLORS[radius],
            linewidth=1.6 if radius is None else 1.25,
            label=observer_label(radius),
        )
        axis.plot(
            scaled,
            _clipped(scaled, reference_rate, limits),
            color=OBSERVER_COLORS[radius],
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            alpha=0.55,
        )
    summary = _transition_span(summaries, 2, 80.0, PRIMARY_RADIUS)
    if summary is not None and summary.departures and summary.entries:
        axis.axvspan(
            float(np.median(summary.departures)),
            float(np.median(summary.entries)),
            color=OBSERVER_COLORS[PRIMARY_RADIUS],
            alpha=0.13,
            zorder=0,
        )
    axis.set(
        xlim=(0.35, 5.6),
        ylim=limits,
        xlabel=r"$\kappa_c U$",
        ylabel=r"$\gamma_{\rm eff}/\kappa_c$",
        title=r"Quadrupole check, $\ell=2$, $L/M=80$, $N=4096$",
    )
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    path = output_dir / "sds_ell2_L80_transition.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def create_report(
    output_dir: Path,
    *,
    archives: Archives | None = None,
    grid: SweepGrid | None = None,
) -> list[Path]:
    """Create every final crossover figure, table, and diagnostic record."""

    archives = archives or Archives()
    grid = grid or SweepGrid()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, sweep_rows = measure_transitions(archives, grid=grid)
    _write_rows(
        output_dir / "transition_intervals.csv",
        [summary.as_row() for summary in summaries],
    )
    _write_rows(output_dir / "transition_sweep.csv", sweep_rows)

    figures = [
        plot_transition_intervals(archives, summaries, output_dir),
        plot_transition_uncertainty(summaries, output_dir),
        plot_scaled_universality(archives, summaries, output_dir),
        plot_ell2_check(archives, summaries, output_dir),
    ]
    convergence_figure, convergence_rows = analyze_convergence(
        archives, output_dir, grid=grid
    )
    figures.append(convergence_figure)

    onset_rows = []
    for ell, resolution in ((1, 2048), (2, 4096)):
        reference = archives.load(schwarzschild_name(ell, resolution))
        for radius in SDS_OBSERVERS:
            power = ell + 2 if radius is None else 2 * ell + 3
            onset_rows.append(
                {
                    "ell": ell,
                    "observer": "future_null_infinity"
                    if radius is None
                    else observer_key(radius),
                    "price_index": power,
                    "U_onset_over_M": power_law_onset(reference, radius, float(power)),
                }
            )
    _write_rows(output_dir / "schwarzschild_power_law_onset.csv", onset_rows)

    evolutions = []
    cases = [(1, length, 2048) for length in ELL1_LENGTHS]
    cases.extend(
        (1, length, resolution)
        for length, ladder in CONVERGENCE_LADDER.items()
        for resolution in ladder
        if archives.exists(sds_name(1, length, resolution))
    )
    cases.append((2, 80.0, 4096))
    for ell, length, resolution in dict.fromkeys(cases):
        result = archives.load(sds_name(ell, length, resolution))
        evolutions.append(
            {
                "background": "sds",
                "ell": ell,
                "L_over_M": length,
                "resolution": resolution,
                "timestep": float(result.metadata["numerical"]["timestep"]),
                "final_retarded_time": float(
                    result.signal_times[-1]
                    - result.metadata["retarded_time_offset"]["q"]
                ),
                "maximum_constraint_linf": float(np.max(result.constraint_linf)),
                "wall_seconds": float(result.metadata["wall_seconds"]),
            }
        )
    for ell, resolution in ((1, 2048), (2, 4096)):
        result = archives.load(schwarzschild_name(ell, resolution))
        evolutions.append(
            {
                "background": "schwarzschild",
                "ell": ell,
                "L_over_M": None,
                "resolution": resolution,
                "timestep": float(result.metadata["numerical"]["timestep"]),
                "final_retarded_time": float(
                    result.signal_times[-1]
                    - result.metadata["retarded_time_offset"]["q"]
                ),
                "maximum_constraint_linf": float(np.max(result.constraint_linf)),
                "wall_seconds": float(result.metadata["wall_seconds"]),
            }
        )

    diagnostics = {
        "primary_time_variable": "kappa_c U",
        "rate_definition": "gamma_eff = -d ln A/dU of a centered RMS envelope A",
        "normalization": "gamma_eff/kappa_c; the SdS target is ell",
        "schwarzschild_reference": (
            "local rate of an independent Schwarzschild evolution with the same "
            "physical initial velocity, read at the same areal radius; the SdS "
            "cosmological horizon is compared with future null infinity"
        ),
        "departure_definition": (
            "end of the last interval of width 'persistence' before the entry "
            "over which |gamma_L-gamma_0| <= tolerance*max(|gamma_0|, ell*kappa_c)"
        ),
        "entry_definition": (
            "start of the first interval of width 'persistence' over which "
            "|gamma_eff/kappa_c - ell| <= tolerance*ell, provided the same "
            "tolerance holds over at least 80 percent of the remaining "
            "resolved samples"
        ),
        "unresolved_definition": (
            "a case with no persistent entry into the cosmological band, or no "
            "persistent Schwarzschild agreement before it, is reported as "
            "unresolved rather than assigned a crossing time"
        ),
        "sweep": {
            "smoothing_width_over_M": list(grid.smoothing_widths),
            "rms_fraction": list(grid.rms_fractions),
            "persistence_kappa_c_U": list(grid.persistences),
            "tolerance": list(grid.tolerances),
            "configurations": grid.size,
        },
        "convergence": {
            "observer": observer_key(PRIMARY_RADIUS),
            "lengths": list(CONVERGENCE_LENGTHS),
            "resolutions": {
                str(length): list(ladder)
                for length, ladder in CONVERGENCE_LADDER.items()
            },
            "matched_timestep": 0.0025,
        },
        "transitions": [summary.as_row() for summary in summaries],
        "convergence_rows": convergence_rows,
        "power_law_onset": onset_rows,
        "evolutions": evolutions,
    }
    path = output_dir / "diagnostics.json"
    path.write_text(
        json.dumps(json_safe(diagnostics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    figures.append(path)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final SdS tail crossover analysis with systematic ranges."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sds_scalar/tails/crossover_final"),
    )
    parser.add_argument(
        "--refined-raw",
        type=Path,
        default=Path("results/sds_scalar/tails/crossover_final/raw"),
    )
    parser.add_argument(
        "--baseline-raw",
        type=Path,
        default=Path("results/sds_scalar/tails/high_resolution_rates/raw"),
    )
    args = parser.parse_args()
    for path in create_report(
        args.output_dir,
        archives=Archives(refined=args.refined_raw, baseline=args.baseline_raw),
    ):
        print(path)


if __name__ == "__main__":
    main()
