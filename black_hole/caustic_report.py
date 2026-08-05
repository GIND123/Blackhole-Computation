r"""Figures, tables, and the written report of the Green-function study.

Every figure is generated from the archived modal evolutions produced by
:mod:`black_hole.caustic_study`.  Nothing here re-runs a simulation, so the
report can be regenerated from the stored data alone.

Colour conventions
------------------

The cosmological length is an *ordered* parameter, so the four values
``L/M = 20, 40, 80, 160`` are drawn from a single perceptually uniform ramp
rather than from unrelated categorical hues; the asymptotically flat
reference is always black.  Line style carries the same information as
colour wherever more than two curves share an axis, so no figure depends on
colour alone.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from scipy.signal import hilbert

from .caustic_study import (
    ANGULAR_LADDER,
    BROAD_SOURCE,
    COSMOLOGICAL_LENGTHS,
    NARROW_SOURCE,
    PHOTON_SPHERE_PERIOD,
    RADIAL_LADDER,
    TIMESTEP_LADDER,
    VALIDATION_WINDOW,
    case_label,
    case_title,
    direction_waveform,
    echo_phase_shifts,
    equatorial_waveform,
    find_caustic_pulses,
    flat_limit_norms,
    harmonic_matrix,
    load_case,
    load_convergence,
    load_narrow,
    modal_energy_spectrum,
    pulse_timing_shifts,
)
from .crossover_final import EnvelopeSettings, envelope_rate
from .localized_source import (
    LocalizedSourceParameters,
    angular_profile,
    angular_spectral_weights,
    build_mode_catalogue,
    radial_profile,
    time_profile,
)
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    minimal_height,
    tortoise_coordinate,
)
from .source_evolution import SourcedSimulationResult
from .static_reference import (
    StaticReferenceGrid,
    reflection_free_time,
    solve_static_mode,
)
from .tail_analysis import json_safe

LOGGER = logging.getLogger(__name__)

FLAT_COLOR = "#101010"
LENGTH_COLORS = {
    20.0: "#46085c",
    40.0: "#355f8d",
    80.0: "#1f958b",
    160.0: "#4ec36b",
}
LENGTH_STYLES = {
    20.0: (0, (5, 2)),
    40.0: (0, (1.4, 1.6)),
    80.0: (0, (6, 1.8, 1.2, 1.8)),
    160.0: "solid",
}
AXIS_COLORS = {0.0: "#0b6ea8", np.pi: "#c1481a"}
AXIS_LABELS = {0.0: r"source axis  $\varphi=0$", np.pi: r"antipode  $\varphi=\pi$"}
FIELD_MAP = "RdBu_r"

PHOTON_HALF_ORBIT = 0.5 * PHOTON_SPHERE_PERIOD


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    return path


def _panel_tag(axis, text: str, location: str = "left") -> None:
    axis.set_title(text, loc=location, fontsize=10.5, pad=6)


# --------------------------------------------------------------------------
# Figure 1: the emitter
# --------------------------------------------------------------------------


def plot_source_definition(output_dir: Path) -> Path:
    """Show the emitter profiles, its angular spectrum, and the mode table."""

    _style()
    source = BROAD_SOURCE
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.2))

    axis = axes[0, 0]
    killing = np.linspace(
        source.time_center - 1.6 * source.time_half_width,
        source.time_center + 1.6 * source.time_half_width,
        800,
    )
    axis.plot(killing, time_profile(killing, source), color="#0b6ea8", linewidth=2.0)
    axis.axvspan(*source.killing_time_support, color="#0b6ea8", alpha=0.08, lw=0)
    axis.set_xlabel(r"Killing time $t/M$")
    axis.set_ylabel(r"$T(t)$")
    _panel_tag(axis, r"(a) temporal factor, compact in $t$")
    axis.annotate(
        "evaluated at\n" r"$t=\tau-h_L(r)$",
        xy=(source.time_center, 1.0),
        xytext=(source.time_center + 3.4, 0.72),
        fontsize=9.5,
        ha="left",
        color="#333333",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
    )

    axis = axes[0, 1]
    radius = np.linspace(2.0, 10.0, 900)
    axis.plot(radius, radial_profile(radius, source), color="#0b6ea8", linewidth=2.0)
    axis.axvspan(*source.radial_support, color="#0b6ea8", alpha=0.08, lw=0)
    for location, label in ((2.0, r"$r_b=2M$"), (3.0, "photon sphere"), (6.0, r"$r_{\rm s}$")):
        axis.axvline(location, color="#777777", linewidth=0.8, linestyle=":")
        axis.text(
            location + 0.12,
            0.93,
            label,
            rotation=90,
            va="top",
            fontsize=8.6,
            color="#555555",
        )
    axis.set_xlabel(r"areal radius $r/M$")
    axis.set_ylabel(r"$R(r)$")
    _panel_tag(axis, r"(b) radial factor, compact in $r$")

    axis = axes[1, 0]
    angle = np.linspace(-np.pi, np.pi, 900)
    axis.plot(
        angle / np.pi,
        angular_profile(np.cos(angle), source),
        color="#0b6ea8",
        linewidth=2.0,
        label=rf"broad, $\sigma={source.angular_width:.2f}$",
    )
    axis.plot(
        angle / np.pi,
        angular_profile(np.cos(angle), NARROW_SOURCE),
        color="#c1481a",
        linewidth=1.6,
        linestyle=(0, (5, 2)),
        label=rf"narrow, $\sigma={NARROW_SOURCE.angular_width:.2f}$",
    )
    axis.set_xlim(-0.5, 0.5)
    axis.set_xlabel(r"angle from the emitter $\gamma/\pi$")
    axis.set_ylabel(r"$\hat\Omega(\gamma)$")
    axis.legend(loc="upper right", fontsize=9.2)
    _panel_tag(axis, r"(c) angular factor, unit integral")

    axis = axes[1, 1]
    for label, model, color, marker in (
        ("broad", source, "#0b6ea8", "o"),
        ("narrow", NARROW_SOURCE, "#c1481a", "s"),
    ):
        ells = np.arange(0, 29)
        weights = angular_spectral_weights(model.angular_concentration, 28)
        axis.semilogy(
            ells,
            np.maximum(weights, 1e-12),
            color=color,
            marker=marker,
            markersize=3.4,
            linewidth=1.4,
            label=rf"{label}, $\kappa={model.angular_concentration:g}$",
        )
    axis.axvline(16, color="#0b6ea8", linewidth=0.9, linestyle=":")
    axis.axvline(24, color="#c1481a", linewidth=0.9, linestyle=":")
    axis.text(
        16.4, 2e-11, r"$\ell_{\max}=16$", fontsize=8.8, color="#0b6ea8",
        rotation=90, va="bottom",
    )
    axis.text(
        24.4, 2e-11, r"$\ell_{\max}=24$", fontsize=8.8, color="#c1481a",
        rotation=90, va="bottom",
    )
    axis.set_xlabel(r"harmonic index $\ell$")
    axis.set_ylabel(r"$g_\ell=i_\ell(\kappa)/i_0(\kappa)$")
    axis.set_ylim(1e-12, 3.0)
    axis.legend(loc="lower left", fontsize=9.2)
    _panel_tag(axis, r"(d) exact angular spectrum")

    figure.suptitle(
        "Localized emitter for the retarded Green function: "
        r"$\nabla^a\nabla_a\Phi=S$ from rest, $r_{\rm s}=6M$ in the equatorial plane",
        fontsize=11.8,
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    path = Path(output_dir) / "green_source_definition.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Figure 2: the equatorial field
# --------------------------------------------------------------------------


def plot_caustic_field(
    output_dir: Path,
    case: str | float = "schwarzschild",
    times: tuple[float, ...] = (26.0, 32.0, 36.0, 40.0, 44.0, 50.0, 58.0, 72.0),
) -> Path:
    """Polar maps of the reduced field in the equatorial plane."""

    _style()
    result = load_case(output_dir, case)
    phi = np.linspace(0.0, 2.0 * np.pi, 241)
    reduced = np.einsum(
        "tmr,mp->trp",
        result.modal_snapshots,
        harmonic_matrix(result, np.full_like(phi, 0.5 * np.pi), phi),
    )
    available = result.snapshot_times
    columns = 4
    rows = int(np.ceil(len(times) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.05 * columns, 3.35 * rows),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes).ravel()
    rho = result.snapshot_rho
    scale = float(np.percentile(np.abs(reduced), 99.85))
    # One shared symmetric-log scale: the wavefront weakens by orders of
    # magnitude between panels, and a linear scale would show only the first.
    norm = SymLogNorm(
        linthresh=0.01 * scale, linscale=0.4, vmin=-scale, vmax=scale, base=10
    )
    mesh = None
    horizon_label = (
        r"$\mathscr{I}^+$" if case == "schwarzschild" else r"$\mathcal{H}^+_c$"
    )
    if case == "schwarzschild":
        photon_sphere = 1.0 / 3.0
        emitter_rho = 1.0 - 2.0 / BROAD_SOURCE.center_radius
    else:
        inner = float(result.metadata["horizons"]["black_hole"])
        outer = float(result.metadata["horizons"]["cosmological"])
        photon_sphere = (1.0 - inner / 3.0) / (1.0 - inner / outer)
        emitter_rho = (1.0 - inner / BROAD_SOURCE.center_radius) / (
            1.0 - inner / outer
        )
    for axis, target in zip(axes, times):
        index = int(np.argmin(np.abs(available - target)))
        mesh = axis.pcolormesh(
            phi,
            rho,
            reduced[index],
            cmap=FIELD_MAP,
            norm=norm,
            shading="gouraud",
            rasterized=True,
        )
        axis.plot(
            np.linspace(0, 2 * np.pi, 200),
            np.full(200, photon_sphere),
            color="#333333",
            linewidth=0.8,
            linestyle=":",
        )
        axis.set_rmax(1.0)
        axis.set_rticks([])
        axis.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        axis.set_xticklabels(["", r"$\pi/2$", r"$\pi$", ""], fontsize=8.2)
        axis.grid(color="#cccccc", linewidth=0.4, alpha=0.7)
        # Mark where the emitter sits on this slice.
        axis.plot(
            [0.0], [emitter_rho], marker="*", markersize=8.0,
            color="#111111", linestyle="none",
        )
        axis.set_title(rf"$\tau={available[index]:.0f}M$", fontsize=10.4, pad=5)
    for axis in axes[len(times) :]:
        axis.set_visible(False)
    bar = figure.colorbar(
        mesh, ax=axes.tolist(), fraction=0.022, pad=0.035, extend="both"
    )
    bar.set_label(r"reduced field $u=r\Phi$  (equatorial plane)")
    bar.set_ticks([-1.0, -0.1, 0.0, 0.1, 1.0])
    figure.suptitle(
        f"{case_title(case)}: equatorial wavefront in the compactified radial "
        rf"coordinate $\rho$ — centre is the horizon, rim is {horizon_label}, "
        "dotted circle is the photon sphere",
        fontsize=11.4,
        y=0.995,
    )
    path = Path(output_dir) / f"caustic_field_{case_label(case)}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Figure 3: the waterfall at the outer boundary
# --------------------------------------------------------------------------


def _caustic_ridges(
    reference_time: float, phi: np.ndarray, windings: int
) -> list[np.ndarray]:
    """Photon-sphere arrival estimates for each winding branch."""

    slope = PHOTON_SPHERE_PERIOD / (2.0 * np.pi)
    ridges = []
    for number in range(windings):
        ridges.append(reference_time + slope * (2.0 * np.pi * number + phi))
        ridges.append(
            reference_time + slope * (2.0 * np.pi * (number + 1) - phi)
        )
    return ridges


def plot_caustic_waterfall(
    output_dir: Path, case: str | float = "schwarzschild"
) -> Path:
    """Angle-resolved waveform at the outer boundary with echo identification."""

    _style()
    result = load_case(output_dir, case)
    phi = np.linspace(0.0, 2.0 * np.pi, 361)
    times, field = equatorial_waveform(result, phi)
    limit = 150.0
    inside = (times >= 0.0) & (times <= limit)
    pulses = find_caustic_pulses(result, end=limit)

    figure = plt.figure(figsize=(12.2, 7.4))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.35, 1.0), hspace=0.32)
    axis = figure.add_subplot(grid[0])
    scale = float(np.percentile(np.abs(field[inside]), 99.6))
    # The echo amplitude falls by orders of magnitude across the window, so a
    # symmetric logarithmic norm is needed for the later crossings to be
    # visible at all next to the direct pulse.
    mesh = axis.pcolormesh(
        times[inside],
        phi / np.pi,
        field[inside].T,
        cmap=FIELD_MAP,
        norm=SymLogNorm(
            linthresh=0.01 * scale, linscale=0.4, vmin=-scale, vmax=scale, base=10
        ),
        shading="gouraud",
        rasterized=True,
    )
    direct = min(pulses, key=lambda pulse: pulse.time).time
    for number, ridge in enumerate(_caustic_ridges(direct, phi, 3)):
        axis.plot(
            ridge,
            phi / np.pi,
            color="#111111",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            alpha=0.75,
            label="photon-sphere prediction" if number == 0 else None,
        )
    axis.set_xlim(times[inside][0], limit)
    axis.set_ylim(0.0, 2.0)
    axis.set_ylabel(r"equatorial angle $\varphi/\pi$")
    axis.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axis.legend(loc="lower right", fontsize=9.2, labelcolor="#111111")
    _panel_tag(
        axis,
        r"(a) $u(U,\varphi)$ at the outer boundary; dashed lines are null rays "
        rf"winding on the $r=3M$ orbit, $\mathrm{{d}}U/\mathrm{{d}}\varphi=3\sqrt{{3}}M$",
    )
    bar = figure.colorbar(
        mesh,
        ax=axis,
        fraction=0.03,
        pad=0.012,
        extend="both",
        ticks=[-1.0, -0.1, 0.0, 0.1, 1.0],
    )
    bar.set_label(r"$u=r\Phi$")

    axis = figure.add_subplot(grid[1])
    for angle in (0.0, np.pi):
        _, trace = direction_waveform(result, angle)
        axis.plot(
            times,
            trace,
            color=AXIS_COLORS[angle],
            linewidth=1.5,
            linestyle="solid" if angle == 0.0 else (0, (5, 2)),
            label=AXIS_LABELS[angle],
        )
    # The crossings are marked by their measured time rather than by a point
    # on the curve: the envelope peak generally falls near a zero crossing of
    # the field, where a marker would sit misleadingly close to the axis.
    for pulse in pulses:
        axis.axvline(
            pulse.time,
            color=AXIS_COLORS[pulse.phi],
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            alpha=0.75,
        )
        axis.annotate(
            str(pulse.index),
            xy=(pulse.time, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9.0,
            color=AXIS_COLORS[pulse.phi],
        )
    axis.set_yscale("symlog", linthresh=0.05, linscale=0.45)
    axis.set_yticks([-10.0, -1.0, 0.0, 1.0, 10.0])
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.set_xlim(times[inside][0], limit)
    axis.set_xlabel(r"geometric retarded time $U/M$")
    axis.set_ylabel(r"$u=r\Phi$")
    axis.legend(loc="upper right", fontsize=9.4)
    _panel_tag(
        axis,
        "(b) the caustic sequence alternates between the two directions; "
        rf"successive markers are one half orbit ($\pi\cdot 3\sqrt{{3}} M={PHOTON_HALF_ORBIT:.2f}M$) apart",
    )
    figure.suptitle(
        f"{case_title(case)}: direct signal and caustic echoes of a localized "
        "emitter at $r=6M$",
        fontsize=12.0,
        y=0.975,
    )
    path = Path(output_dir) / f"caustic_waterfall_{case_label(case)}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Figure 4: quantitative echo structure
# --------------------------------------------------------------------------


def plot_echo_structure(output_dir: Path) -> tuple[Path, list[dict], list[dict]]:
    """Timing, amplitude decay, and phase relation of the caustic sequence."""

    _style()
    result = load_case(output_dir, "schwarzschild")
    pulses = find_caustic_pulses(result, end=150.0)
    phases = echo_phase_shifts(result, pulses)
    times = result.retarded_time

    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))

    axis = axes[0, 0]
    numbers = np.asarray([pulse.index for pulse in pulses], dtype=float)
    arrivals = np.asarray([pulse.time for pulse in pulses])
    slope, intercept = np.polyfit(numbers, arrivals, 1)
    for pulse in pulses:
        axis.plot(
            pulse.index,
            pulse.time,
            marker="o" if pulse.phi == 0.0 else "s",
            markersize=6.0,
            color=AXIS_COLORS[pulse.phi],
            linestyle="none",
        )
    axis.plot(
        numbers,
        slope * numbers + intercept,
        color="#111111",
        linewidth=1.2,
        label=rf"fit: $\Delta U={slope:.2f}M$ per crossing",
    )
    axis.plot(
        numbers,
        PHOTON_HALF_ORBIT * numbers + intercept,
        color="#111111",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label=rf"$\pi\cdot3\sqrt{{3}} M={PHOTON_HALF_ORBIT:.2f}M$",
    )
    axis.set_xlabel("caustic crossing number")
    axis.set_ylabel(r"arrival $U/M$")
    axis.legend(loc="upper left", fontsize=9.2)
    _panel_tag(axis, "(a) arrival times are equally spaced")

    axis = axes[0, 1]
    magnitudes = np.asarray([pulse.envelope for pulse in pulses])
    axis.semilogy(
        numbers[1:],
        magnitudes[1:],
        marker="o",
        markersize=5.4,
        linewidth=1.3,
        color="#0b6ea8",
        label="caustic crossings",
    )
    axis.semilogy(
        numbers[:1],
        magnitudes[:1],
        marker="o",
        markersize=6.4,
        markerfacecolor="white",
        color="#0b6ea8",
        linestyle="none",
        label="direct signal",
    )
    decay = np.polyfit(numbers[1:], np.log(magnitudes[1:]), 1)
    axis.semilogy(
        numbers,
        np.exp(np.polyval(decay, numbers)),
        color="#111111",
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        label=rf"$e^{{{decay[0]:.3f}n}}$",
    )
    axis.set_xlabel("caustic crossing number")
    axis.set_ylabel(r"envelope $|u+i\mathcal{H}[u]|$ at the peak")
    axis.legend(loc="lower left", fontsize=9.2)
    _panel_tag(axis, "(b) the first caustic exceeds the direct signal")

    axis = axes[1, 0]
    series = [("broad emitter", phases, "#0b6ea8", "o")]
    try:
        narrow = load_narrow(output_dir, "schwarzschild")
    except FileNotFoundError:
        narrow = None
    else:
        series.append(
            (
                "sharpened emitter",
                echo_phase_shifts(narrow, find_caustic_pulses(narrow, end=150.0)),
                "#c1481a",
                "s",
            )
        )
    for label, rows, color, marker in series:
        if not rows:
            continue
        axis.plot(
            np.arange(1, len(rows) + 1),
            [row["phase_over_half_pi"] for row in rows],
            marker=marker,
            markersize=6.0,
            linewidth=1.3,
            color=color,
            label=label,
        )
    for level in (-1.0, 1.0):
        axis.axhline(level, color="#111111", linewidth=1.0, linestyle=(0, (4, 2)))
    axis.text(
        1.0,
        1.05,
        r"geometric-optics Gouy shift $\pm\pi/2$",
        va="bottom",
        fontsize=9.0,
        color="#111111",
    )
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.set_ylim(-2.0, 2.0)
    if phases:
        axis.set_xticks(np.arange(1, len(phases) + 1))
    axis.set_xlabel("consecutive caustic pair")
    axis.set_ylabel(r"measured phase / $(\pi/2)$")
    axis.legend(loc="lower left", fontsize=9.0)
    _panel_tag(axis, "(c) phase accumulated at each caustic")

    axis = axes[1, 1]
    if len(pulses) >= 3 and len(phases) >= 2:
        step = float(np.median(np.diff(times)))
        span = int(round(9.0 / step))
        first, second = pulses[1], pulses[2]
        measured = phases[1]["phase_over_half_pi"] * 0.5 * np.pi
        _, trace_first = direction_waveform(result, first.phi)
        _, trace_second = direction_waveform(result, second.phi)
        centre_first = int(np.argmin(np.abs(times - first.time)))
        centre_second = int(np.argmin(np.abs(times - second.time)))
        offsets = (np.arange(-span, span + 1)) * step
        window_first = trace_first[centre_first - span : centre_first + span + 1]
        window_second = trace_second[centre_second - span : centre_second + span + 1]
        # Rotating the analytic signal of the earlier pulse by the measured
        # phase turns the number in panel (c) into a curve that can be laid
        # over the later pulse; nothing here is fitted to the overlay.
        rotated = np.real(np.exp(1j * measured) * hilbert(window_first))
        axis.plot(
            offsets,
            window_second / np.max(np.abs(window_second)),
            color="#c1481a",
            linewidth=2.2,
            label=f"crossing {second.index}",
        )
        axis.plot(
            offsets,
            window_first / np.max(np.abs(window_first)),
            color="#0b6ea8",
            linewidth=1.4,
            linestyle=(0, (1.4, 1.6)),
            label=f"crossing {first.index}",
        )
        axis.plot(
            offsets,
            rotated / np.max(np.abs(rotated)),
            color="#111111",
            linewidth=1.3,
            linestyle=(0, (5, 2)),
            label=(
                f"crossing {first.index} rotated by "
                rf"${measured / np.pi:.2f}\pi$"
            ),
        )
        axis.set_xlabel(r"$U-U_{\rm peak}$  $(M)$")
        axis.set_ylabel("normalized pulse")
        axis.legend(loc="lower left", fontsize=8.8)
    _panel_tag(axis, "(d) each echo is the previous one, phase rotated")

    figure.suptitle(
        "Schwarzschild caustic echoes: geometric-optics timing, decay, and phase",
        fontsize=12.0,
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    path = Path(output_dir) / "caustic_echo_structure.png"
    figure.savefig(path)
    plt.close(figure)
    rows = [pulse.as_dict() for pulse in pulses]
    for row, fitted in zip(rows, slope * numbers + intercept):
        row["fitted_arrival_over_M"] = float(fitted)
    return path, rows, phases


# --------------------------------------------------------------------------
# Figure 5: independent validation of the source term
# --------------------------------------------------------------------------


def plot_source_validation(
    output_dir: Path,
    modes: tuple[tuple[int, int], ...] = ((0, 0), (2, 2), (5, 3)),
    grids: tuple[int, ...] = (4201, 8401, 16801),
) -> tuple[Path, list[dict]]:
    """Cross-check the hyperboloidal source against a static-coordinate solve."""

    _style()
    result = load_case(output_dir, "schwarzschild")
    catalogue = build_mode_catalogue(
        BROAD_SOURCE, int(result.metadata["numerical"]["angular_ell_max"])
    )
    mode_indices = tuple(
        int(np.flatnonzero((result.mode_ell == ell) & (result.mode_m == order))[0])
        for ell, order in modes
    )
    if not (
        np.array_equal(result.mode_ell, catalogue.ell)
        and np.array_equal(result.mode_m, catalogue.m)
    ):
        raise ValueError("The stored mode ordering differs from the catalogue.")
    observer = 8.0
    parameters = SchwarzschildScalarParameters(mass=1.0, ell=0)
    height = float(minimal_height(np.asarray(observer), parameters, 4.0))
    observer_index = int(
        np.argmin(np.abs(result.observer_areal_radius - observer))
    )
    killing_times = result.signal_times - height

    rows: list[dict] = []
    curves: dict[int, dict] = {}
    for points in grids:
        grid = StaticReferenceGrid(points=points)
        limit = reflection_free_time(grid, BROAD_SOURCE, observer)
        for index in mode_indices:
            ell = int(catalogue.ell[index])
            order = int(catalogue.m[index])
            reference = solve_static_mode(
                ell=ell,
                mode_amplitude=float(catalogue.amplitude[index]),
                source=BROAD_SOURCE,
                observer_radii=(observer,),
                end_time=min(float(killing_times[-1]), limit),
                grid=grid,
            )
            static_times = reference["times"]
            static_signal = reference["signals"][:, 0]
            window = (static_times >= 15.0) & (static_times <= limit)
            hyperboloidal = np.interp(
                static_times[window], killing_times, result.expanded_modal_signals()[:, observer_index, index]
            )
            difference = hyperboloidal - static_signal[window]
            relative = float(
                np.linalg.norm(difference) / np.linalg.norm(static_signal[window])
            )
            rows.append(
                {
                    "ell": ell,
                    "m": order,
                    "static_grid_points": points,
                    "static_spacing": float(grid.spacing),
                    "reflection_free_time_over_M": float(limit),
                    "relative_l2_difference": relative,
                    "maximum_absolute_difference": float(np.max(np.abs(difference))),
                    "reference_peak": float(np.max(np.abs(static_signal[window]))),
                }
            )
            if points == grids[-1]:
                curves[index] = {
                    "ell": ell,
                    "m": order,
                    "times": static_times[window],
                    "static": static_signal[window],
                    "hyperboloidal": hyperboloidal,
                }

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    palette = ["#0b6ea8", "#1f958b", "#c1481a"]
    axis = axes[0]
    for color, (index, data) in zip(palette, curves.items()):
        axis.plot(
            data["times"],
            data["static"],
            color=color,
            linewidth=4.5,
            alpha=0.30,
            solid_capstyle="butt",
        )
        axis.plot(
            data["times"],
            data["hyperboloidal"],
            color=color,
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            label=rf"$\ell={data['ell']},\,m={data['m']}$",
        )
    axis.set_xlim(15.0, 160.0)
    axis.set_xlabel(r"Killing time $t/M$")
    axis.set_ylabel(r"$u_{\ell m}$ at $r=8M$")
    axis.legend(loc="upper right", fontsize=9.2)
    _panel_tag(axis, "(a) thick: static solve, dashed: hyperboloidal")

    axis = axes[1]
    peak = 0.0
    for color, (index, data) in zip(palette, curves.items()):
        residual = np.abs(data["hyperboloidal"] - data["static"])
        peak = max(peak, float(np.max(residual)))
        # The residual is identically zero before the emitter switches on;
        # those samples carry no information on a logarithmic axis.
        axis.semilogy(
            data["times"],
            np.where(residual > 0.0, residual, np.nan),
            color=color,
            linewidth=1.2,
            label=rf"$\ell={data['ell']}$",
        )
    axis.set_ylim(1e-7 * peak, 6.0 * peak)
    axis.set_xlim(15.0, 160.0)
    axis.set_xlabel(r"Killing time $t/M$")
    axis.set_ylabel("absolute difference")
    axis.legend(loc="upper right", fontsize=9.2, ncol=3)
    _panel_tag(axis, "(b) pointwise residual")

    axis = axes[2]
    for color, index in zip(palette, mode_indices):
        selected = [row for row in rows if row["ell"] == int(catalogue.ell[index])]
        spacing = np.asarray([row["static_spacing"] for row in selected])
        error = np.asarray([row["relative_l2_difference"] for row in selected])
        axis.loglog(
            spacing,
            error,
            marker="o",
            markersize=5.4,
            linewidth=1.3,
            color=color,
            label=rf"$\ell={int(catalogue.ell[index])}$",
        )
    coarsest = max(row["relative_l2_difference"] for row in rows)
    guide = np.asarray([0.025, 0.1])
    axis.loglog(
        guide,
        1.6 * coarsest * (guide / 0.1) ** 2,
        color="#111111",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label=r"second order",
    )
    axis.set_xlabel(r"static-grid spacing $\Delta r_*/M$")
    axis.set_ylabel(r"relative $L^2$ difference")
    axis.legend(loc="lower right", fontsize=9.2)
    _panel_tag(axis, "(c) refining the static reference")

    figure.suptitle(
        "Independent verification of the source term: hyperboloidal bridge "
        "evolution against a static-coordinate leapfrog solve",
        fontsize=11.8,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output_dir) / "source_term_validation.png"
    figure.savefig(path)
    plt.close(figure)
    return path, rows


# --------------------------------------------------------------------------
# Figure 6 and 7: the flat limit
# --------------------------------------------------------------------------


def plot_flat_limit_waveforms(output_dir: Path) -> Path:
    """Compare the SdS Green function with the Schwarzschild one."""

    _style()
    reference = load_case(output_dir, "schwarzschild")
    candidates = {
        length: load_case(output_dir, length) for length in COSMOLOGICAL_LENGTHS
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(12.6, 7.2), sharex="col", height_ratios=(1.35, 1.0)
    )
    window = VALIDATION_WINDOW
    grid = np.linspace(window[0], window[1], 3001)
    for column, angle in enumerate((0.0, np.pi)):
        exact_times, exact = direction_waveform(reference, angle)
        top, bottom = axes[0, column], axes[1, column]
        top.plot(
            exact_times,
            exact,
            color=FLAT_COLOR,
            linewidth=2.2,
            alpha=0.9,
            label=r"Schwarzschild $\mathscr{I}^+$",
        )
        exact_grid = np.interp(grid, exact_times, exact)
        for length, result in candidates.items():
            times, trace = direction_waveform(result, angle)
            top.plot(
                times,
                trace,
                color=LENGTH_COLORS[length],
                linestyle=LENGTH_STYLES[length],
                linewidth=1.35,
                label=rf"SdS $L/M={length:g}$",
            )
            difference = np.abs(np.interp(grid, times, trace) - exact_grid)
            # Both signals vanish identically before the emitter switches on.
            bottom.semilogy(
                grid,
                np.where(difference > 0.0, difference, np.nan),
                color=LENGTH_COLORS[length],
                linestyle=LENGTH_STYLES[length],
                linewidth=1.25,
            )
        top.set_xlim(window)
        top.axhline(0.0, color="#aaaaaa", linewidth=0.7)
        top.set_ylabel(r"$u=r\Phi$")
        _panel_tag(
            top,
            f"({'ac'[column]}) {AXIS_LABELS[angle]}",
        )
        bottom.set_xlim(window)
        bottom.set_ylim(1e-4, 20.0)
        bottom.set_xlabel(r"geometric retarded time $U=\tau-q_L$  $(M)$")
        bottom.set_ylabel(r"$|u_L-u_0|$")
        _panel_tag(bottom, f"({'bd'[column]}) pointwise difference")
    axes[0, 0].legend(loc="lower right", fontsize=8.8, ncol=2)
    figure.suptitle(
        "One physical emitter, four cosmological lengths: the SdS Green "
        r"function at $\mathcal{H}^+_c$ converges to the Schwarzschild one at "
        r"$\mathscr{I}^+$",
        fontsize=12.0,
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    path = Path(output_dir) / "sds_flat_limit_waveforms.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_flat_limit_convergence(
    output_dir: Path,
) -> tuple[Path, list[dict], list[dict]]:
    """Flat-limit norms and caustic timing shifts against ``L/M``."""

    _style()
    reference = load_case(output_dir, "schwarzschild")
    reference_pulses = find_caustic_pulses(reference, end=VALIDATION_WINDOW[1])
    norm_rows: list[dict] = []
    timing_rows: list[dict] = []
    for length in COSMOLOGICAL_LENGTHS:
        result = load_case(output_dir, length)
        row = {"cosmological_length_over_M": length}
        row.update(flat_limit_norms(reference, result))
        row["retarded_time_offset_q_over_M"] = float(
            result.metadata["retarded_time_offset"]["q"]
        )
        row["surface_gravity_kappa_c_M"] = float(
            result.metadata["surface_gravity_cosmological"]
        )
        row["maximum_constraint"] = float(np.max(result.constraint_linf))
        norm_rows.append(row)
        pulses = find_caustic_pulses(result, end=VALIDATION_WINDOW[1])
        for entry in pulse_timing_shifts(reference_pulses, pulses):
            entry["cosmological_length_over_M"] = length
            timing_rows.append(entry)

    lengths = np.asarray([row["cosmological_length_over_M"] for row in norm_rows])
    errors = np.asarray([row["relative_l2"] for row in norm_rows])
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.05))

    axis = axes[0]
    axis.loglog(
        lengths,
        errors,
        marker="o",
        markersize=6.5,
        linewidth=1.5,
        color="#0b6ea8",
        label=r"relative $L^2$",
    )
    axis.loglog(
        lengths,
        np.asarray([row["relative_linf"] for row in norm_rows]),
        marker="s",
        markersize=5.6,
        linewidth=1.3,
        linestyle=(0, (4, 2)),
        color="#c1481a",
        label=r"relative $L^\infty$",
    )
    exponent = np.polyfit(np.log(lengths), np.log(errors), 1)[0]
    axis.loglog(
        lengths,
        errors[0] * (lengths / lengths[0]) ** -1.0,
        color="#111111",
        linewidth=1.0,
        linestyle=":",
        label=r"$\propto M/L$",
    )
    axis.set_xlabel(r"$L/M$")
    axis.set_ylabel("difference from Schwarzschild")
    axis.set_xticks(lengths)
    axis.set_xticklabels([f"{value:g}" for value in lengths])
    axis.legend(loc="lower left", fontsize=9.2)
    _panel_tag(axis, rf"(a) fitted slope ${exponent:.2f}$ on $\mathcal{{J}}$")

    axis = axes[1]
    for length in COSMOLOGICAL_LENGTHS:
        selected = [
            row for row in timing_rows if row["cosmological_length_over_M"] == length
        ]
        if not selected:
            continue
        axis.plot(
            [row["pulse"] for row in selected],
            [row["timing_shift_over_M"] for row in selected],
            marker="o",
            markersize=5.4,
            linewidth=1.3,
            color=LENGTH_COLORS[length],
            linestyle=LENGTH_STYLES[length],
            label=rf"$L/M={length:g}$",
        )
    axis.axhline(0.0, color="#111111", linewidth=0.9)
    axis.set_xlabel("caustic crossing number")
    axis.set_ylabel(r"$U_{\rm SdS}-U_{\rm Schw}$  $(M)$")
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.legend(loc="upper left", fontsize=9.0)
    _panel_tag(axis, "(b) cosmological shift of the echo timing")

    axis = axes[2]
    for length in COSMOLOGICAL_LENGTHS:
        selected = [
            row for row in timing_rows if row["cosmological_length_over_M"] == length
        ]
        if not selected:
            continue
        axis.plot(
            [row["pulse"] for row in selected],
            [row["envelope_ratio"] for row in selected],
            marker="o",
            markersize=5.4,
            linewidth=1.3,
            color=LENGTH_COLORS[length],
            linestyle=LENGTH_STYLES[length],
        )
    axis.axhline(1.0, color="#111111", linewidth=0.9)
    axis.set_xlabel("caustic crossing number")
    axis.set_ylabel(r"envelope ratio $\rm SdS/Schw$ at the peak")
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _panel_tag(axis, "(c) cosmological shift of the echo amplitude")

    figure.suptitle(
        "Quantified flat limit of the retarded Green function on the "
        rf"validated window $\mathcal{{J}}=[{VALIDATION_WINDOW[0]:g}M,\,{VALIDATION_WINDOW[1]:g}M]$",
        fontsize=11.8,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output_dir) / "sds_flat_limit_convergence.png"
    figure.savefig(path)
    plt.close(figure)
    return path, norm_rows, timing_rows


# --------------------------------------------------------------------------
# Figure 8: late time
# --------------------------------------------------------------------------


def _mode_amplitude(result: SourcedSimulationResult, ell: int) -> np.ndarray:
    """Return the ``m``-summed amplitude of one harmonic at the outer boundary."""

    index = result.outer_index()
    selected = result.expanded_modal_signals()[:, index, result.mode_ell == ell]
    return np.sqrt(np.sum(selected**2, axis=1))


def plot_late_time(output_dir: Path) -> tuple[Path, list[dict]]:
    """The cosmological end state: a constant monopole and ``gamma/kappa_c=l``."""

    _style()
    reference = load_case(output_dir, "schwarzschild")
    results = {length: load_case(output_dir, length) for length in COSMOLOGICAL_LENGTHS}
    figure, axes = plt.subplots(1, 4, figsize=(16.4, 4.15))

    axis = axes[0]
    monopole_index = int(np.flatnonzero(reference.mode_ell == 0)[0])
    outer = reference.outer_index()
    # The field vanishes identically before the emitter switches on; those
    # samples are dropped rather than plotted at log of zero.
    flat_monopole = np.abs(reference.expanded_modal_signals()[:, outer, monopole_index])
    axis.semilogy(
        reference.retarded_time,
        np.where(flat_monopole > 0.0, flat_monopole, np.nan),
        color=FLAT_COLOR,
        linewidth=2.0,
        label=r"Schwarzschild $\mathscr{I}^+$",
    )
    rows: list[dict] = []
    for length, result in results.items():
        index = int(np.flatnonzero(result.mode_ell == 0)[0])
        times = result.retarded_time
        trace = result.expanded_modal_signals()[:, result.outer_index(), index]
        magnitude = np.abs(trace)
        axis.semilogy(
            times,
            np.where(magnitude > 0.0, magnitude, np.nan),
            color=LENGTH_COLORS[length],
            linestyle=LENGTH_STYLES[length],
            linewidth=1.4,
            label=rf"$L/M={length:g}$",
        )
        tail = trace[times > 0.75 * times[-1]]
        rows.append(
            {
                "cosmological_length_over_M": length,
                "monopole_final_value": float(trace[-1]),
                "monopole_tail_mean": float(np.mean(tail)),
                "monopole_tail_relative_drift": float(
                    np.std(tail) / max(abs(np.mean(tail)), 1e-300)
                ),
            }
        )
    axis.set_xlim(0.0, 600.0)
    axis.set_ylim(1e-4, 40.0)
    axis.set_xlabel(r"$U/M$")
    axis.set_ylabel(r"$|u_{00}|$ at the outer boundary")
    axis.legend(loc="lower left", fontsize=9.0)
    _panel_tag(axis, "(a) the monopole freezes on SdS")

    for column, ell in ((1, 1), (2, 2)):
        axis = axes[column]
        settings = EnvelopeSettings(30.0 if ell == 1 else 45.0, 0.5)
        for length, result in results.items():
            kappa = float(result.metadata["surface_gravity_cosmological"])
            times = result.retarded_time
            amplitude = _mode_amplitude(result, ell)
            rate, _ = envelope_rate(times, amplitude, settings)
            axis.plot(
                kappa * times,
                rate / kappa,
                color=LENGTH_COLORS[length],
                linestyle=LENGTH_STYLES[length],
                linewidth=1.4,
                label=rf"$L/M={length:g}$",
            )
            # Quote the rate over the last quarter of the resolved record, so
            # a long run is not averaged together with its own approach.
            cosmological = kappa * times
            resolved = np.isfinite(rate) & (
                cosmological > max(2.5, 0.75 * float(cosmological[-1]))
            )
            if resolved.any():
                rows.append(
                    {
                        "cosmological_length_over_M": length,
                        "ell": ell,
                        "late_rate_over_kappa_c": float(
                            np.median((rate / kappa)[resolved])
                        ),
                        "quoted_window_kappa_c_U": [
                            float(cosmological[resolved][0]),
                            float(cosmological[resolved][-1]),
                        ],
                        "target": float(ell),
                    }
                )
        axis.axhline(float(ell), color="#111111", linewidth=1.0, linestyle=(0, (4, 2)))
        axis.axhspan(0.9 * ell, 1.1 * ell, color="#999999", alpha=0.16, lw=0)
        axis.set_xlim(0.5, 8.5)
        axis.set_ylim(-0.5, 2.0 * ell + 1.5)
        axis.set_xlabel(r"$\kappa_c U$")
        axis.set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
        if column == 1:
            axis.legend(loc="upper right", fontsize=9.0)
        _panel_tag(
            axis,
            rf"({'bc'[column - 1]}) $\ell={ell}$: target $\gamma/\kappa_c={ell}$",
        )

    axis = axes[3]
    try:
        refined = load_convergence(output_dir, "sds_L80_N1536")
    except FileNotFoundError:
        refined = None
    if refined is not None:
        production = results[80.0]
        kappa = float(production.metadata["surface_gravity_cosmological"])
        for ell, color in ((1, "#0b6ea8"), (2, "#c1481a")):
            for label, run, style, width in (
                (r"$N=1024$", production, "solid", 1.8),
                (r"$N=1536$", refined, (0, (4, 2)), 1.3),
            ):
                amplitude = _mode_amplitude(run, ell)
                axis.semilogy(
                    kappa * run.retarded_time,
                    np.where(amplitude > 0.0, amplitude, np.nan),
                    color=color,
                    linestyle=style,
                    linewidth=width,
                    label=rf"$\ell={ell}$, {label}",
                )
            floor = float(
                np.min(_mode_amplitude(production, ell)[production.retarded_time > 200.0])
            )
            rows.append(
                {
                    "cosmological_length_over_M": 80.0,
                    "ell": ell,
                    "production_late_minimum": floor,
                    "refined_late_minimum": float(
                        np.min(
                            _mode_amplitude(refined, ell)[
                                refined.retarded_time > 200.0
                            ]
                        )
                    ),
                }
            )
        axis.set_xlim(0.5, 3.8)
        axis.set_ylim(1e-9, 3e1)
        axis.legend(loc="upper right", fontsize=8.6, ncol=2)
    axis.set_xlabel(r"$\kappa_c U$")
    axis.set_ylabel(r"$\left[\sum_m u_{\ell m}^2\right]^{1/2}$ at $L/M=80$")
    _panel_tag(axis, "(d) what limits the quadrupole rate")

    figure.suptitle(
        "Cosmological end state of the mixed-mode Green function at the "
        r"cosmological horizon",
        fontsize=11.8,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output_dir) / "sds_late_time.png"
    figure.savefig(path)
    plt.close(figure)
    return path, rows


# --------------------------------------------------------------------------
# Figure 9: convergence
# --------------------------------------------------------------------------


def _outer_trace(result: SourcedSimulationResult, angle: float) -> tuple:
    return direction_waveform(result, angle)


def _relative_difference(
    coarse: SourcedSimulationResult,
    fine: SourcedSimulationResult,
    angle: float = np.pi,
    window: tuple[float, float] = (5.0, 150.0),
    samples: int = 3001,
) -> float:
    grid = np.linspace(*window, samples)
    coarse_times, coarse_trace = _outer_trace(coarse, angle)
    fine_times, fine_trace = _outer_trace(fine, angle)
    first = np.interp(grid, coarse_times, coarse_trace)
    second = np.interp(grid, fine_times, fine_trace)
    return float(np.linalg.norm(first - second) / np.linalg.norm(second))


def plot_convergence(output_dir: Path) -> tuple[Path, list[dict]]:
    """Radial, temporal, angular, and stencil refinement diagnostics."""

    _style()
    rows: list[dict] = []
    figure, axes = plt.subplots(1, 4, figsize=(15.2, 3.95))

    finest_radial = load_convergence(output_dir, f"radial_N{RADIAL_LADDER[-1]}")
    axis = axes[0]
    resolutions = np.asarray(RADIAL_LADDER[:-1], dtype=float)
    values = []
    for resolution in RADIAL_LADDER[:-1]:
        coarse = load_convergence(output_dir, f"radial_N{resolution}")
        difference = _relative_difference(coarse, finest_radial)
        values.append(difference)
        rows.append(
            {
                "ladder": "radial",
                "parameter": resolution,
                "relative_difference_to_finest": difference,
                "maximum_constraint": float(np.max(coarse.constraint_linf)),
            }
        )
    radial_floor = float(min(values))
    axis.loglog(
        resolutions, values, marker="o", markersize=6.0, linewidth=1.4, color="#0b6ea8"
    )
    guide = resolutions.astype(float)
    axis.loglog(
        guide,
        values[0] * (guide / guide[0]) ** -8.0,
        color="#111111",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="eighth order",
    )
    axis.set_xscale("linear")
    axis.set_xticks(resolutions)
    axis.set_xticklabels([f"{int(value)}" for value in resolutions])
    axis.set_xlim(440.0, 1100.0)
    axis.set_xlabel(r"radial points $N$")
    axis.set_ylabel(r"relative $L^2$ vs $N=1536$")
    axis.legend(loc="upper right", fontsize=9.0)
    _panel_tag(axis, r"(a) radial: saturates near $10^{-6}$")

    axis = axes[1]
    finest_step = load_convergence(output_dir, f"timestep_dt{TIMESTEP_LADDER[-1]:g}")
    steps = np.asarray(TIMESTEP_LADDER[:-1], dtype=float)
    values = []
    for step in TIMESTEP_LADDER[:-1]:
        coarse = load_convergence(output_dir, f"timestep_dt{step:g}")
        difference = _relative_difference(coarse, finest_step)
        values.append(difference)
        rows.append(
            {
                "ladder": "timestep",
                "parameter": step,
                "relative_difference_to_finest": difference,
                "maximum_constraint": float(np.max(coarse.constraint_linf)),
            }
        )
    axis.loglog(
        steps,
        values,
        marker="o",
        markersize=6.0,
        linewidth=1.4,
        color="#1f958b",
        label=r"timestep, vs $\Delta\tau=0.001M$",
    )
    # Plotted against the spatial floor, because the point of this ladder is
    # that the time integration contributes nothing at this working point.
    axis.axhline(
        radial_floor,
        color="#111111",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="radial floor",
    )
    axis.set_xscale("linear")
    axis.set_xticks(steps)
    axis.set_xticklabels([f"{value:g}" for value in steps])
    axis.set_ylim(3e-12, 3.0 * radial_floor)
    axis.set_xlabel(r"timestep $\Delta\tau/M$")
    axis.set_ylabel(r"relative $L^2$")
    axis.legend(loc="center left", fontsize=8.8)
    _panel_tag(axis, "(b) temporal error is negligible")

    axis = axes[2]
    finest_angular = load_convergence(output_dir, f"angular_lmax{ANGULAR_LADDER[-1]}")
    truncations = np.asarray(ANGULAR_LADDER[:-1], dtype=float)
    values = []
    for ell_max in ANGULAR_LADDER[:-1]:
        coarse = load_convergence(output_dir, f"angular_lmax{ell_max}")
        difference = _relative_difference(coarse, finest_angular)
        values.append(difference)
        rows.append(
            {
                "ladder": "angular",
                "parameter": ell_max,
                "relative_difference_to_finest": difference,
                "maximum_constraint": float(np.max(coarse.constraint_linf)),
            }
        )
    axis.semilogy(
        truncations,
        values,
        marker="o",
        markersize=6.0,
        linewidth=1.4,
        color="#c1481a",
    )
    axis.set_xticks(truncations)
    axis.set_xlabel(r"angular truncation $\ell_{\max}$")
    axis.set_ylabel(r"relative $L^2$ vs $\ell_{\max}=20$")
    _panel_tag(axis, "(c) angular truncation")

    axis = axes[3]
    stencil = load_convergence(output_dir, "stencil_order6")
    cross = _relative_difference(stencil, finest_radial)
    rows.append(
        {
            "ladder": "stencil",
            "parameter": 6,
            "relative_difference_to_finest": cross,
            "maximum_constraint": float(np.max(stencil.constraint_linf)),
        }
    )
    for case, color, style in (
        ("schwarzschild", FLAT_COLOR, "solid"),
        *(
            (length, LENGTH_COLORS[length], LENGTH_STYLES[length])
            for length in COSMOLOGICAL_LENGTHS
        ),
    ):
        result = load_case(output_dir, case)
        axis.semilogy(
            result.diagnostic_times,
            np.maximum(result.constraint_linf, 1e-18),
            color=color,
            linestyle=style,
            linewidth=1.3,
            label=case_title(case),
        )
        rows.append(
            {
                "ladder": "production",
                "parameter": case_label(case),
                "relative_difference_to_finest": np.nan,
                "maximum_constraint": float(np.max(result.constraint_linf)),
            }
        )
    axis.set_xlabel(r"bridge time $\tau/M$")
    axis.set_ylabel(r"$\|\psi-\partial_\rho u\|_\infty$")
    axis.legend(loc="lower right", fontsize=8.4)
    _panel_tag(axis, "(d) reduction constraint")

    figure.suptitle(
        "Refinement of the sourced evolution, measured on the antipodal "
        "waveform at the outer boundary",
        fontsize=11.8,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = Path(output_dir) / "green_convergence.png"
    figure.savefig(path)
    plt.close(figure)
    return path, rows


# --------------------------------------------------------------------------
# Figure 10: sharpening the emitter
# --------------------------------------------------------------------------


def plot_narrow_source(output_dir: Path) -> tuple[Path, list[dict]]:
    """Contrast the broad production emitter with the sharpened one."""

    _style()
    broad = load_case(output_dir, "schwarzschild")
    narrow = load_narrow(output_dir, "schwarzschild")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    axis = axes[0]
    rows: list[dict] = []
    for label, result, color, style in (
        (rf"broad, $\sigma={BROAD_SOURCE.angular_width:.2f}$", broad, "#0b6ea8", "solid"),
        (
            rf"narrow, $\sigma={NARROW_SOURCE.angular_width:.2f}$",
            narrow,
            "#c1481a",
            (0, (5, 2)),
        ),
    ):
        times, trace = direction_waveform(result, np.pi)
        axis.plot(
            times,
            trace / np.max(np.abs(trace)),
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=label,
        )
        pulses = find_caustic_pulses(result, end=150.0)
        for pulse in pulses:
            rows.append({"emitter": label.split(",")[0], **pulse.as_dict()})
    axis.set_xlim(20.0, 140.0)
    axis.set_xlabel(r"$U/M$")
    axis.set_ylabel("normalized antipodal waveform")
    axis.legend(loc="upper right", fontsize=9.2)
    _panel_tag(axis, "(a) the echoes sharpen")

    axis = axes[1]
    phi = np.linspace(0.0, 2.0 * np.pi, 481)
    for label, result, color, style in (
        ("broad", broad, "#0b6ea8", "solid"),
        ("narrow", narrow, "#c1481a", (0, (5, 2))),
    ):
        times, field = equatorial_waveform(result, phi)
        pulses = [
            pulse for pulse in find_caustic_pulses(result, end=150.0) if pulse.phi > 1.0
        ]
        index = int(np.argmin(np.abs(times - pulses[0].time)))
        profile = field[index]
        axis.plot(
            phi / np.pi,
            profile / np.max(np.abs(profile)),
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=label,
        )
    axis.axvline(1.0, color="#111111", linewidth=0.9, linestyle=":")
    axis.set_xlabel(r"equatorial angle $\varphi/\pi$")
    axis.set_ylabel("normalized field at the first caustic")
    axis.legend(loc="lower left", fontsize=9.2)
    _panel_tag(axis, "(b) the caustic focus narrows")

    axis = axes[2]
    for label, result, color, marker in (
        ("broad", broad, "#0b6ea8", "o"),
        ("narrow", narrow, "#c1481a", "s"),
    ):
        ells, power = modal_energy_spectrum(result, (5.0, 150.0))
        axis.semilogy(
            ells,
            power / power[0],
            color=color,
            marker=marker,
            markersize=4.4,
            linewidth=1.3,
            label=label,
        )
    axis.set_xlabel(r"harmonic index $\ell$")
    axis.set_ylabel(r"outgoing power, normalized to $\ell=0$")
    axis.legend(loc="lower left", fontsize=9.2)
    _panel_tag(axis, "(c) mixed-mode content at the boundary")

    figure.suptitle(
        "Sharpening the emitter: a narrower source approaches the point-source "
        "Green function",
        fontsize=11.8,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output_dir) / "narrow_source_comparison.png"
    figure.savefig(path)
    plt.close(figure)
    return path, rows


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _geometric_optics_reference(pulse_rows: list[dict]) -> dict:
    r"""Compare the measured caustic sequence with geometric optics.

    At future null infinity the normalized clock ``U = tau - q_0`` coincides
    with the ordinary retarded time ``t - r_*``, with ``r_*`` normalized at
    the same reference radius ``r_0 = 4M``.  The direct signal of an emitter
    centred at ``(t_0, r_{\rm s})`` therefore arrives at
    ``U = t_0 - r_*(r_{\rm s})``, and each further caustic crossing costs one
    half period of the ``r=3M`` photon orbit.
    """

    parameters = SchwarzschildScalarParameters(mass=1.0, ell=0)
    emitter_tortoise = float(
        tortoise_coordinate(np.asarray(BROAD_SOURCE.center_radius), parameters, 4.0)
    )
    predicted = BROAD_SOURCE.time_center - emitter_tortoise
    arrivals = np.asarray([row["U_over_M"] for row in pulse_rows], dtype=float)
    numbers = np.asarray([row["pulse"] for row in pulse_rows], dtype=float)
    spacing = float(np.polyfit(numbers, arrivals, 1)[0]) if arrivals.size > 1 else np.nan
    return {
        "emitter_tortoise_radius_over_M": emitter_tortoise,
        "predicted_direct_arrival_over_M": predicted,
        "measured_direct_arrival_over_M": float(arrivals[0]) if arrivals.size else np.nan,
        "photon_sphere_full_orbit_over_M": PHOTON_SPHERE_PERIOD,
        "photon_sphere_half_orbit_over_M": PHOTON_HALF_ORBIT,
        "measured_crossing_spacing_over_M": spacing,
        "spacing_relative_error": (
            spacing / PHOTON_HALF_ORBIT - 1.0 if np.isfinite(spacing) else np.nan
        ),
    }


def source_mode_rows(ell_max: int = 16) -> list[dict]:
    catalogue = build_mode_catalogue(BROAD_SOURCE, ell_max)
    return [
        {
            "ell": int(ell),
            "m": int(order),
            "source_amplitude": float(amplitude),
        }
        for ell, order, amplitude in zip(
            catalogue.ell, catalogue.m, catalogue.amplitude
        )
    ]


def run_summary_rows(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for case in ("schwarzschild", *COSMOLOGICAL_LENGTHS):
        result = load_case(output_dir, case)
        numerical = result.metadata["numerical"]
        rows.append(
            {
                "case": case_label(case),
                "background": result.metadata["background"],
                "radial_points": numerical["radial_resolution"],
                "ell_max": numerical["angular_ell_max"],
                "modes": int(result.mode_ell.size),
                "timestep": numerical["timestep"],
                "final_time_over_M": result.metadata["final_time"],
                "coordinate_cfl": result.metadata["radial_discretization"][
                    "coordinate_cfl"
                ],
                "retarded_time_offset_q_over_M": result.metadata[
                    "retarded_time_offset"
                ]["q"],
                "kappa_c_M": result.metadata["surface_gravity_cosmological"],
                "kappa_c_U_final": result.metadata["surface_gravity_cosmological"]
                * (
                    result.metadata["final_time"]
                    - result.metadata["retarded_time_offset"]["q"]
                ),
                "maximum_constraint": float(np.max(result.constraint_linf)),
                "wall_seconds": result.metadata["wall_seconds"],
            }
        )
    return rows


def create_report(output_dir: Path) -> list[Path]:
    """Regenerate every figure and table of the Green-function study."""

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    written: list[Path] = []

    written.append(plot_source_definition(output_dir))
    written.append(_write_rows(tables / "source_modes.csv", source_mode_rows()))
    written.append(plot_caustic_field(output_dir, "schwarzschild"))
    written.append(plot_caustic_field(output_dir, 80.0))
    written.append(plot_caustic_waterfall(output_dir, "schwarzschild"))

    path, pulse_rows, phase_rows = plot_echo_structure(output_dir)
    written.append(path)
    written.append(_write_rows(tables / "caustic_pulses.csv", pulse_rows))
    written.append(_write_rows(tables / "caustic_phase.csv", phase_rows))

    path, validation_rows = plot_source_validation(output_dir)
    written.append(path)
    written.append(_write_rows(tables / "source_validation.csv", validation_rows))

    written.append(plot_flat_limit_waveforms(output_dir))
    path, norm_rows, timing_rows = plot_flat_limit_convergence(output_dir)
    written.append(path)
    written.append(_write_rows(tables / "sds_flat_limit.csv", norm_rows))
    written.append(_write_rows(tables / "sds_pulse_timing.csv", timing_rows))

    path, late_rows = plot_late_time(output_dir)
    written.append(path)
    written.append(_write_rows(tables / "late_time.csv", late_rows))

    path, convergence_rows = plot_convergence(output_dir)
    written.append(path)
    written.append(_write_rows(tables / "convergence.csv", convergence_rows))

    try:
        path, narrow_rows = plot_narrow_source(output_dir)
    except FileNotFoundError:
        LOGGER.warning("no sharpened-emitter archive; skipping that comparison")
        narrow_rows = []
    else:
        written.append(path)
        written.append(_write_rows(tables / "narrow_source_pulses.csv", narrow_rows))

    written.append(_write_rows(tables / "run_summary.csv", run_summary_rows(output_dir)))

    summary = {
        "caustic_pulses": pulse_rows,
        "caustic_phase": phase_rows,
        "source_validation": validation_rows,
        "flat_limit": norm_rows,
        "pulse_timing": timing_rows,
        "late_time": late_rows,
        "convergence": convergence_rows,
        "runs": run_summary_rows(output_dir),
        "geometric_optics": _geometric_optics_reference(pulse_rows),
    }
    digest = output_dir / "green_function_summary.json"
    digest.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    written.append(digest)
    for item in written:
        LOGGER.info("wrote %s", item)
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/green_function")
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    create_report(arguments.output_dir)


if __name__ == "__main__":
    main()
