"""Diagnostic figures for the antipodal caustic.

These are measurement figures rather than presentation renders.  Everything is
drawn in physical areal radius on a symmetric linear scale that is held fixed
across a sequence, so brightness differences between panels are amplitude
differences in the solution.  The height and colour surface follows the
convention of the earlier Schwarzschild caustic study, where the same scalar
sets both the elevation and the colour of the sheet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .caustic_diagnostics import (  # noqa: E402
    DEFAULT_OUTER_RADIUS,
    HORIZON_RADIUS,
    PHOTON_SPHERE_RADIUS,
    axial_focus,
    equatorial_section,
    full_width_half_maximum,
    load,
    locate_focus,
    transverse_profile,
    truncation_study,
    wavefront_ridge,
)

OUTPUT_ROOT = Path("results/caustic_diagnostics")
# The colour range is a percentile of |Phi| over the panels being shown, not a
# fraction of the largest value.  The focused region occupies a large part of
# the crop and carries most of the amplitude, so a fraction of the maximum
# either saturates the whole wedge into one flat block or hides the wavefront
# entirely.  Clipping the brightest tenth keeps the ridge and the cusp legible
# while the scale stays symmetric, linear, and shared across the sequence.
DISPLAY_PERCENTILE = 90.0
# A single snapshot is far more skewed than a sequence: at the time the
# wavefront wraps the hole the median of |Phi| is some four hundred times below
# the maximum, so the same percentile that suits the shared sequence leaves the
# arms of the front invisible.  The height sheet therefore clips harder.
SURFACE_PERCENTILE = 70.0


def _diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "signed_field",
        ["#08306b", "#2b7bba", "#94c4df", "#f7f7f7", "#fdbe85", "#e6550d", "#7f2704"],
    )


def _mark_geometry(axis, *, source_radius: float = 6.0, legend: bool = False) -> None:
    axis.add_patch(
        plt.Circle((0.0, 0.0), HORIZON_RADIUS, color="black", zorder=6,
                   label="event horizon" if legend else None)
    )
    axis.add_patch(
        plt.Circle((0.0, 0.0), PHOTON_SPHERE_RADIUS, fill=False, color="0.25",
                   linestyle="--", linewidth=1.0, zorder=7,
                   label="photon sphere" if legend else None)
    )
    axis.plot([source_radius], [0.0], marker="*", color="black", markersize=11,
              markeredgecolor="white", markeredgewidth=0.6, zorder=8,
              label="emitter" if legend else None, linestyle="none")


def section_sequence(
    archive: str | Path,
    *,
    output_dir: Path = OUTPUT_ROOT,
    outer_radius: float = DEFAULT_OUTER_RADIUS,
    columns: int = 6,
    stem: str = "caustic_section_sequence",
) -> dict:
    """Draw the equatorial sections around the focus on one fixed scale."""

    result = load(archive)
    focus = locate_focus(result, outer_radius=outer_radius)
    times = np.asarray(result.snapshot_times, dtype=float)
    centre = focus["snapshot_index"]
    span = columns * 2
    first = max(0, centre - span // 2)
    chosen = list(range(first, min(times.size, first + span)))

    sections = [
        equatorial_section(result, index, outer_radius=outer_radius)
        for index in chosen
    ]
    scale = float(
        np.percentile(
            np.concatenate([np.abs(section.field).ravel() for section in sections]),
            DISPLAY_PERCENTILE,
        )
    )

    rows = int(np.ceil(len(sections) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.05 * columns, 3.2 * rows), squeeze=False
    )
    colormap = _diverging()
    mesh = None
    for axis, section in zip(axes.flat, sections):
        x, y = section.cartesian()
        for sign in (1.0, -1.0):
            mesh = axis.pcolormesh(
                x, sign * y, section.field, cmap=colormap, vmin=-scale, vmax=scale,
                shading="gouraud", rasterized=True,
            )
        ridge = wavefront_ridge(section)
        strong = ridge["value"] > 0.12 * float(ridge["value"].max())
        for sign in (1.0, -1.0):
            axis.plot(
                ridge["radius"][strong] * np.cos(ridge["angle"][strong]),
                sign * ridge["radius"][strong] * np.sin(ridge["angle"][strong]),
                color="0.15", linewidth=0.7, alpha=0.75, zorder=5,
            )
        record = axial_focus(section)
        axis.plot([-record["radius_over_M"]], [0.0], marker="o", markersize=6,
                  markerfacecolor="none", markeredgecolor="#00a0a0",
                  markeredgewidth=1.4, zorder=9, linestyle="none")
        _mark_geometry(axis)
        axis.set_aspect("equal")
        axis.set_xlim(-outer_radius, outer_radius)
        axis.set_ylim(-outer_radius, outer_radius)
        axis.set_xticks([-20, -10, 0, 10, 20])
        axis.set_yticks([-20, -10, 0, 10, 20])
        axis.tick_params(labelsize=7)
        axis.set_title(
            r"$\tau/M=%.2f$   $|\Phi|_{\rm axis}=%.2e$"
            % (section.bridge_time, record["peak_abs_field"]),
            fontsize=8,
        )
    for axis in axes.flat[len(sections):]:
        axis.axis("off")
    for axis in axes[-1]:
        axis.set_xlabel(r"$x/M$", fontsize=8)
    for row in axes:
        row[0].set_ylabel(r"$y/M$", fontsize=8)

    bar = figure.colorbar(mesh, ax=axes, fraction=0.016, pad=0.012)
    bar.set_label(
        r"$\Phi=u/r$   (shared symmetric linear scale, clipped at the "
        + f"{DISPLAY_PERCENTILE:g}th percentile)",
        fontsize=9,
    )
    figure.suptitle(
        "Equatorial section through the antipodal caustic, "
        r"emitter at $r=6M$, $L/M=80$",
        fontsize=12,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "display_scale": scale,
        "focus": focus,
        "snapshot_cadence_over_M": float(np.median(np.diff(times))),
        "shown_bridge_times": [section.bridge_time for section in sections],
    }


def focus_profile(
    archive: str | Path,
    *,
    output_dir: Path = OUTPUT_ROOT,
    outer_radius: float = DEFAULT_OUTER_RADIUS,
    stem: str = "caustic_focus_profile",
) -> dict:
    """Measure the focus: transverse width, radial cut, and what limits it."""

    result = load(archive)
    focus = locate_focus(result, outer_radius=outer_radius)
    index = focus["snapshot_index"]
    radius = focus["radius_over_M"]
    concentration = float(result.metadata["source"]["angular_concentration"])
    source_width = np.rad2deg(concentration**-0.5)
    available = int(np.asarray(result.response_ell, dtype=int).max())

    angle, profile = transverse_profile(result, index, radius)
    width = full_width_half_maximum(angle, profile)
    study = truncation_study(
        result, index, radius,
        truncations=tuple(v for v in (20, 30, 40, 50, 60, 80) if v <= available),
    )

    section = equatorial_section(result, index, outer_radius=outer_radius)
    axial = section.angle >= np.pi - np.deg2rad(1.0)
    radial_cut = np.abs(section.field[:, axial]).max(axis=1)

    times = np.asarray(result.snapshot_times, dtype=float)
    history = []
    for position in range(times.size):
        record = axial_focus(
            equatorial_section(result, position, outer_radius=outer_radius)
        )
        history.append(record)

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))

    left = np.rad2deg(np.pi - angle)
    axes[0, 0].plot(left, profile, color="#08306b", linewidth=1.6)
    axes[0, 0].axhline(0.0, color="0.6", linewidth=0.8)
    peak = float(np.abs(profile).max())
    axes[0, 0].axhline(0.5 * peak * np.sign(profile[np.argmax(np.abs(profile))]),
                       color="#e6550d", linestyle=":", linewidth=1.1,
                       label="half maximum")
    axes[0, 0].axvline(0.5 * width, color="#e6550d", linestyle="--", linewidth=1.0)
    axes[0, 0].set(
        xlabel=r"angle from the antipodal axis, $180^\circ-\gamma$ (degrees)",
        ylabel=r"$\Phi$",
        title=(r"Transverse cut at $r=%.2fM$: FWHM $=%.2f^\circ$, "
               r"emitter width $=%.2f^\circ$" % (radius, width, source_width)),
    )
    axes[0, 0].set_xlim(0.0, 40.0)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].semilogy(section.radius, radial_cut, color="#08306b", linewidth=1.5)
    axes[0, 1].axvline(radius, color="#00a0a0", linestyle="--", linewidth=1.2,
                       label="axial focus")
    axes[0, 1].axvline(PHOTON_SPHERE_RADIUS, color="0.35", linestyle=":",
                       linewidth=1.1, label="photon sphere")
    axes[0, 1].set(
        xlabel=r"areal radius $r/M$", ylabel=r"$|\Phi|$ on the antipodal axis",
        title=r"Radial cut along $\gamma=180^\circ$ at $\tau/M=%.2f$"
        % focus["bridge_time"],
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.25, which="both")

    orders = [row["ell_max"] for row in study]
    widths = [row["fwhm_degrees"] for row in study]
    changes = [max(row["relative_peak_change"], 1e-18) for row in study]
    axes[1, 0].plot(orders, widths, "o-", color="#08306b", label="measured FWHM")
    axes[1, 0].axhline(source_width, color="#e6550d", linestyle="--",
                       label=r"emitter angular width $\kappa^{-1/2}$")
    axes[1, 0].set(xlabel=r"angular truncation $\ell_{\max}$",
                   ylabel="focus FWHM (degrees)",
                   title="The focus does not narrow with a longer sum")
    axes[1, 0].set_ylim(0.0, max(widths) * 1.35)
    twin = axes[1, 0].twinx()
    twin.semilogy(orders, changes, "s--", color="#7f2704", markersize=5,
                  label="relative change in peak")
    twin.set_ylabel("relative change in peak amplitude", color="#7f2704")
    twin.tick_params(axis="y", colors="#7f2704")
    axes[1, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(times, [record["peak_abs_field"] for record in history],
                    "o-", color="#08306b", markersize=3.5, label=r"$|\Phi|$ on axis")
    axes[1, 1].axvline(focus["bridge_time"], color="#00a0a0", linestyle="--",
                       linewidth=1.2, label="selected snapshot")
    axes[1, 1].set(xlabel=r"bridge time $\tau/M$",
                   ylabel=r"peak $|\Phi|$ on the antipodal axis",
                   title="Focusing history, used to choose the snapshot")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25, which="both")

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    truncation_path = tables / f"{stem}_truncation.csv"
    with truncation_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(study[0]))
        writer.writeheader()
        writer.writerows(study)
    history_path = tables / f"{stem}_focus_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["bridge_time", "radius_over_M", "peak_abs_field",
                  "transverse_reference", "amplification"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    return {
        "png": str(png),
        "pdf": str(pdf),
        "focus": focus,
        "fwhm_degrees": width,
        "emitter_angular_width_degrees": source_width,
        "width_in_emitter_widths": width / source_width,
        "truncation_table": str(truncation_path),
        "focus_history_table": str(history_path),
        "truncation_study": study,
    }


def height_and_colour(
    archive: str | Path,
    *,
    output_dir: Path = OUTPUT_ROOT,
    outer_radius: float = DEFAULT_OUTER_RADIUS,
    snapshot_index: int | None = None,
    bridge_time: float | None = None,
    stem: str = "caustic_height_colour",
    elevation: float = 46.0,
    azimuth: float = -62.0,
) -> dict:
    """Draw the wavefront as a height and colour sheet over the equatorial plane."""

    result = load(archive)
    if snapshot_index is None and bridge_time is not None:
        snapshot_index = int(
            np.argmin(np.abs(np.asarray(result.snapshot_times) - bridge_time))
        )
    if snapshot_index is None:
        snapshot_index = locate_focus(result, outer_radius=outer_radius)[
            "snapshot_index"
        ]
    section = equatorial_section(
        result, snapshot_index, outer_radius=outer_radius, angles=721
    )
    radius = section.radius
    angle = np.concatenate((-section.angle[::-1][:-1], section.angle))
    field = np.concatenate((section.field[:, ::-1][:, :-1], section.field), axis=1)
    x = np.outer(radius, np.cos(angle))
    y = np.outer(radius, np.sin(angle))

    scale = float(np.percentile(np.abs(field), SURFACE_PERCENTILE))
    clipped = np.clip(field, -scale, scale)
    colormap = _diverging()
    colours = colormap((clipped + scale) / (2.0 * scale))

    figure = plt.figure(figsize=(11.0, 7.4))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x, y, clipped, facecolors=colours, rstride=1, cstride=1,
        linewidth=0.0, antialiased=True, shade=False,
    )
    # Without this the vector output carries a quad per grid cell and reaches
    # tens of megabytes; the sheet is an image, the axes stay vector.
    surface.set_rasterized(True)
    ring = np.linspace(-np.pi, np.pi, 361)
    for value, colour, width in (
        (HORIZON_RADIUS, "black", 2.4),
        (PHOTON_SPHERE_RADIUS, "0.35", 1.3),
    ):
        axis.plot(value * np.cos(ring), value * np.sin(ring),
                  np.full_like(ring, -scale), color=colour, linewidth=width,
                  zorder=10)
    axis.set_zlim(-scale, scale)
    axis.set_xlabel(r"$x/M$", labelpad=2)
    axis.set_ylabel(r"$y/M$", labelpad=2)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_box_aspect((1.0, 1.0, 0.34), zoom=1.35)
    axis.set_zticks([])
    axis.set_xticks([-20, -10, 0, 10, 20])
    axis.set_yticks([-20, -10, 0, 10, 20])
    axis.tick_params(labelsize=8, pad=0)
    # The panes and the vertical axis carry no information here and cost most
    # of the canvas, so the sheet itself gets the space instead.
    for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        pane._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.35)
    axis.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    axis.set_title(
        r"Wavefront as height and colour, $\tau/M=%.2f$, "
        r"clipped at $\pm%.1e$" % (section.bridge_time, scale),
        fontsize=12,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "bridge_time": section.bridge_time,
        "display_scale": scale,
    }


def build_all(archive: str | Path, *, output_dir: Path = OUTPUT_ROOT) -> dict:
    """Build every diagnostic and record what was measured."""

    summary = {
        "archive": str(archive),
        "section_sequence": section_sequence(archive, output_dir=output_dir),
        "focus_profile": focus_profile(archive, output_dir=output_dir),
        "height_and_colour": height_and_colour(archive, output_dir=output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    record = output_dir / "caustic_diagnostics.json"
    with record.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=float)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    summary = build_all(arguments.archive, output_dir=arguments.output_dir)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
