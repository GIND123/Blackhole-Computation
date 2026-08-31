"""Presentation alternatives for the antipodal focus of the narrow source.

The height-and-colour sheet in :mod:`black_hole.caustic_diagnostic_figures`
sets its display range from a percentile of the disturbed region.  At the
narrow-source focus that range is ``1.69e-3`` while the peak is ``2.08e-2``,
so the strongest feature in the section is saturated by a factor of twelve.
That is acceptable in a development diagnostic and is not acceptable in a
figure that is meant to show the focus.

Every figure here is built under two rules.

* **No clipping.**  The colour limits are the signed extremes of the data that
  is drawn, so the peak sample lands exactly on the end of the colour bar and
  never inside a saturated block.  Each builder asserts this and records the
  drawn extremes beside the measured peak.
* **A monotone transfer function.**  Weak structure is recovered with a signed
  ``asinh`` stretch rather than by clipping.  The stretch is linear for
  ``|Phi| << beta`` and logarithmic for ``|Phi| >> beta``, is invertible, and
  preserves sign, so relative amplitudes remain readable.  ``beta`` is the
  median ``|Phi|`` of the disturbed region and is recorded with the figure.

The cameras are chosen so that the focus is the nearest feature to the viewer
and is not occluded by the wavefront ridge behind it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import AsinhNorm, LinearSegmentedColormap  # noqa: E402

from .caustic_diagnostics import (  # noqa: E402
    HORIZON_RADIUS,
    WAVE_ZONE_RADIUS,
    PHOTON_SPHERE_RADIUS,
    axial_focus,
    equatorial_section,
    load,
    transverse_profile,
    wavefront_ridge,
)

OUTPUT_ROOT = Path("results/caustic_focus_figures")
NARROW_ARCHIVE = Path("results/caustic_diagnostics/raw/narrow_source_L80_fine.npz")
# The focus of the narrow source sits at this bridge time; the sequence and the
# time stack are centred on it.
FOCUS_BRIDGE_TIME = 48.0
# The figures crop to the region that carries the converging front.  The
# emitter sits at r = 6M and the focus at r = 6.25M on the opposite axis, so
# this keeps both with room to spare and drops the quiet outer field.
CROP_RADIUS = 12.0
# Samples below this fraction of the snapshot peak are undisturbed and are
# excluded when the linear width of the stretch is measured.
ACTIVE_FRACTION = 0.01


def _diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "signed_field",
        ["#08306b", "#2b7bba", "#94c4df", "#f7f7f7", "#fdbe85", "#e6550d", "#7f2704"],
    )


def _linear_width(field: np.ndarray) -> float:
    """Return the ``asinh`` linear width from the disturbed part of a field.

    Using the median of the disturbed region rather than a fraction of the
    peak keeps the stretch comparable between emitters: a narrower source
    leaves more of the crop quiet, which would drag a whole-section statistic
    down with it.
    """

    magnitude = np.abs(field)
    peak = float(magnitude.max())
    if peak <= 0.0:
        return 1.0
    active = magnitude[magnitude > ACTIVE_FRACTION * peak]
    if active.size == 0:
        return peak
    return float(np.median(active))


def _unclipped_norm(field: np.ndarray) -> tuple[AsinhNorm, dict]:
    """Return a signed ``asinh`` norm whose limits contain every sample."""

    peak = float(np.abs(field).max())
    beta = _linear_width(field)
    norm = AsinhNorm(linear_width=beta, vmin=-peak, vmax=peak)
    record = {
        "transfer_function": "signed asinh",
        "linear_width_beta": beta,
        "colour_limit": peak,
        "drawn_minimum": float(field.min()),
        "drawn_maximum": float(field.max()),
        "peak_is_clipped": False,
    }
    # The limits are the signed extremes of the same array, so this cannot
    # fail; it is kept so that a future change to the crop cannot silently
    # reintroduce saturation.
    assert record["colour_limit"] >= abs(record["drawn_minimum"]) - 1e-15
    assert record["colour_limit"] >= abs(record["drawn_maximum"]) - 1e-15
    return norm, record


def _colourbar(figure, mappable, axis, norm: AsinhNorm, *, label: str):
    """Attach a colour bar whose ticks carry physical field values."""

    peak = float(norm.vmax)
    beta = float(norm.linear_width)
    ticks = sorted(
        {
            -peak,
            -10.0 ** np.floor(np.log10(peak)),
            -beta,
            0.0,
            beta,
            10.0 ** np.floor(np.log10(peak)),
            peak,
        }
    )
    ticks = [value for value in ticks if -peak <= value <= peak]
    bar = figure.colorbar(mappable, ax=axis, ticks=ticks, pad=0.02, shrink=0.85)
    bar.ax.set_yticklabels([f"{value:+.2e}" for value in ticks], fontsize=7)
    bar.set_label(label, fontsize=8)
    return bar


def _half_plane(section) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the upper equatorial half-plane of one section.

    The source is axisymmetric about the emitter direction, so the equatorial
    field depends only on the angle from the emitter and the lower half-plane
    is an exact reflection of the upper one.  Drawing one half therefore loses
    nothing and leaves the other half free for a magnified inset.
    """

    x = np.outer(section.radius, np.cos(section.angle))
    y = np.outer(section.radius, np.sin(section.angle))
    return x, y, section.field


def _mirror(section) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the full equatorial disc from the half-plane section."""

    angle = np.concatenate([section.angle, 2.0 * np.pi - section.angle[-2::-1]])
    field = np.concatenate([section.field, section.field[:, -2::-1]], axis=1)
    x = np.outer(section.radius, np.cos(angle))
    y = np.outer(section.radius, np.sin(angle))
    return x, y, field


def _mark_geometry(
    axis, *, source_radius: float = 6.0, upper_half_only: bool = False
) -> None:
    axis.add_patch(
        plt.Circle((0.0, 0.0), HORIZON_RADIUS, color="#101010", zorder=5)
    )
    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            PHOTON_SPHERE_RADIUS,
            fill=False,
            color="#404040",
            linestyle="--",
            linewidth=0.8,
            zorder=6,
        )
    )
    axis.plot(
        [source_radius], [0.0], marker="*", color="#111111", markersize=7, zorder=7
    )


def _snapshot_indices(result, times) -> list[int]:
    archived = np.asarray(result.snapshot_times, dtype=float)
    return [int(np.argmin(np.abs(archived - value))) for value in times]


def _save(figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def equatorial_inset(
    archive: str | Path = NARROW_ARCHIVE,
    *,
    output_dir: Path = OUTPUT_ROOT,
    stem: str = "focus_equatorial_inset",
    times: tuple[float, ...] = (47.0, 47.5, 48.0, 48.5),
    angles: int = 721,
) -> dict:
    """Compact equatorial sequence with a magnified antipodal inset.

    The panels share one signed ``asinh`` scale taken from the whole sequence,
    so brightness differences between panels are amplitude differences and no
    panel clips its own peak.  The inset sits over the quiet part of the crop
    and magnifies the antipodal axis, where the converging front is otherwise
    only a few panel pixels across.
    """

    result = load(archive)
    indices = _snapshot_indices(result, times)
    sections = [
        equatorial_section(result, index, outer_radius=CROP_RADIUS, angles=angles)
        for index in indices
    ]
    stacked = np.concatenate([section.field.ravel() for section in sections])
    norm, display = _unclipped_norm(stacked)
    colours = _diverging()

    centre = int(np.argmin(np.abs(np.asarray(times) - FOCUS_BRIDGE_TIME)))
    focus = axial_focus(sections[centre])
    half_x, half_y = 4.5, 2.4
    magnification = CROP_RADIUS / half_x

    figure, axes = plt.subplots(
        1, len(sections), figsize=(3.3 * len(sections), 3.8), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for axis, section in zip(axes, sections):
        x, y, field = _mirror(section)
        mesh = axis.pcolormesh(
            x, y, field, cmap=colours, norm=norm, shading="auto", rasterized=True
        )
        _mark_geometry(axis)
        axis.set(
            aspect="equal",
            xlim=(-CROP_RADIUS, CROP_RADIUS),
            ylim=(-CROP_RADIUS, CROP_RADIUS),
            title=rf"$\tau/M={section.bridge_time:.2f}$",
        )
        axis.set_xlabel(r"$x/M$", fontsize=8)
        axis.tick_params(labelsize=7)

        inset = axis.inset_axes([0.03, 0.70, 0.94, 0.27])
        inset.pcolormesh(
            x, y, field, cmap=colours, norm=norm, shading="auto", rasterized=True
        )
        inset.set(
            aspect="equal",
            xlim=(-focus["radius_over_M"] - half_x,
                  -focus["radius_over_M"] + half_x),
            ylim=(-half_y, half_y),
            xticks=[],
            yticks=[],
        )
        for spine in inset.spines.values():
            spine.set(color="#1a1a1a", linewidth=0.9)
        inset.text(
            0.015, 0.08, rf"$\times{magnification:.1f}$ about the axis",
            transform=inset.transAxes, fontsize=7, color="#1a1a1a",
        )

    axes[0].set_ylabel(r"$y/M$", fontsize=8)
    _colourbar(figure, mesh, list(axes), norm, label=r"$\Phi$")
    figure.suptitle(
        "Narrow source through the antipodal focus on one shared unclipped "
        "signed scale; magnified antipodal axis above each panel",
        fontsize=9.5,
    )
    png, pdf = _save(figure, Path(output_dir) / stem)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "bridge_times": [section.bridge_time for section in sections],
        "crop_radius_over_M": CROP_RADIUS,
        "inset_half_width_over_M": half_x,
        "inset_magnification": magnification,
        "focus": focus,
        "display": display,
    }


# The strong-field region immediately outside the horizon is not part of the
# focus and its steep gradient dominates a surface plot, so surfaces start
# here.  This is the same cut the ridge tracer uses.
SURFACE_INNER_RADIUS = 2.2


def _surface_grid(
    section,
    *,
    radial_stride: int,
    angular_stride: int,
    inner_radius: float = SURFACE_INNER_RADIUS,
    angle_range: tuple[float, float] | None = None,
):
    """Return a decimated Cartesian grid of one section for surface plotting.

    ``angle_range`` keeps only the angles inside it, which is how the cutaway
    removes a wedge.  Angles are measured from the emitter direction, so the
    antipodal axis is at ``pi``.
    """

    keep = section.radius >= inner_radius
    radius = section.radius[keep][::radial_stride]
    angle = section.angle[::angular_stride]
    field = section.field[keep][::radial_stride, ::angular_stride]
    full_angle = np.concatenate([angle, 2.0 * np.pi - angle[-2::-1]])
    full_field = np.concatenate([field, field[:, -2::-1]], axis=1)
    if angle_range is not None:
        inside = (full_angle >= angle_range[0]) & (full_angle <= angle_range[1])
        full_angle = full_angle[inside]
        full_field = full_field[:, inside]
    x = np.outer(radius, np.cos(full_angle))
    y = np.outer(radius, np.sin(full_angle))
    return x, y, full_field


def _tidy_3d(axis) -> None:
    """Mute the panes so the surface, not the box, carries the figure."""

    for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane.pane.set_facecolor("#ffffff")
        pane.pane.set_alpha(1.0)
        pane.pane.set_edgecolor("#d9d9d9")
    axis.grid(True, alpha=0.25)


def birdseye_surface(
    archive: str | Path = NARROW_ARCHIVE,
    *,
    output_dir: Path = OUTPUT_ROOT,
    stem: str = "focus_birdseye_surface",
    time: float = FOCUS_BRIDGE_TIME,
    angles: int = 721,
    elevation: float = 58.0,
    azimuth: float = 200.0,
) -> dict:
    """Unclipped bird's-eye surface of the focus.

    Height and colour carry the same signed field, following the convention of
    the earlier Schwarzschild caustic study.  Unlike the development sheet, the
    vertical axis spans the full signed range of the drawn data, so the focus
    is the tallest feature in the frame rather than a saturated plateau.  The
    camera looks down the antipodal axis from above so the peak is nearest the
    viewer and nothing in front of it can occlude it.
    """

    result = load(archive)
    index = _snapshot_indices(result, [time])[0]
    section = equatorial_section(result, index, outer_radius=CROP_RADIUS,
                                 angles=angles)
    focus = axial_focus(section)
    x, y, field = _surface_grid(section, radial_stride=2, angular_stride=3)
    norm, display = _unclipped_norm(field)
    colours = _diverging()

    figure = plt.figure(figsize=(8.2, 6.2))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x, y, field,
        facecolors=colours(norm(field)),
        rstride=1, cstride=1, linewidth=0.0, antialiased=False, shade=False,
    )
    # A surface of this many quads is unreadable as vector art and produces a
    # multi-megabyte PDF, so it is rasterized inside an otherwise vector figure.
    surface.set_rasterized(True)
    peak = float(np.abs(field).max())
    axis.set(
        xlabel=r"$x/M$", ylabel=r"$y/M$", zlabel=r"$\Phi$",
        xlim=(-CROP_RADIUS, CROP_RADIUS),
        ylim=(-CROP_RADIUS, CROP_RADIUS),
        zlim=(-peak, peak),
    )
    axis.view_init(elev=elevation, azim=azimuth)
    axis.tick_params(labelsize=7)
    axis.zaxis.set_tick_params(pad=1.0)
    _tidy_3d(axis)
    axis.set_title(
        rf"Narrow-source focus at $\tau/M={section.bridge_time:.2f}$: "
        "height and colour are the same unclipped field",
        fontsize=10,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colours)
    mappable.set_array(field)
    bar = _colourbar(figure, mappable, axis, norm, label=r"$\Phi$")
    bar.ax.set_position(bar.ax.get_position().translated(0.07, 0.0))
    png, pdf = _save(figure, Path(output_dir) / stem)
    return {
        "png": str(png), "pdf": str(pdf),
        "bridge_time": section.bridge_time,
        "inner_radius_over_M": SURFACE_INNER_RADIUS,
        "focus_sign": "negative excursion",
        "camera": {"elevation_degrees": elevation, "azimuth_degrees": azimuth},
        "vertical_limit": peak,
        "focus": focus,
        "display": display,
    }


def _axial_profiles(result, indices, *, angles: int = 721):
    """Return the antipodal-axis profiles of a set of snapshots.

    The section angle runs from the emitter at ``0`` to the antipode at ``pi``,
    so the last column is the axis along which the front converges.
    """

    radius = None
    profiles = []
    times = []
    for index in indices:
        section = equatorial_section(result, index, outer_radius=CROP_RADIUS,
                                     angles=angles)
        radius = section.radius
        profiles.append(section.field[:, -1])
        times.append(section.bridge_time)
    return np.asarray(times), radius, np.asarray(profiles)


def axial_cutaway(
    archive: str | Path = NARROW_ARCHIVE,
    *,
    output_dir: Path = OUTPUT_ROOT,
    stem: str = "focus_axial_cutaway",
    time: float = FOCUS_BRIDGE_TIME,
    angles: int = 721,
    elevation: float = 20.0,
    azimuth: float = 250.0,
) -> dict:
    """Three-dimensional cutaway along the plane that contains the focus.

    The surface is cut on the plane ``y = 0``, which is the plane the emitter
    and the antipodal focus both lie in, and the exposed cross-section is drawn
    as a filled curtain.  One therefore reads the peak amplitude directly off
    the cut face instead of inferring it from a shaded surface, and the camera
    sits low and on the antipodal side so no part of the retained surface
    stands between the viewer and the focus.
    """

    result = load(archive)
    index = _snapshot_indices(result, [time])[0]
    section = equatorial_section(result, index, outer_radius=CROP_RADIUS,
                                 angles=angles)
    focus = axial_focus(section)
    keep = section.radius >= SURFACE_INNER_RADIUS
    radius = section.radius[keep]

    x, y, field = _surface_grid(
        section, radial_stride=2, angular_stride=3, angle_range=(0.0, np.pi)
    )
    norm, display = _unclipped_norm(section.field[keep])
    colours = _diverging()

    figure = plt.figure(figsize=(8.6, 5.8))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x, y, field,
        facecolors=colours(norm(field)),
        rstride=1, cstride=1, linewidth=0.0, antialiased=False, shade=False,
        alpha=0.92,
    )
    surface.set_rasterized(True)

    # The exposed face: the antipodal arm at gamma = pi and the emitter arm at
    # gamma = 0, both drawn in the cut plane as filled cross-sections.
    for column, sign, label in ((-1, -1.0, "antipodal arm"), (0, 1.0, "emitter arm")):
        profile = section.field[keep, column]
        axis.plot(sign * radius, np.zeros_like(radius), profile,
                  color="#252525", linewidth=1.1, zorder=10)
        for position, value in zip(sign * radius[::2], profile[::2]):
            axis.plot([position, position], [0.0, 0.0], [0.0, value],
                      color=colours(norm(value)), linewidth=0.7, alpha=0.85)
        del label

    peak = float(np.abs(section.field[keep]).max())
    axis.plot([-focus["radius_over_M"]], [0.0], [focus["peak_abs_field"] * -1.0],
              marker="o", markersize=5, color="#252525", zorder=12)
    axis.set(
        xlabel=r"$x/M$", ylabel=r"$y/M$", zlabel=r"$\Phi$",
        xlim=(-CROP_RADIUS, CROP_RADIUS),
        ylim=(-CROP_RADIUS, CROP_RADIUS),
        zlim=(-peak, peak),
    )
    axis.view_init(elev=elevation, azim=azimuth)
    axis.tick_params(labelsize=7)
    _tidy_3d(axis)
    axis.set_title(
        rf"Cutaway on the focus plane at $\tau/M={section.bridge_time:.2f}$: "
        "the cut face carries the unclipped cross-section",
        fontsize=10,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colours)
    mappable.set_array(section.field[keep])
    _colourbar(figure, mappable, axis, norm, label=r"$\Phi$")
    png, pdf = _save(figure, Path(output_dir) / stem)
    return {
        "png": str(png), "pdf": str(pdf),
        "bridge_time": section.bridge_time,
        "cut_plane": "y = 0, containing the emitter and the antipodal focus",
        "camera": {"elevation_degrees": elevation, "azimuth_degrees": azimuth},
        "focus": focus,
        "display": display,
    }


def _selected_focus(result, indices, radius, *, angles: int) -> tuple[int, int, dict]:
    """Return the focus that the established estimator selects in this range.

    A raw extremum is the wrong marker.  The strongest wave-zone excursion on
    the antipodal axis over ``46`` to ``50`` sits at the inner edge of the
    band, at ``r = 4.00M`` and ``tau = 48.75M``, and
    :func:`black_hole.caustic_diagnostics.axial_focus` rejects exactly that:
    a maximum on a band edge means the converging front has not arrived or has
    already passed.  These figures therefore mark the snapshot and radius that
    ``axial_focus`` accepts with the largest angular contrast, so the marker
    agrees with the measurement tables rather than competing with them.
    """

    best: tuple[int, dict] | None = None
    for position, index in enumerate(indices):
        section = equatorial_section(
            result, index, outer_radius=CROP_RADIUS, angles=angles
        )
        record = axial_focus(section)
        if not record["interior_maximum"]:
            continue
        if best is None or record["amplification"] > best[1]["amplification"]:
            best = (position, record)
    if best is None:
        raise ValueError(
            "No snapshot in the requested range has an interior axial maximum."
        )
    position, record = best
    column = int(np.argmin(np.abs(radius - record["radius_over_M"])))
    return position, column, record


def axial_time_stack(
    archive: str | Path = NARROW_ARCHIVE,
    *,
    output_dir: Path = OUTPUT_ROOT,
    stem: str = "focus_axial_time_stack",
    first_time: float = 46.0,
    last_time: float = 50.0,
    stride: int = 2,
    angles: int = 721,
    elevation: float = 22.0,
    azimuth: float = -72.0,
) -> dict:
    """Time stack of the antipodal-axis profile through the focus.

    Each curtain is the field along the antipodal axis at one bridge time,
    placed at its own time rather than at an arbitrary offset, so the depth
    axis is physical.  The stack shows the front converging to small radius,
    reaching its extremum, and re-expanding, which no single snapshot shows.
    Every curtain uses the one common unclipped scale.
    """

    result = load(archive)
    archived = np.asarray(result.snapshot_times, dtype=float)
    indices = [
        int(index)
        for index in np.flatnonzero(
            (archived >= first_time - 1e-9) & (archived <= last_time + 1e-9)
        )
    ]
    times, radius, profiles = _axial_profiles(result, indices, angles=angles)
    inside = radius >= SURFACE_INNER_RADIUS
    radius, profiles = radius[inside], profiles[:, inside]
    norm, display = _unclipped_norm(profiles)
    row, column, focus = _selected_focus(
        result, indices, radius, angles=angles
    )
    # Thin the stack so the curtains do not hide one another, but keep the
    # snapshot the estimator selected whatever the stride.
    keep = sorted({*range(0, len(indices), max(1, stride)), row})
    times, profiles = times[keep], profiles[keep]
    row = keep.index(row)

    figure = plt.figure(figsize=(8.8, 6.0))
    axis = figure.add_subplot(111, projection="3d")
    vertices = [
        np.column_stack(
            [
                np.concatenate([radius[:1], radius, radius[-1:]]),
                np.concatenate([[0.0], profile, [0.0]]),
            ]
        )
        for profile in profiles
    ]
    # The curtains are coded by their own bridge time rather than by the field,
    # because the field is already the height and a second encoding of it would
    # only make neighbouring curtains harder to tell apart.
    shades = plt.cm.viridis(np.linspace(0.85, 0.1, len(profiles)))
    curtains = PolyCollection(
        vertices,
        facecolors=shades,
        edgecolors="#303030",
        linewidths=0.6,
        alpha=0.82,
    )
    axis.add_collection3d(curtains, zs=times, zdir="y")
    axis.plot(
        [radius[column], radius[column]], [times[row], times[row]],
        [0.0, profiles[row, column]],
        color="#b2182b", linewidth=1.2, zorder=12,
    )
    axis.plot(
        [radius[column]], [times[row]], [profiles[row, column]],
        marker="o", markersize=6.0, color="#b2182b", zorder=13,
    )
    peak = float(np.abs(profiles).max())
    axis.set(
        xlabel=r"areal radius $r/M$", ylabel=r"bridge time $\tau/M$",
        zlabel=r"$\Phi$ on the antipodal axis",
        xlim=(SURFACE_INNER_RADIUS, CROP_RADIUS),
        ylim=(float(times[0]), float(times[-1])),
        zlim=(-peak, peak),
    )
    axis.view_init(elev=elevation, azim=azimuth)
    axis.tick_params(labelsize=7)
    _tidy_3d(axis)
    axis.set_title(
        "Antipodal-axis time stack through the focus; the marker is the "
        rf"selected focus at $\tau/M={times[row]:.2f}$, "
        rf"$r/M={radius[column]:.2f}$",
        fontsize=10,
    )
    png, pdf = _save(figure, Path(output_dir) / stem)
    return {
        "png": str(png), "pdf": str(pdf),
        "first_bridge_time": float(times[0]),
        "last_bridge_time": float(times[-1]),
        "curtains": int(times.size),
        "curtain_stride": stride,
        "focus": focus,
        "marked_bridge_time": float(times[row]),
        "marked_radius_over_M": float(radius[column]),
        "marked_field": float(profiles[row, column]),
        "camera": {"elevation_degrees": elevation, "azimuth_degrees": azimuth},
        "display": display,
    }


def axial_spacetime_map(
    archive: str | Path = NARROW_ARCHIVE,
    *,
    output_dir: Path = OUTPUT_ROOT,
    stem: str = "focus_axial_spacetime_map",
    first_time: float = 40.0,
    last_time: float = 54.0,
    angles: int = 721,
) -> dict:
    """The antipodal axis as a spacetime map.

    Nothing is projected away: the field on the antipodal axis is drawn against
    radius and bridge time on the same unclipped signed scale as the
    three-dimensional views.  The converging and re-expanding branches meet at
    the focus, and the marker is the measured wave-zone extremum rather than a
    feature read off the image.
    """

    result = load(archive)
    archived = np.asarray(result.snapshot_times, dtype=float)
    indices = [
        int(index)
        for index in np.flatnonzero(
            (archived >= first_time - 1e-9) & (archived <= last_time + 1e-9)
        )
    ]
    times, radius, profiles = _axial_profiles(result, indices, angles=angles)
    inside = radius >= SURFACE_INNER_RADIUS
    radius, profiles = radius[inside], profiles[:, inside]
    norm, display = _unclipped_norm(profiles)
    colours = _diverging()
    row, column, focus = _selected_focus(
        result, indices, radius, angles=angles
    )

    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    mesh = axis.pcolormesh(
        radius, times, profiles, cmap=colours, norm=norm, shading="auto",
        rasterized=True,
    )
    axis.axvline(PHOTON_SPHERE_RADIUS, color="#404040", linestyle="--",
                 linewidth=0.9, label="photon sphere")
    axis.axvline(WAVE_ZONE_RADIUS, color="#404040", linestyle=":",
                 linewidth=0.9, label="wave-zone cut")
    axis.plot(
        radius[column], times[row], marker="o", markersize=7,
        markerfacecolor="none", markeredgecolor="#111111", markeredgewidth=1.5,
        label="selected focus",
    )
    axis.set(
        xlabel=r"areal radius $r/M$", ylabel=r"bridge time $\tau/M$",
        xlim=(SURFACE_INNER_RADIUS, CROP_RADIUS),
        title="Field on the antipodal axis, unclipped signed scale",
    )
    axis.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axis.tick_params(labelsize=8)
    _colourbar(figure, mesh, axis, norm, label=r"$\Phi$")
    png, pdf = _save(figure, Path(output_dir) / stem)
    return {
        "png": str(png), "pdf": str(pdf),
        "first_bridge_time": float(times[0]),
        "last_bridge_time": float(times[-1]),
        "snapshots": int(times.size),
        "focus": focus,
        "marked_bridge_time": float(times[row]),
        "marked_radius_over_M": float(radius[column]),
        "marked_field": float(profiles[row, column]),
        "display": display,
    }


BUILDERS = {
    "equatorial_inset": equatorial_inset,
    "birdseye_surface": birdseye_surface,
    "axial_cutaway": axial_cutaway,
    "axial_time_stack": axial_time_stack,
    "axial_spacetime_map": axial_spacetime_map,
}


def build_all(
    archive: str | Path = NARROW_ARCHIVE,
    *,
    output_dir: Path = OUTPUT_ROOT,
    names: tuple[str, ...] | None = None,
) -> dict:
    """Build the alternatives and record what each one drew.

    The record carries the display parameters of every figure, so the claim
    that none of them clips its peak is checkable after the fact rather than
    only at the moment of drawing.
    """

    chosen = tuple(BUILDERS) if names is None else tuple(names)
    summary = {
        "archive": str(archive),
        "focus_bridge_time": FOCUS_BRIDGE_TIME,
        "crop_radius_over_M": CROP_RADIUS,
        "figures": {
            name: BUILDERS[name](archive, output_dir=Path(output_dir))
            for name in chosen
        },
    }
    summary["every_figure_is_unclipped"] = all(
        not record["display"]["peak_is_clipped"]
        for record in summary["figures"].values()
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    record = Path(output_dir) / "caustic_focus_figures.json"
    with record.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=float)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=NARROW_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--figure",
        action="append",
        choices=sorted(BUILDERS),
        help="build only this figure; may be repeated",
    )
    arguments = parser.parse_args()
    summary = build_all(
        arguments.archive,
        output_dir=arguments.output_dir,
        names=None if arguments.figure is None else tuple(arguments.figure),
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
