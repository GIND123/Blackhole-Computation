"""Data driven three dimensional views of the localized scalar response.

The plotting commands consume archived modal responses.  A separate runner
creates dense radial snapshots at measured caustic peaks for the cutaway view.
No interpolation in time or fitted time translation is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy import special

from .caustic_analysis import analytic_signal_estimate
from .localized_source import LocalizedSourceParameters, angular_spectral_weights
from .source_evolution import (
    SourcedNumericalParameters,
    SourcedSimulationResult,
    load_sourced_result,
)


ARCHIVE_ROOT = Path("results/regulator_production_v3/raw/source/fine")
OUTPUT_ROOT = Path("results/caustic_visualizations")
LENGTHS = (80.0, 160.0, 320.0, 640.0)
SOURCE = LocalizedSourceParameters(
    amplitude=1.0,
    center_radius=6.0,
    radial_half_width=0.75,
    time_center=30.0,
    time_half_width=2.0,
    angular_concentration=64.0,
)
PULSE_WINDOWS = ((18.0, 35.0, 0.0), (35.0, 53.0, np.pi))


def _case_name(length: float | None) -> str:
    return "schwarzschild" if length is None else f"sds_L{length:g}"


def _archive(root: Path, length: float | None) -> Path:
    return Path(root) / f"{_case_name(length)}.npz"


def angular_field(
    result: SourcedSimulationResult,
    response: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Reconstruct the reduced field with the spherical addition theorem."""

    theta, phi = np.broadcast_arrays(theta, phi)
    cosine_gamma = np.sin(theta) * np.cos(phi)
    weights = angular_spectral_weights(
        float(result.metadata["source"]["angular_concentration"]),
        int(result.response_ell[-1]),
    )
    field = np.zeros_like(cosine_gamma, dtype=float)
    lookup = {int(ell): index for index, ell in enumerate(result.response_ell)}
    for ell in result.response_ell:
        ell = int(ell)
        field += (
            response[lookup[ell]]
            * weights[ell]
            * (2.0 * ell + 1.0)
            / (4.0 * np.pi)
            * special.eval_legendre(ell, cosine_gamma)
        )
    return field


def field_on_sphere(
    result: SourcedSimulationResult,
    retarded_time: float,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    observer: int | None = None,
    interpolate_time: bool = False,
) -> tuple[float, np.ndarray]:
    """Return an archived or linearly interpolated field on one sphere."""

    observer = result.outer_index() if observer is None else observer
    if interpolate_time:
        if not result.retarded_time[0] <= retarded_time <= result.retarded_time[-1]:
            raise ValueError("Requested retarded time lies outside the archive.")
        right = int(np.searchsorted(result.retarded_time, retarded_time))
        if right == 0 or result.retarded_time[right] == retarded_time:
            index = right
            response = np.asarray(
                result.response_signals[index, observer], dtype=float
            )
        else:
            left = right - 1
            fraction = float(
                (retarded_time - result.retarded_time[left])
                / (result.retarded_time[right] - result.retarded_time[left])
            )
            response = (
                (1.0 - fraction) * result.response_signals[left, observer]
                + fraction * result.response_signals[right, observer]
            )
        actual_time = float(retarded_time)
    else:
        index = int(np.argmin(np.abs(result.retarded_time - retarded_time)))
        response = np.asarray(result.response_signals[index, observer], dtype=float)
        actual_time = float(result.retarded_time[index])
    return actual_time, angular_field(result, response, theta, phi)


def modal_response_at_time(
    result: SourcedSimulationResult, retarded_time: float, observer: int | None = None
) -> np.ndarray:
    """Interpolate the distinct ell responses at one geometric time."""

    observer = result.outer_index() if observer is None else observer
    if not result.retarded_time[0] <= retarded_time <= result.retarded_time[-1]:
        raise ValueError("Requested retarded time lies outside the archive.")
    right = int(np.searchsorted(result.retarded_time, retarded_time))
    if right == 0 or result.retarded_time[right] == retarded_time:
        return np.asarray(result.response_signals[right, observer], dtype=float)
    left = right - 1
    fraction = float(
        (retarded_time - result.retarded_time[left])
        / (result.retarded_time[right] - result.retarded_time[left])
    )
    return np.asarray(
        (1.0 - fraction) * result.response_signals[left, observer]
        + fraction * result.response_signals[right, observer],
        dtype=float,
    )


def validate_modal_candidate(
    candidate_archive: Path,
    reference_archive: Path,
    output_dir: Path,
    retarded_time: float = 44.0,
) -> Path:
    """Compare two angular reconstructions with an exact Parseval norm."""

    candidate = load_sourced_result(candidate_archive)
    reference = load_sourced_result(reference_archive)
    for name in ("response_ell", "mode_ell", "mode_m"):
        if not np.array_equal(getattr(candidate, name), getattr(reference, name)):
            raise ValueError(f"Candidate and reference have different {name} arrays.")
    if not np.array_equal(
        candidate.mode_source_amplitude, reference.mode_source_amplitude
    ):
        raise ValueError("Candidate and reference source projections differ.")
    candidate_response = modal_response_at_time(candidate, retarded_time)
    reference_response = modal_response_at_time(reference, retarded_time)
    response_lookup = {
        int(ell): index for index, ell in enumerate(candidate.response_ell)
    }
    indices = np.asarray(
        [response_lookup[int(ell)] for ell in candidate.mode_ell], dtype=int
    )
    candidate_modes = (
        candidate_response[indices] * candidate.mode_source_amplitude
    )
    reference_modes = (
        reference_response[indices] * reference.mode_source_amplitude
    )
    difference = candidate_modes - reference_modes
    report = {
        "candidate_archive": str(candidate_archive),
        "reference_archive": str(reference_archive),
        "retarded_time_U_over_M": retarded_time,
        "angular_ell_max": int(candidate.response_ell[-1]),
        "reconstructed_real_modes": int(candidate.mode_ell.size),
        "sphere_relative_l2": float(
            np.linalg.norm(difference) / np.linalg.norm(reference_modes)
        ),
        "maximum_modal_difference_relative_to_reference_maximum": float(
            np.max(np.abs(difference)) / np.max(np.abs(reference_modes))
        ),
        "norm_evaluation": "exact Parseval sum over stored real harmonic modes",
        "time_translation_fitted": False,
    }
    destination = Path(output_dir) / "dedalus_candidate_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return destination


def measured_pulse_times(result: SourcedSimulationResult) -> tuple[float, ...]:
    """Measure the direct pulse and first antipodal caustic echo."""

    times = result.retarded_time
    observer = result.outer_index()
    weights = angular_spectral_weights(
        float(result.metadata["source"]["angular_concentration"]),
        int(result.response_ell[-1]),
    )
    measured: list[float] = []
    for start, end, gamma in PULSE_WINDOWS:
        # P_l(cos gamma), not cos(gamma)**l.  The two agree only on the axis,
        # where P_l(1) = 1 and P_l(-1) = (-1)**l, so the difference is silent
        # for the two windows used here and wrong for any other direction.
        angular = np.asarray(
            [
                special.eval_legendre(int(ell), float(np.cos(gamma)))
                for ell in result.response_ell
            ]
        )
        coefficients = (
            (2.0 * result.response_ell + 1.0)
            * weights[result.response_ell]
            * angular
            / (4.0 * np.pi)
        )
        waveform = result.response_signals[:, observer, :] @ coefficients
        estimate = analytic_signal_estimate(times, waveform, (start, end))
        measured.append(float(estimate["time"]))
    return tuple(measured)


def requested_snapshot_times(
    result: SourcedSimulationResult, timestep: float = 0.0005
) -> tuple[float, ...]:
    """Convert measured retarded peak times into exact bridge-time steps."""

    offset = float(result.metadata["retarded_time_offset"]["q"])
    bridge_times = np.asarray(measured_pulse_times(result)) + offset
    return tuple(float(round(value / timestep) * timestep) for value in bridge_times)


def sequence_snapshot_times(
    result: SourcedSimulationResult,
    first_retarded: float,
    last_retarded: float,
    count: int,
    timestep: float,
) -> tuple[float, ...]:
    """Return evenly spaced retarded times as exact bridge integration steps.

    The measured direct and antipodal peak times are always included, so a
    sequence never displaces the two times the timing audit actually measured.
    """

    if count < 2:
        raise ValueError("A snapshot sequence needs at least two times.")
    offset = float(result.metadata["retarded_time_offset"]["q"])
    requested = np.linspace(float(first_retarded), float(last_retarded), int(count))
    bridge = np.concatenate(
        (requested + offset, np.asarray(requested_snapshot_times(result, timestep)))
    )
    stepped = np.round(bridge / timestep) * timestep
    return tuple(float(value) for value in np.unique(stepped))


def run_snapshot_case(
    length: float | None,
    archive_root: Path = ARCHIVE_ROOT,
    output_dir: Path = OUTPUT_ROOT,
    backend: str = "finite-difference",
    snapshot_times: tuple[float, ...] | None = None,
    name_suffix: str = "",
) -> Path:
    """Rerun one final-resolution case with dense caustic-time snapshots.

    ``snapshot_times`` overrides the default pair of measured peak times with an
    explicit sequence of bridge times, used to build the time-resolved views.
    """

    reference = load_sourced_result(_archive(archive_root, length))
    if backend not in {"finite-difference", "dedalus"}:
        raise ValueError("backend must be 'finite-difference' or 'dedalus'.")
    timestep = 0.0005 if backend == "finite-difference" else 0.002
    if snapshot_times is None:
        snapshots = requested_snapshot_times(reference, timestep=timestep)
    else:
        snapshots = tuple(sorted(float(value) for value in snapshot_times))
        if min(snapshots) <= 0.0:
            raise ValueError("Snapshot times must be positive bridge times.")
    suffix = ("" if backend == "finite-difference" else "_dedalus") + name_suffix
    destination = Path(output_dir) / "raw" / f"{_case_name(length)}{suffix}.npz"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_suffix(".running")
    with reservation.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_case_name(length) + "\n")
    numerical = SourcedNumericalParameters(
        radial_resolution=2048 if backend == "finite-difference" else 512,
        angular_ell_max=50,
        timestep=timestep,
        end_time=max(snapshots) + 0.25,
        signal_dt=0.01,
        diagnostic_dt=1.0,
        snapshot_dt=max(snapshots) + 1.0,
        snapshot_end_time=0.0,
        snapshot_radial_points=1024 if backend == "finite-difference" else 512,
        requested_snapshot_times=snapshots,
        observer_radii=(8.0, 12.0, None),
        compact_modal_storage=True,
    )
    if backend == "dedalus":
        from .dedalus_source_evolution import run_sourced_dedalus_simulation

        result = run_sourced_dedalus_simulation(
            background="schwarzschild" if length is None else "sds",
            source=SOURCE,
            numerical=numerical,
            cosmological_length=80.0 if length is None else length,
            timestepper="RK443",
            dealias=1.5,
        )
    else:
        from .source_evolution import run_sourced_simulation

        result = run_sourced_simulation(
            background="schwarzschild" if length is None else "sds",
            source=SOURCE,
            numerical=numerical,
            cosmological_length=80.0 if length is None else length,
        )
    result.metadata["visualization_sampling"] = {
        "selection": "analytic envelope peaks measured from final ell_max=50 archive",
        "measured_retarded_peak_times": list(measured_pulse_times(reference)),
        "requested_bridge_snapshot_times": list(snapshots),
        "dense_radial_points": numerical.snapshot_radial_points,
        "backend": backend,
        "time_translation_fitted": False,
    }
    temporary = destination.with_suffix(".incomplete.npz")
    result.save(temporary)
    temporary.rename(destination)
    reservation.unlink()
    return destination


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_sphere_time(
    archive: Path,
    output_dir: Path = OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Plot successive extraction spheres at the measured pulse peaks."""

    _style()
    result = load_sourced_result(archive)
    peak_times = measured_pulse_times(result)
    theta = np.linspace(0.0, np.pi, 55)
    phi = np.linspace(-np.pi, np.pi, 109)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    evaluations = [field_on_sphere(result, time, tt, pp) for time in peak_times]
    actual_times = [evaluation[0] for evaluation in evaluations]
    fields = [evaluation[1] for evaluation in evaluations]
    scale = max(float(np.max(np.abs(field))) for field in fields)
    normalization = colors.SymLogNorm(
        linthresh=1e-3 * scale, vmin=-scale, vmax=scale, base=10
    )
    cmap = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(8.0, 6.8))
    axis = fig.add_subplot(111, projection="3d")
    for shell, (time, field) in enumerate(zip(actual_times, fields)):
        radius = 1.0
        center = 2.55 * (shell - 0.5)
        x = center + radius * np.sin(tt) * np.cos(pp)
        y = radius * np.sin(tt) * np.sin(pp)
        z = radius * np.cos(tt)
        axis.plot_surface(
            x,
            y,
            z,
            facecolors=cmap(normalization(field)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=1.0,
        )
        axis.text(center, 0.0, radius + 0.18, rf"$U={time:.2f}M$", ha="center")
    axis.quiver(-1.2, 0.0, -1.35, 4.9, 0.0, 0.0, color="0.25", arrow_length_ratio=0.04)
    axis.text(1.25, 0.0, -1.55, "retarded time", ha="center", color="0.25")
    axis.set_box_aspect((2.4, 1, 1))
    axis.set_axis_off()
    axis.view_init(elev=20, azim=-55)
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap=cmap)
    bar = fig.colorbar(scalar, ax=axis, shrink=0.64, pad=0.02)
    bar.set_label(r"reduced field $u(U,\theta,\varphi)$")
    fig.suptitle("Angular caustic sequence on successive time spheres", y=0.90)
    paths = _save(fig, Path(output_dir) / "sphere_time_caustic")
    caption = (
        "Successive time spheres carry the numerically reconstructed reduced "
        "field at the direct and first antipodal envelope peaks. Display radius "
        "is fixed, while horizontal placement orders retarded time. The common "
        "signed color scale shows the angular field formed by the final ell_max=50 "
        "modal response; no time alignment is fitted."
    )
    (Path(output_dir) / "sphere_time_caustic_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    metadata = {
        "archive": str(archive),
        "measured_retarded_peak_times": list(peak_times),
        "actual_archived_retarded_times": actual_times,
        "angular_ell_max": int(result.response_ell[-1]),
        "common_color_limit": scale,
        "time_translation_fitted": False,
    }
    with (Path(output_dir) / "sphere_time_caustic.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)
    return paths


def plot_regulator_comparison(
    archive_root: Path = ARCHIVE_ROOT,
    output_dir: Path = OUTPUT_ROOT,
    retarded_time: float = 44.0,
) -> tuple[Path, Path]:
    """Compare outer-boundary angular fields at one common retarded time."""

    _style()
    cases = LENGTHS
    theta = np.linspace(0.0, np.pi, 121)
    phi = np.linspace(-np.pi, np.pi, 241)
    pp, latitude = np.meshgrid(phi, np.pi / 2.0 - theta)
    tt = np.pi / 2.0 - latitude
    reference = load_sourced_result(_archive(archive_root, None))
    reference_time, reference_field = field_on_sphere(
        reference, retarded_time, tt, pp, interpolate_time=True
    )
    values: list[
        tuple[SourcedSimulationResult, float, np.ndarray, np.ndarray, float]
    ] = []
    for length in cases:
        result = load_sourced_result(_archive(archive_root, length))
        actual, field = field_on_sphere(
            result, retarded_time, tt, pp, interpolate_time=True
        )
        residual = field - reference_field
        relative_l2 = float(np.linalg.norm(residual) / np.linalg.norm(reference_field))
        values.append((result, actual, field, residual, relative_l2))
    field_scale = max(
        float(np.max(np.abs(reference_field))),
        *(float(np.max(np.abs(field))) for _, _, field, _, _ in values),
    )
    residual_scale = max(
        float(np.max(np.abs(residual))) for _, _, _, residual, _ in values
    )
    field_norm = colors.TwoSlopeNorm(
        vmin=-field_scale, vcenter=0.0, vmax=field_scale
    )
    residual_norm = colors.TwoSlopeNorm(
        vmin=-residual_scale, vcenter=0.0, vmax=residual_scale
    )
    fig = plt.figure(figsize=(13.2, 6.1))
    grid = fig.add_gridspec(2, 5, left=0.025, right=0.985, top=0.86, bottom=0.19)
    top_axes = [fig.add_subplot(grid[0, column], projection="mollweide") for column in range(5)]
    bottom_axes = [fig.add_subplot(grid[1, column], projection="mollweide") for column in range(5)]
    for axis, length, (result, actual, field, _, _) in zip(
        top_axes[:4], cases, values
    ):
        field_mesh = axis.pcolormesh(
            pp,
            latitude,
            field,
            shading="auto",
            cmap="RdBu_r",
            norm=field_norm,
            rasterized=True,
        )
        horizon = float(result.metadata["horizons"]["cosmological"])
        axis.set_title(rf"$L/M={length:g}$, $r_c/M={horizon:.1f}$", fontsize=9, pad=6)
    field_mesh = top_axes[-1].pcolormesh(
        pp,
        latitude,
        reference_field,
        shading="auto",
        cmap="RdBu_r",
        norm=field_norm,
        rasterized=True,
    )
    top_axes[-1].set_title(r"Schwarzschild, $\mathcal{I}^+$", fontsize=9, pad=6)

    for axis, length, (_, _, _, residual, relative_l2) in zip(
        bottom_axes[:4], cases, values
    ):
        residual_mesh = axis.pcolormesh(
            pp,
            latitude,
            residual,
            shading="auto",
            cmap="RdBu_r",
            norm=residual_norm,
            rasterized=True,
        )
        axis.set_title(
            rf"$\|\delta u_L\|_2/\|u_0\|_2={relative_l2:.3f}$",
            fontsize=9,
            pad=6,
        )
    bottom_axes[-1].set_axis_off()
    bottom_axes[-1].text(
        0.5,
        0.58,
        r"$L\longrightarrow\infty$" "\n" r"$r_c\longrightarrow\mathcal{I}^+$" "\n" r"$\delta u_L\longrightarrow0$",
        transform=bottom_axes[-1].transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.55,
    )
    for axis in top_axes + bottom_axes[:4]:
        axis.grid(alpha=0.18)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    field_bar = fig.colorbar(
        field_mesh,
        ax=top_axes,
        location="right",
        shrink=0.78,
        pad=0.01,
    )
    field_bar.set_label(r"field $u$")
    residual_bar = fig.colorbar(
        residual_mesh,
        ax=bottom_axes[:4],
        location="bottom",
        orientation="horizontal",
        shrink=0.65,
        pad=0.11,
    )
    residual_bar.set_label(r"residual $\delta u_L=u_L-u_0$")
    fig.suptitle(
        rf"Cosmological flat limit at common retarded time $U={reference_time:.3f}M$"
    )
    paths = _save(fig, Path(output_dir) / "regulator_angular_comparison")
    caption = (
        "Outer-boundary angular field at one common geometric retarded time. "
        "The top row compares four finite-L cosmological-horizon fields with "
        "the independently evolved Schwarzschild field at future null "
        "infinity. The bottom row shows their signed residuals. Cosmological "
        "horizon radii and sphere L2 residuals are printed above the panels. "
        "Every case uses the final ell_max=50 localized-source archive. The "
        "field row and residual row each use one common signed color scale. "
        "No relative time translation is fitted."
    )
    (Path(output_dir) / "regulator_angular_comparison_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    metadata = {
        "requested_retarded_time": retarded_time,
        "actual_retarded_times": {
            _case_name(length): actual
            for length, (_, actual, _, _, _) in zip(cases, values)
        },
        "schwarzschild_retarded_time": reference_time,
        "angular_ell_max": 50,
        "common_field_color_limit": field_scale,
        "common_residual_color_limit": residual_scale,
        "relative_sphere_l2_residuals": {
            _case_name(length): relative_l2
            for length, (_, _, _, _, relative_l2) in zip(cases, values)
        },
        "time_translation_fitted": False,
    }
    with (Path(output_dir) / "regulator_angular_comparison.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)
    return paths


def plot_cutaway(
    archive: Path,
    output_dir: Path = OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Create a meridional cutaway from a dense peak-time snapshot archive."""

    _style()
    result = load_sourced_result(archive)
    requested = result.metadata.get("visualization_sampling", {}).get(
        "requested_bridge_snapshot_times", []
    )
    if len(requested) < 2 or result.snapshot_times.size < 4:
        raise ValueError("The cutaway requires a targeted dense snapshot archive.")
    target = float(requested[1])
    snapshot = int(np.argmin(np.abs(result.snapshot_times - target)))
    response = result.response_snapshots[snapshot]
    radius = result.snapshot_areal_radius
    finite = np.isfinite(radius)
    radius = radius[finite]
    response = response[:, finite]
    # A logarithmic display radius keeps the strong field and cosmological
    # regions visible together. The caption states this transformation.
    display_radius = np.log(radius / radius[0])
    angle = np.linspace(-np.pi, np.pi, 241)
    rr, aa = np.meshgrid(display_radius, angle, indexing="ij")
    theta = np.full_like(aa, np.pi / 2.0)
    field = np.empty_like(rr)
    for radial_index in range(radius.size):
        field[radial_index] = angular_field(
            result, response[:, radial_index], theta[radial_index], aa[radial_index]
        ) / radius[radial_index]
    x = rr * np.cos(aa)
    y = rr * np.sin(aa)
    limit = float(np.nanpercentile(np.abs(field), 99.5))
    norm = colors.SymLogNorm(
        linthresh=max(limit * 2.0e-2, np.finfo(float).tiny),
        vmin=-limit,
        vmax=limit,
    )
    cmap = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(8.2, 7.0))
    axis = fig.add_subplot(111, projection="3d")

    # The colored meridional plane is a direct reconstruction. Thin the
    # plotting mesh only after evaluating the field at all stored radii.
    radial_stride = max(1, radius.size // 280)
    angular_stride = 2
    sl = (slice(None, None, radial_stride), slice(None, None, angular_stride))
    axis.plot_surface(
        x[sl],
        y[sl],
        np.zeros_like(x[sl]),
        facecolors=cmap(norm(field[sl])),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=0.72,
    )

    # Axisymmetry about the source axis makes each signed meridional contour
    # an exact three dimensional isosurface after revolution. Plot only the
    # longest components to suppress tiny contour fragments at the floor.
    contour_figure, contour_axis = plt.subplots()
    negative_extent = float(abs(np.nanmin(field)))
    positive_extent = float(max(np.nanmax(field), 0.0))
    if positive_extent >= 0.2 * negative_extent:
        signed_levels = (-0.5 * limit, 0.5 * limit)
        surface_colors = ("#053061", "#b2182b")
        surface_description = "signed levels minus and plus one half of the display scale"
    else:
        signed_levels = (-0.65 * negative_extent, -0.30 * negative_extent)
        surface_colors = ("#053061", "#4393c3")
        surface_description = (
            "two negative levels, 0.65 and 0.30 times the magnitude of the field minimum"
        )
    contours = contour_axis.contour(x, y, field, levels=signed_levels)
    contour_segments = contours.allsegs
    plt.close(contour_figure)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 97)
    for level_segments, surface_color in zip(contour_segments, surface_colors):
        for segment in sorted(level_segments, key=len, reverse=True)[:4]:
            if len(segment) < 12:
                continue
            axial = segment[:, 0, None]
            transverse = np.abs(segment[:, 1, None])
            axis.plot_surface(
                np.broadcast_to(axial, (len(segment), azimuth.size)),
                transverse * np.cos(azimuth),
                transverse * np.sin(azimuth),
                color=surface_color,
                edgecolor=surface_color,
                linewidth=0.12,
                antialiased=True,
                shade=True,
                alpha=0.52,
            )

    outer = float(display_radius[-1])
    sphere_angle = np.linspace(0.0, 2.0 * np.pi, 49)
    sphere_polar = np.linspace(0.0, np.pi, 25)
    hole_radius = 0.045 * outer
    hs, hp = np.meshgrid(sphere_angle, sphere_polar)
    axis.plot_surface(
        hole_radius * np.cos(hp),
        hole_radius * np.sin(hp) * np.cos(hs),
        hole_radius * np.sin(hp) * np.sin(hs),
        color="black",
        linewidth=0.0,
        shade=True,
    )
    axis.plot(
        [0.0, outer], [0.0, 0.0], [0.0, 0.0], color="0.15", linewidth=0.8
    )
    annotation_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75}
    axis.text(
        0.72 * outer,
        0.0,
        0.10 * outer,
        "source direction",
        ha="center",
        color="black",
        bbox=annotation_box,
    )
    axis.text(
        -0.72 * outer,
        0.0,
        0.10 * outer,
        "antipode",
        ha="center",
        color="black",
        bbox=annotation_box,
    )
    axis.set_xlim(-outer, outer)
    axis.set_ylim(-outer, outer)
    axis.set_zlim(-0.72 * outer, 0.72 * outer)
    axis.set_box_aspect((2.0, 2.0, 1.44), zoom=1.24)
    axis.view_init(elev=25.0, azim=-58.0)
    axis.set_axis_off()
    axis.set_position([0.01, 0.02, 0.80, 0.92])
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    bar = fig.colorbar(scalar_map, ax=axis, shrink=0.68, pad=0.02)
    bar.set_label(r"physical scalar field $\Phi=u/r$")
    retarded = result.snapshot_times[snapshot] - float(
        result.metadata["retarded_time_offset"]["q"]
    )
    axis.set_title(
        rf"Antipodal caustic at $U={retarded:.3f}M$",
        fontsize=14,
        pad=4,
    )
    paths = _save(fig, Path(output_dir) / "caustic_cutaway")
    sampling = result.metadata.get("visualization_sampling", {})
    backend = str(sampling.get("backend", "finite-difference"))
    method = (
        "Dedalus 3 ChebyshevT evolution"
        if backend == "dedalus"
        else "eighth order finite difference evolution"
    )
    caption = (
        "Cutaway through the numerically reconstructed physical scalar field "
        "at the measured first antipodal caustic peak. The field has one sign "
        "at this measured phase. Transparent blue surfaces mark "
        + surface_description
        + ". The colored plane is a meridional slice. Axisymmetry "
        "about the source direction is used only to revolve the measured "
        "meridional contours into three dimensional isosurfaces. The display "
        "radius is log(r/r_b), which "
        "keeps the near-hole and outer regions visible in one panel. The data "
        f"use ell_max={int(result.response_ell[-1])}, {radius.size} stored radial "
        f"points, and a {method} at the exact requested timestep."
    )
    (Path(output_dir) / "caustic_cutaway_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    metadata = {
        "archive": str(archive),
        "requested_bridge_time": target,
        "actual_bridge_time": float(result.snapshot_times[snapshot]),
        "actual_retarded_time": retarded,
        "angular_ell_max": int(result.response_ell[-1]),
        "stored_radial_points": int(radius.size),
        "backend": backend,
        "radial_discretization": result.metadata.get("radial_discretization"),
        "reproducibility": result.metadata.get("reproducibility"),
        "display_radius": "log(r/r_b)",
        "signed_isosurface_levels": list(signed_levels),
        "field_minimum": float(np.nanmin(field)),
        "field_maximum": float(np.nanmax(field)),
        "common_color_limit": limit,
        "time_translation_fitted": False,
    }
    with (Path(output_dir) / "caustic_cutaway.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-snapshots")
    run.add_argument(
        "cases",
        nargs="+",
        choices=("schwarzschild", "80", "160", "320", "640"),
    )
    run.add_argument(
        "--backend",
        choices=("finite-difference", "dedalus"),
        default="finite-difference",
    )
    run.add_argument(
        "--sequence",
        nargs=3,
        type=float,
        metavar=("FIRST_U", "LAST_U", "COUNT"),
        help="store an evenly spaced retarded-time sequence in addition to the "
        "two measured peak times",
    )
    run.add_argument("--name-suffix", default="")
    subparsers.add_parser("sphere-time")
    comparison = subparsers.add_parser("regulator")
    comparison.add_argument("--time", type=float, default=44.0)
    cutaway = subparsers.add_parser("cutaway")
    cutaway.add_argument("archive", type=Path)
    validation = subparsers.add_parser("validate-dedalus")
    validation.add_argument("candidate", type=Path)
    validation.add_argument("reference", type=Path)
    validation.add_argument("--time", type=float, default=44.0)
    arguments = parser.parse_args()
    if arguments.command == "run-snapshots":
        for case in arguments.cases:
            length = None if case == "schwarzschild" else float(case)
            times = None
            if arguments.sequence is not None:
                first, last, count = arguments.sequence
                times = sequence_snapshot_times(
                    load_sourced_result(_archive(arguments.archive_root, length)),
                    first,
                    last,
                    int(count),
                    0.0005 if arguments.backend == "finite-difference" else 0.002,
                )
            print(
                run_snapshot_case(
                    length,
                    arguments.archive_root,
                    arguments.output_dir,
                    arguments.backend,
                    snapshot_times=times,
                    name_suffix=arguments.name_suffix,
                )
            )
    elif arguments.command == "sphere-time":
        print(
            plot_sphere_time(
                _archive(arguments.archive_root, 80.0), arguments.output_dir
            )
        )
    elif arguments.command == "regulator":
        print(
            plot_regulator_comparison(
                arguments.archive_root, arguments.output_dir, arguments.time
            )
        )
    elif arguments.command == "cutaway":
        print(plot_cutaway(arguments.archive, arguments.output_dir))
    elif arguments.command == "validate-dedalus":
        print(
            validate_modal_candidate(
                arguments.candidate,
                arguments.reference,
                arguments.output_dir,
                arguments.time,
            )
        )


if __name__ == "__main__":
    main()
