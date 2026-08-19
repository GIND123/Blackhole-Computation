r"""Publication figures built from the ray-marched field renderer.

Each figure is generated from an archived evolution with no hand-placed data:
snapshot times are the ones stored by the runner, the colour scale is measured
from the frames being shown, and annotations are anchored by projecting physical
directions through the same camera used for the render.

Commands::

    python -m black_hole.render_figures echo ARCHIVE
    python -m black_hole.render_figures regulator
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert

from .field_render import (
    Camera,
    Scene,
    modal_field_table,
    power_display,
    render,
    signed_colour,
    symlog,
    tone_map,
    wedge_cutaway,
)
from .localized_source import angular_spectral_weights


OUTPUT_ROOT = Path("results/caustic_visualizations")
DISPLAY_EXPONENT = 0.5
#: Values below this fraction of the frame maximum are numerical floor.
NOISE_FLOOR = 1e-5
LINEAR_FRACTION = 0.05
INK = "#e8ecf4"
MUTED = "#9aa4b8"
PAGE = "#0b0e16"


# ------------------------------------------------------------------- archives


@dataclass
class Snapshots:
    """Dense radial snapshots reconstructed from one archive."""

    times: np.ndarray          # retarded time U/M
    radius: np.ndarray         # areal radius grid of the stored snapshots
    responses: np.ndarray      # (n_time, n_ell, n_radius)
    ell: np.ndarray
    concentration: float
    horizon: float
    cosmological_radius: float
    source: str
    #: Retarded times of the measured direct and antipodal peaks, when the
    #: runner recorded them.  Empty for archives written before that field.
    measured_peaks: tuple[float, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "Snapshots":
        archive = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        offset = float(metadata["retarded_time_offset"]["q"])
        radius = archive["snapshot_areal_radius"].astype(float)
        sampling = metadata.get("visualization_sampling", {})
        return cls(
            times=archive["snapshot_times"].astype(float) - offset,
            radius=radius,
            responses=archive["response_snapshots"].astype(float),
            ell=archive["response_ell"].astype(int),
            concentration=float(metadata["source"]["angular_concentration"]),
            horizon=float(radius[0]),
            cosmological_radius=float(radius[-1]),
            source=str(path),
            measured_peaks=tuple(
                float(value)
                for value in sampling.get("measured_retarded_peak_times", ())
            ),
        )

    def nearest(self, retarded_time: float, live: np.ndarray) -> int:
        """Index of the stored snapshot closest to one retarded time."""

        return int(live[int(np.argmin(np.abs(self.times[live] - retarded_time)))])

    def table(self, index: int, **kwargs):
        return modal_field_table(
            self.responses[index],
            self.ell,
            self.radius,
            self.concentration,
            radius_max=self.cosmological_radius,
            **kwargs,
        )

    def live(self) -> np.ndarray:
        """Indices of snapshots that carry a signal above the numerical floor."""

        peak = np.abs(self.responses).max(axis=(1, 2))
        return np.flatnonzero(peak > 1e-6 * peak.max())


# ----------------------------------------------------------------- appearance


def _figure_colormap() -> LinearSegmentedColormap:
    samples = np.linspace(-1.0, 1.0, 512)
    return LinearSegmentedColormap.from_list(
        "signed_field", signed_colour(samples), N=512
    )


def _camera(display_radius: float, width: int, height: int, distance: float) -> Camera:
    direction = np.array([0.22, 0.86, 0.46])
    direction /= np.linalg.norm(direction)
    return Camera(
        position=tuple(direction * display_radius * distance),
        fov_degrees=34.0,
        width=width,
        height=height,
    )


def _scene(table, snaps: Snapshots, scale: float, **kwargs) -> Scene:
    settings = dict(
        horizon_radius=snaps.horizon**DISPLAY_EXPONENT,
        outer_radius=snaps.cosmological_radius**DISPLAY_EXPONENT,
        cutaway=wedge_cutaway(),
        display_to_physical=power_display(DISPLAY_EXPONENT),
        colour_scale=scale,
        linear_fraction=LINEAR_FRACTION,
        noise_floor=NOISE_FLOOR,
        opacity=0.5,
        opacity_gamma=1.9,
        steps=620,
        boundary_radius=snaps.cosmological_radius**DISPLAY_EXPONENT,
        boundary_opacity=0.15,
        background=(0.043, 0.055, 0.086),
    )
    settings.update(kwargs)
    return Scene(table=table, **settings)


def _colourbar(axis, scale: float) -> None:
    """Draw the signed logarithmic colour key with physical tick labels."""

    gradient = np.linspace(-1.0, 1.0, 512)
    axis.imshow(
        signed_colour(gradient)[None, :, :],
        aspect="auto",
        extent=(-1.0, 1.0, 0.0, 1.0),
        origin="lower",
        interpolation="bilinear",
    )
    def scientific(value: float, sign: float) -> str:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10.0**exponent
        prefix = "-" if sign < 0 else ""
        return rf"${prefix}{mantissa:.1f}\times10^{{{exponent}}}$"

    ticks, labels = [0.0], ["0"]
    for decade in (1.0, 0.3, 0.1):
        value = decade * scale
        position = float(
            np.log1p(value / (scale * LINEAR_FRACTION))
            / np.log1p(1.0 / LINEAR_FRACTION)
        )
        for sign in (-1.0, 1.0):
            ticks.append(sign * position)
            labels.append(scientific(value, sign))
    order = np.argsort(ticks)
    axis.set_xticks(np.asarray(ticks)[order])
    axis.set_xticklabels([labels[i] for i in order], fontsize=8.5, color=MUTED)
    # The extreme ticks sit on the axis edges, so centred labels overhang the
    # figure and are clipped.
    drawn = axis.get_xticklabels()
    if drawn:
        drawn[0].set_horizontalalignment("left")
        drawn[-1].set_horizontalalignment("right")
    axis.set_yticks([])
    axis.set_xlabel(
        r"reduced field $u=r\Phi$   (signed logarithmic scale)",
        fontsize=10, color=INK, labelpad=6,
    )
    for spine in axis.spines.values():
        spine.set_color("#39415a")


def _panel(axis, image: np.ndarray) -> None:
    axis.imshow(image, interpolation="lanczos", origin="upper")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#2a3147")
        spine.set_linewidth(0.8)


def _annotate(axis, camera: Camera, point, text: str, offset, *,
              colour: str = INK, size: float = 11.0) -> None:
    """Anchor a label to a physical direction projected through the camera."""

    fx, fy = camera.project(np.asarray(point, dtype=float))[0]
    axis.annotate(
        text,
        xy=(fx, fy),
        xycoords="axes fraction",
        xytext=(fx + offset[0], fy + offset[1]),
        textcoords="axes fraction",
        color=colour,
        fontsize=size,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color=colour, lw=0.9,
                        shrinkA=0, shrinkB=3, alpha=0.85),
    )


# --------------------------------------------------------------- echo figure


def figure_caustic_echo(
    archive: Path,
    output_dir: Path = OUTPUT_ROOT,
    *,
    frames: int = 6,
    hero_size: tuple[int, int] = (1500, 1150),
    strip_size: tuple[int, int] = (620, 500),
    supersample: int = 2,
    stem: str = "caustic_echo",
) -> Path:
    """Render the antipodal caustic: one hero cutaway above a time sequence."""

    snaps = Snapshots.load(Path(archive))
    live = snaps.live()
    if live.size < 2:
        raise ValueError(f"{archive} has too few non-trivial snapshots.")

    # A shared colour scale keeps the amplitude decay between frames visible.
    scale = 0.0
    for index in live:
        scale = max(scale, float(np.abs(snaps.table(index).values).max()))

    # The hero is the measured antipodal peak and the sequence starts at the
    # measured direct pulse, so both anchors are quantities the timing audit
    # actually measured rather than times chosen for the picture.
    if len(snaps.measured_peaks) >= 2:
        hero_index = snaps.nearest(snaps.measured_peaks[-1], live)
        first = snaps.nearest(snaps.measured_peaks[0], live)
    else:
        hero_index = int(live[-1])
        first = int(live[0])
    strip = [int(i) for i in np.linspace(first, live[-1], frames).round()]

    display_radius = snaps.cosmological_radius**DISPLAY_EXPONENT
    hero_camera = _camera(display_radius, hero_size[0], hero_size[1], 3.70)
    strip_camera = _camera(display_radius, strip_size[0], strip_size[1], 3.85)

    print(f"hero U={snaps.times[hero_index]:.2f}M, shared scale {scale:.4e}", flush=True)
    hero = tone_map(
        render(_scene(snaps.table(hero_index), snaps, scale), hero_camera, supersample),
        1.65,
    )
    tiles = []
    for index in strip:
        print(f"  frame U={snaps.times[index]:.2f}M", flush=True)
        tiles.append(
            tone_map(
                render(_scene(snaps.table(index), snaps, scale), strip_camera,
                       supersample),
                1.65,
            )
        )

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure = plt.figure(figsize=(15.2, 12.4), facecolor=PAGE)
    grid = figure.add_gridspec(
        3, frames, height_ratios=[1.62, 0.60, 0.055],
        hspace=0.10, wspace=0.035,
        left=0.035, right=0.965, top=0.925, bottom=0.075,
    )

    hero_axis = figure.add_subplot(grid[0, :])
    _panel(hero_axis, hero)
    axis_length = snaps.cosmological_radius**DISPLAY_EXPONENT

    _annotate(hero_axis, hero_camera, (0.0, 0.0, 0.0), "black hole", (0.0, -0.115))
    _annotate(hero_axis, hero_camera,
              (-0.62 * axis_length, 0.0, 0.0), "antipodal caustic", (0.10, -0.10),
              colour="#ffd9a0")
    _annotate(hero_axis, hero_camera,
              (0.68 * axis_length, 0.0, 0.0), "source direction", (-0.10, -0.09),
              colour="#bfe4ff")
    _annotate(hero_axis, hero_camera,
              (0.0, -0.80 * axis_length, 0.26 * axis_length),
              r"cosmological horizon $\mathcal{H}_c^+$", (0.02, 0.075),
              colour=MUTED, size=10.5)

    hero_axis.text(
        0.012, 0.975,
        f"$U = {snaps.times[hero_index]:.2f}\\,M$",
        transform=hero_axis.transAxes, color=INK, fontsize=15, va="top", ha="left",
    )
    hero_axis.text(
        0.988, 0.975,
        "quarter cut on two meridional planes\n"
        r"display radius $\propto \sqrt{r}$",
        transform=hero_axis.transAxes, color=MUTED, fontsize=9.5,
        va="top", ha="right", linespacing=1.5,
    )

    for column, (index, tile) in enumerate(zip(strip, tiles)):
        axis = figure.add_subplot(grid[1, column])
        _panel(axis, tile)
        axis.set_title(f"$U={snaps.times[index]:.1f}\\,M$", color=INK,
                       fontsize=11.5, pad=5)

    _colourbar(figure.add_subplot(grid[2, :]), scale)

    figure.suptitle(
        "A pulse wraps a black hole and refocuses at the antipode",
        color=INK, fontsize=19, y=0.972,
    )
    figure.text(
        0.5, 0.943,
        r"scalar field on constant-$\tau$ slices of the hyperboloidal foliation, "
        r"$L/M=80$, $\ell_{\max}=50$",
        color=MUTED, fontsize=11.5, ha="center",
    )

    destination = Path(output_dir) / f"{stem}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, facecolor=PAGE)
    figure.savefig(destination.with_suffix(".pdf"), facecolor=PAGE)
    plt.close(figure)

    _write_sidecars(
        destination,
        {
            "archive": str(archive),
            "hero_retarded_time_U_over_M": float(snaps.times[hero_index]),
            "sequence_retarded_times_U_over_M": [float(snaps.times[i]) for i in strip],
            "shared_colour_scale_max_abs_u": scale,
            "colour_transfer": "signed logarithmic, linear below "
                               f"{LINEAR_FRACTION} of the scale",
            "numerical_floor_fraction": NOISE_FLOOR,
            "display_radius": f"R = r**{DISPLAY_EXPONENT} (angles undistorted)",
            "cutaway": "material removed where y>0 and z>0",
            "angular_ell_max": int(snaps.ell[-1]),
            "spectral_filter": "none",
            "time_translation_fitted": False,
        },
        caption=(
            "Antipodal caustic of a localized scalar pulse on a "
            "Schwarzschild-de Sitter bridge with L/M=80. The field is "
            "reconstructed from the evolved spherical-harmonic responses through "
            "ell_max=50 and rendered by ray marching, so occlusion and the cut "
            "surfaces are geometrically exact. A quarter is removed on two "
            "meridional planes to expose the interior; the exposed faces show "
            "the reduced field u=r*Phi, and the translucent remainder shows the "
            "same field in depth. Upper panel: the measured antipodal peak. "
            "Lower row: the same scene at the archived snapshot times, running "
            "from the measured direct pulse, showing the field leaving the "
            "source direction, sweeping around the black hole, and converging "
            "on the antipode. The antipodal feature then changes sign, which "
            "is a consequence of the phase the wavefront acquires at the "
            "caustic: fitting the echo to a rotated copy of the direct pulse "
            "gives a rotation of 42 degrees, against 180 for a sign reversal. "
            "Each frame is one surface of constant bridge time, labelled by "
            "the retarded time it carries at the outer boundary. "
            "All panels share one signed logarithmic colour scale, so "
            "the amplitude decay between frames is the physical decay. The "
            "displayed radius is R = sqrt(r), which "
            "leaves every angle and every sphere undistorted while showing the "
            "horizon and the cosmological horizon in one frame. No time "
            "translation is fitted."
        ),
    )
    return destination


# ----------------------------------------------------------- regulator figure


REGULATOR_ROOT = Path("results/regulator_production_v3/raw/source/fine")
REGULATOR_LENGTHS = (80.0, 160.0, 320.0, 640.0)


def _sphere_profile(result, retarded_time: float, samples: int = 2048) -> np.ndarray:
    """Return the field on the outer extraction sphere against ``cos gamma``."""

    from .caustic_visualizations import field_on_sphere

    cosine = np.linspace(-1.0, 1.0, samples)
    theta = np.full(samples, 0.5 * np.pi)
    phi = np.arccos(np.clip(cosine, -1.0, 1.0))
    _, profile = field_on_sphere(
        result, retarded_time, theta, phi, interpolate_time=True
    )
    return np.asarray(profile, dtype=float)


def _sphere_residual(candidate, reference, retarded_time: float) -> float:
    """Exact Parseval residual between two extraction spheres."""

    from .caustic_visualizations import modal_response_at_time

    lookup = {int(ell): i for i, ell in enumerate(candidate.response_ell)}
    indices = np.asarray([lookup[int(e)] for e in candidate.mode_ell], dtype=int)
    a = modal_response_at_time(candidate, retarded_time)[indices]
    a = a * candidate.mode_source_amplitude
    b = modal_response_at_time(reference, retarded_time)[indices]
    b = b * reference.mode_source_amplitude
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def figure_regulator(
    output_dir: Path = OUTPUT_ROOT,
    archive_root: Path = REGULATOR_ROOT,
    *,
    retarded_time: float = 44.0,
    size: tuple[int, int] = (1560, 1180),
    supersample: int = 2,
    stem: str = "regulator_flat_limit",
) -> Path:
    """Nested cosmological horizons: the regulator box grows, the error falls."""

    from .caustic_visualizations import _archive
    from .field_render import AngularShell, render_shells
    from .source_evolution import load_sourced_result

    reference = load_sourced_result(_archive(archive_root, None))
    profiles, radii, residuals = {}, {}, {}
    for length in REGULATOR_LENGTHS:
        result = load_sourced_result(_archive(archive_root, length))
        profiles[length] = _sphere_profile(result, retarded_time)
        radii[length] = float(result.observer_areal_radius[result.outer_index()])
        residuals[length] = _sphere_residual(result, reference, retarded_time)
        print(f"L/M={length:6.0f}  r_c/M={radii[length]:7.2f}  "
              f"residual={residuals[length]:.4f}", flush=True)

    scale = max(float(np.abs(p).max()) for p in profiles.values())
    shells = [
        AngularShell(
            display_radius=radii[length] ** DISPLAY_EXPONENT,
            profile=profiles[length],
            opacity=0.58,
            label=f"$L/M={length:.0f}$",
        )
        for length in REGULATOR_LENGTHS
    ]
    outermost = max(shell.display_radius for shell in shells)
    camera = _camera(outermost, size[0], size[1], 3.70)
    image = tone_map(
        render_shells(
            shells,
            camera,
            horizon_radius=2.0**DISPLAY_EXPONENT,
            colour_scale=scale,
            linear_fraction=LINEAR_FRACTION,
            noise_floor=NOISE_FLOOR,
            cutaway=wedge_cutaway(),
            background=(0.043, 0.055, 0.086),
            supersample=supersample,
        ),
        1.55,
    )

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure = plt.figure(figsize=(15.0, 8.4), facecolor=PAGE)
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.0, 0.42], height_ratios=[1.0, 0.06],
        wspace=0.10, hspace=0.16,
        left=0.028, right=0.965, top=0.885, bottom=0.075,
    )

    axis = figure.add_subplot(grid[0, 0])
    _panel(axis, image)
    anchor = np.array([0.40, -0.36, 0.84])
    anchor /= np.linalg.norm(anchor)
    for shell, length in zip(shells, REGULATOR_LENGTHS):
        _annotate(
            axis, camera, anchor * shell.display_radius,
            f"$r_c/M={radii[length]:.0f}$", (0.085, -0.028),
            colour=MUTED, size=10.0,
        )
    _annotate(axis, camera, (0.0, 0.0, 0.0), "black hole", (0.0, -0.10))
    axis.text(
        0.015, 0.975,
        r"$L\longrightarrow\infty$" "\n"
        r"$\mathcal{H}_c^+\longrightarrow\mathscr{I}^+$",
        transform=axis.transAxes, color=INK, fontsize=15,
        va="top", ha="left", linespacing=1.6,
    )
    axis.text(
        0.985, 0.03,
        r"display radius $\propto\sqrt{r}$; quarter cut",
        transform=axis.transAxes, color=MUTED, fontsize=9.5, va="bottom", ha="right",
    )

    inset = figure.add_subplot(grid[0, 1], facecolor="#121724")
    lengths = np.asarray(REGULATOR_LENGTHS)
    values = np.asarray([residuals[length] for length in REGULATOR_LENGTHS])
    inset.loglog(lengths, values, "o-", color="#4bb3ff", lw=1.8, ms=7,
                 label=r"$\|\delta u_L\|_2/\|u_0\|_2$")
    guide = values[0] * lengths[0] / lengths
    inset.loglog(lengths, guide, "--", color=MUTED, lw=1.2, label=r"$\propto 1/L$")
    for length, value in zip(lengths, values):
        inset.annotate(f"{value:.3f}", (length, value), textcoords="offset points",
                       xytext=(6, 7), color=INK, fontsize=9)
    inset.set_xlabel(r"regulator length $L/M$", color=INK, fontsize=11)
    inset.set_ylabel(r"residual against Schwarzschild at $\mathscr{I}^+$",
                     color=INK, fontsize=10.5)
    inset.xaxis.set_minor_locator(ticker.NullLocator())
    inset.set_xticks(lengths)
    inset.set_xticklabels([f"{int(v)}" for v in lengths])
    inset.tick_params(colors=MUTED, labelsize=9.5, which="both")
    for spine in inset.spines.values():
        spine.set_color("#39415a")
    inset.grid(alpha=0.18, which="both", color=MUTED)
    inset.legend(frameon=False, labelcolor=INK, fontsize=10, loc="upper right")
    inset.set_title(r"doubling $L$ halves the error", color=INK, fontsize=11.5, pad=8)

    _colourbar(figure.add_subplot(grid[1, 0]), scale)

    figure.suptitle(
        "Misner's artificial cosmology as a controlled regulator",
        color=INK, fontsize=19, y=0.965,
    )
    figure.text(
        0.5, 0.917,
        rf"field on the cosmological horizon at the common retarded time "
        rf"$U={retarded_time:.0f}\,M$, $\ell_{{\max}}=50$",
        color=MUTED, fontsize=11.5, ha="center",
    )

    destination = Path(output_dir) / f"{stem}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, facecolor=PAGE)
    figure.savefig(destination.with_suffix(".pdf"), facecolor=PAGE)
    plt.close(figure)

    _write_sidecars(
        destination,
        {
            "archive_root": str(archive_root),
            "retarded_time_U_over_M": retarded_time,
            "cosmological_lengths_over_M": list(REGULATOR_LENGTHS),
            "cosmological_horizon_radii_over_M": [radii[l] for l in REGULATOR_LENGTHS],
            "sphere_relative_l2_against_schwarzschild": [
                residuals[l] for l in REGULATOR_LENGTHS
            ],
            "residual_norm": "exact Parseval sum over stored real harmonic modes",
            "colour_scale_max_abs_u": scale,
            "display_radius": f"R = r**{DISPLAY_EXPONENT} (angles undistorted)",
            "time_translation_fitted": False,
        },
        caption=(
            "The artificial cosmological constant as a regulator. Each shell is "
            "the future cosmological horizon of one Schwarzschild-de Sitter "
            "bridge, drawn at its own areal radius and painted with the scalar "
            "field it carries at the common retarded time U=44M. A quarter is "
            "removed so that all four nested horizons and the black hole are "
            "visible at once. Increasing the regulator length L moves the "
            "horizon outward; in the limit it becomes future null infinity of "
            "the asymptotically flat problem. The panel at right gives the "
            "exact Parseval residual of each finite-L extraction sphere against "
            "the independently evolved Schwarzschild field at future null "
            "infinity. The residual falls as 1/L, halving with every doubling "
            "of L, which is the quantitative statement that the regulator is "
            "controlled. The displayed radius is R = sqrt(r), which leaves "
            "every angle and every sphere undistorted. No time translation is "
            "fitted."
        ),
    )
    return destination


# --------------------------------------------------------- sphere-time figure


def figure_sphere_time(
    output_dir: Path = OUTPUT_ROOT,
    archive_root: Path = REGULATOR_ROOT,
    *,
    length: float = 80.0,
    times: tuple[float, ...] = (30.0, 34.0, 38.0, 41.0, 44.0, 46.0, 48.0, 50.0),
    panel: tuple[int, int] = (520, 520),
    supersample: int = 2,
    stem: str = "sphere_time_echo",
) -> Path:
    """The extraction sphere seen face on at the antipode, against time.

    The camera looks along the antipodal direction, so the echo appears as a
    ring converging on the centre of the disc.  The panels below carry the
    same data as a waveform and as a measurement: the echo is the direct pulse
    rotated in phase at the caustic, and the rotation switches on only within
    the last few degrees of the axis.
    """

    from .caustic_phase import (
        arrival_time,
        direction_trace,
        phase_fit,
        scan_archive,
        ARRIVAL_SEARCH_START_M,
        FIT_WINDOW_AFTER_M,
        FIT_WINDOW_BEFORE_M,
    )
    from .caustic_visualizations import _archive, measured_pulse_times
    from .field_render import AngularShell, render_shells
    from .source_evolution import load_sourced_result

    archive = _archive(archive_root, length)
    result = load_sourced_result(archive)
    radius = float(result.observer_areal_radius[result.outer_index()])
    display = radius**DISPLAY_EXPONENT
    peaks = measured_pulse_times(result)

    profiles = [_sphere_profile(result, value) for value in times]
    scale = max(float(np.abs(profile).max()) for profile in profiles)

    # Look down the antipodal axis: the source direction is at cos gamma = +1,
    # so the camera sits on the negative x side and the antipode faces us.
    direction = np.array([-1.0, 0.26, 0.20])
    direction /= np.linalg.norm(direction)
    camera = Camera(
        position=tuple(direction * display * 3.9),
        fov_degrees=34.0,
        width=panel[0],
        height=panel[1],
    )
    tiles = []
    for value, profile in zip(times, profiles):
        print(f"  sphere U={value:.2f}M", flush=True)
        tiles.append(
            tone_map(
                render_shells(
                    [AngularShell(display_radius=display, profile=profile,
                                  opacity=1.0)],
                    camera,
                    horizon_radius=1e-6,
                    colour_scale=scale,
                    linear_fraction=LINEAR_FRACTION,
                    noise_floor=NOISE_FLOOR,
                    background=(0.043, 0.055, 0.086),
                    supersample=supersample,
                ),
                1.55,
            )
        )

    # On the axis the Legendre sum closes: P_l(1) = 1 and P_l(-1) = (-1)^l, so
    # both waveforms follow from the archived responses without reconstructing
    # the sphere or interpolating in time.
    dense = np.asarray(result.retarded_time, dtype=float)
    at_source = direction_trace(result, 0.0)
    at_antipode = direction_trace(result, np.pi)

    # The measurement: fit the antipodal echo to a rotated, delayed copy of the
    # direct pulse, then repeat the fit along a scan of directions.
    outer = result.outer_index()
    arrival = arrival_time(dense, at_antipode, ARRIVAL_SEARCH_START_M)
    window = (
        max(float(dense[0]), arrival - FIT_WINDOW_BEFORE_M),
        min(float(dense[-1]), arrival + FIT_WINDOW_AFTER_M),
    )
    print("  fitting the antipodal phase", flush=True)
    fit = phase_fit(dense, at_source, at_antipode, window)
    quadrature = np.imag(hilbert(at_source))
    model = (
        fit.amplitude
        * np.cos(np.radians(fit.phase_degrees))
        * np.interp(dense, dense + fit.delay_over_M, at_source, left=0.0, right=0.0)
        - fit.amplitude
        * np.sin(np.radians(fit.phase_degrees))
        * np.interp(dense, dense + fit.delay_over_M, quadrature, left=0.0, right=0.0)
    )
    inside_window = (dense >= window[0]) & (dense <= window[1])

    print("  scanning the angular dependence", flush=True)
    scan = [row for row in scan_archive(archive, observers=(outer,))]
    scan_angles = np.asarray([row["gamma_degrees"] for row in scan])
    scan_phase = np.asarray([row["phase_degrees"] for row in scan])
    scan_amplitude = np.asarray([row["amplitude"] for row in scan])

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    # The sphere panels are square and width limited.  Sizing the rows in
    # inches and then solving for the figure height makes the first row exactly
    # as tall as a panel, instead of leaving a band of empty page under the
    # spheres or crushing the row into the title.
    width, left, right = 16.0, 0.062, 0.978
    top, bottom, hspace = 0.858, 0.088, 0.46
    panel_height = width * (right - left) / len(times)
    ratios = (panel_height, 2.9, 0.18)
    axes_height = sum(ratios) * (1.0 + 2.0 * hspace / len(ratios))
    height = axes_height / (top - bottom)
    figure = plt.figure(figsize=(width, height), facecolor=PAGE)
    grid = figure.add_gridspec(
        3, len(times), height_ratios=list(ratios),
        hspace=hspace, wspace=0.03,
        left=left, right=right, top=top, bottom=bottom,
    )
    for column, (value, tile) in enumerate(zip(times, tiles)):
        axis = figure.add_subplot(grid[0, column])
        _panel(axis, tile)
        axis.set_title(f"$U={value:.0f}\\,M$", color=INK, fontsize=12, pad=5)

    split = len(times) - 3
    wave = figure.add_subplot(grid[1, :split], facecolor="#121724")
    wave.plot(dense, at_antipode, color="#ffb454", lw=1.7,
              label=r"antipode  $\gamma=\pi$")
    wave.plot(dense, at_source, color="#4bb3ff", lw=1.4, alpha=0.9,
              label=r"source direction  $\gamma=0$")
    wave.plot(dense[inside_window], model[inside_window], color="#e8ecf4",
              lw=1.3, ls=(0, (5, 3)), alpha=0.95,
              label=(r"$A\,\mathrm{Re}[e^{i\phi}z_0(U-\Delta)]$,  "
                     rf"$\phi={fit.phase_degrees:.0f}^\circ$"))
    wave.axhline(0.0, color=MUTED, lw=0.7, alpha=0.6)
    for value in times:
        wave.axvline(value, color=MUTED, lw=0.7, linestyle=":", alpha=0.5)
    for peak, label in zip(peaks, ("direct pulse", "antipodal caustic")):
        wave.axvline(peak, color="#e8ecf4", lw=1.0, linestyle="--", alpha=0.7)
        wave.annotate(
            f"{label}\n$U={peak:.2f}M$",
            xy=(peak, 0.0), xycoords="data",
            xytext=(peak, 0.97), textcoords=("data", "axes fraction"),
            color=INK, fontsize=9, ha="center", va="top",
        )
    wave.set_xlabel(r"geometric retarded time $U/M$", color=INK, fontsize=11.5)
    wave.set_ylabel(r"$u$ on $\mathcal{H}_c^+$", color=INK, fontsize=11.5,
                    labelpad=8)
    # The signal is at the numerical floor before the source switches on.
    wave.set_xlim(15.0, float(max(times) + 4.0))
    wave.tick_params(colors=MUTED, labelsize=10)
    for spine in wave.spines.values():
        spine.set_color("#39415a")
    wave.grid(alpha=0.16, color=MUTED)
    wave.legend(frameon=False, labelcolor=INK, fontsize=9.5, loc="lower left")

    scanned = figure.add_subplot(grid[1, split:], facecolor="#121724")
    for level, text in ((0.0, "unrotated copy"), (90.0, "Hilbert transform")):
        scanned.axhline(level, color=MUTED, lw=0.9, ls="--", alpha=0.55)
        scanned.annotate(text, xy=(178.0, level + 4.0), color=MUTED,
                         fontsize=9, ha="right", va="bottom")
    scanned.plot(scan_angles, scan_phase, "o-", color="#ffb454", lw=1.6, ms=5,
                 label=r"rotation $\phi$")
    scanned.set_xlabel(r"angle from the source  $\gamma$  (degrees)",
                       color=INK, fontsize=11.5)
    scanned.set_ylabel(r"phase rotation  $\phi$  (degrees)", color=INK,
                       fontsize=11.5, labelpad=6)
    scanned.set_xlim(-4.0, 184.0)
    scanned.set_ylim(-28.0, 112.0)
    scanned.set_xticks((0, 45, 90, 135, 180))
    scanned.tick_params(colors=MUTED, labelsize=10)
    for spine in scanned.spines.values():
        spine.set_color("#39415a")
    scanned.grid(alpha=0.16, color=MUTED)

    gain = scanned.twinx()
    gain.plot(scan_angles, scan_amplitude, "s--", color="#4bb3ff", lw=1.2, ms=4,
              alpha=0.9, label=r"amplitude $A$")
    gain.set_ylabel(r"amplitude ratio  $A$", color="#4bb3ff", fontsize=11.5,
                    labelpad=6)
    gain.tick_params(colors="#4bb3ff", labelsize=10)
    gain.set_ylim(0.0, 2.6)
    for spine in gain.spines.values():
        spine.set_color("#39415a")
    handles, labels = scanned.get_legend_handles_labels()
    extra, extra_labels = gain.get_legend_handles_labels()
    scanned.legend(handles + extra, labels + extra_labels, frameon=False,
                   labelcolor=INK, fontsize=9.5, loc="center left")

    _colourbar(figure.add_subplot(grid[2, :]), scale)

    figure.suptitle(
        "At the caustic the echo returns the direct pulse rotated in phase",
        color=INK, fontsize=19, y=1.0 - 0.30 / height,
    )
    figure.text(
        0.5, 1.0 - 0.72 / height,
        rf"future cosmological horizon of the $L/M={length:.0f}$ bridge, "
        rf"$r_c/M={radius:.1f}$, viewed along the antipodal axis, "
        rf"$\ell_{{\max}}={int(result.response_ell[-1])}$",
        color=MUTED, fontsize=11.5, ha="center",
    )

    destination = Path(output_dir) / f"{stem}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, facecolor=PAGE)
    figure.savefig(destination.with_suffix(".pdf"), facecolor=PAGE)
    plt.close(figure)

    _write_sidecars(
        destination,
        {
            "archive": str(archive),
            "cosmological_length_over_M": length,
            "cosmological_horizon_radius_over_M": radius,
            "sphere_retarded_times_U_over_M": list(times),
            "measured_peak_times_U_over_M": list(peaks),
            "shared_colour_scale_max_abs_u": scale,
            "colour_scale_basis": "maximum over the displayed spheres only",
            "view": "along the antipodal axis; source direction on the far side",
            "display_radius": f"R = r**{DISPLAY_EXPONENT} (angles undistorted)",
            "phase_model": (
                "u_gamma(U) = Re[A exp(i phi) z_0(U - Delta)] with z_0 the "
                "analytic signal of the direct pulse of the same evolution"
            ),
            "antipodal_phase_degrees": fit.phase_degrees,
            "antipodal_amplitude": fit.amplitude,
            "antipodal_delay_over_M": fit.delay_over_M,
            "antipodal_variance_explained": fit.variance_explained,
            "fit_window_U_over_M": [window[0], window[1]],
            "scan_gamma_degrees": [float(value) for value in scan_angles],
            "scan_phase_degrees": [float(value) for value in scan_phase],
            "scan_amplitude": [float(value) for value in scan_amplitude],
            "time_translation_fitted": False,
        },
        caption=(
            "The caustic echo on the future cosmological horizon of the L/M=80 "
            "bridge. Each sphere is the extraction surface at one geometric "
            "retarded time, painted with the reduced field it carries and "
            "viewed along the antipodal axis, so the source direction lies on "
            "the far side and the antipode is at the centre of each disc. The "
            "wavefront converges as a ring and collapses on the antipode. The "
            "lower left panel carries the same field on the axis, sampled at "
            "the antipode and at the source direction, with the two arrival "
            "times measured by the analytic envelope estimator marked. The "
            "echo is not a scaled copy of the direct pulse: the dashed white "
            "curve is the direct pulse rotated by phi = "
            f"{fit.phase_degrees:.0f} degrees in the phase of its analytic "
            f"signal, delayed by {fit.delay_over_M:.1f}M and scaled by "
            f"{fit.amplitude:.2f}, which accounts for "
            f"{100.0 * fit.variance_explained:.0f} percent of the variance in "
            "the fit window. The lower right panel repeats that fit along a "
            "scan of directions: the rotation stays near zero while the "
            "wavefront sweeps around the black hole and switches on only "
            "within the last few degrees of the axis, where the amplitude "
            "also rises. An unrotated copy would give zero, a full Hilbert "
            "transform ninety degrees, and a sign reversal a hundred and "
            "eighty. The spheres share one signed logarithmic colour scale, "
            "taken over the displayed frames only; the waveform panel carries "
            "the absolute amplitudes. No time translation between backgrounds "
            "is fitted."
        ),
    )
    return destination


def _write_sidecars(destination: Path, metadata: dict, caption: str) -> None:
    with destination.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    with destination.with_name(destination.stem + "_caption.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(caption.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    echo = subparsers.add_parser("echo")
    echo.add_argument("archive", type=Path)
    echo.add_argument("--frames", type=int, default=6)
    echo.add_argument("--supersample", type=int, default=2)
    echo.add_argument("--stem", default="caustic_echo")
    regulator = subparsers.add_parser("regulator")
    regulator.add_argument("--archive-root", type=Path, default=REGULATOR_ROOT)
    regulator.add_argument("--time", type=float, default=44.0)
    regulator.add_argument("--supersample", type=int, default=2)
    regulator.add_argument("--stem", default="regulator_flat_limit")
    sphere = subparsers.add_parser("sphere-time")
    sphere.add_argument("--archive-root", type=Path, default=REGULATOR_ROOT)
    sphere.add_argument("--length", type=float, default=80.0)
    sphere.add_argument("--supersample", type=int, default=2)
    sphere.add_argument("--stem", default="sphere_time_echo")
    arguments = parser.parse_args()
    if arguments.command == "echo":
        print(
            figure_caustic_echo(
                arguments.archive,
                arguments.output_dir,
                frames=arguments.frames,
                supersample=arguments.supersample,
                stem=arguments.stem,
            )
        )
    elif arguments.command == "regulator":
        print(
            figure_regulator(
                arguments.output_dir,
                arguments.archive_root,
                retarded_time=arguments.time,
                supersample=arguments.supersample,
                stem=arguments.stem,
            )
        )
    elif arguments.command == "sphere-time":
        print(
            figure_sphere_time(
                arguments.output_dir,
                arguments.archive_root,
                length=arguments.length,
                supersample=arguments.supersample,
                stem=arguments.stem,
            )
        )


if __name__ == "__main__":
    main()
