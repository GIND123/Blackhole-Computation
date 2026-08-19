r"""Ray-marched rendering of the reconstructed three-dimensional scalar field.

The archived evolutions store one radial response :math:`R_\ell(r)` for each
retained :math:`\ell`.  Because every background used here is spherically
symmetric and the localized source is axisymmetric about its own direction, the
spherical addition theorem collapses the reconstruction to

.. math::

   u(r,\gamma) = \sum_{\ell} R_\ell(r)\, w_\ell\,
                 \frac{2\ell+1}{4\pi}\, P_\ell(\cos\gamma),

with :math:`\gamma` the angle from the source direction.  That is an exact
rewriting of the full angle-dependent field, not an axisymmetric approximation
of it: evolving every excited :math:`(\ell,m)` separately gives the same field.
The renderer therefore tabulates :math:`u` once on a uniform
:math:`(r,\cos\gamma)` grid and treats it as a genuine three-dimensional volume.

Rendering is deliberately written against NumPy alone.  ``matplotlib``'s 3D
axes use a painter's algorithm with no depth buffer, which is what produces the
moire banding and incorrect occlusion in flat "3D" surface plots.  Marching
rays gives correct occlusion, a real cut surface, and control over the optical
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Callable

import numpy as np
from scipy import special

from .localized_source import angular_spectral_weights


# ---------------------------------------------------------------- field table


@dataclass
class FieldTable:
    """The reduced field ``u`` on a uniform ``(r, cos gamma)`` grid."""

    values: np.ndarray  # (n_radius, n_angle), float32
    radius_min: float
    radius_max: float

    @property
    def n_radius(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_angle(self) -> int:
        return int(self.values.shape[1])

    def sample(self, radius: np.ndarray, cosine: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate the table; zero outside the radial range."""

        span = self.radius_max - self.radius_min
        fr = (radius - self.radius_min) * ((self.n_radius - 1) / span)
        fg = (cosine + 1.0) * ((self.n_angle - 1) / 2.0)
        inside = (fr >= 0.0) & (fr <= self.n_radius - 1)
        fr = np.clip(fr, 0.0, self.n_radius - 1 - 1e-6)
        fg = np.clip(fg, 0.0, self.n_angle - 1 - 1e-6)
        i0 = fr.astype(np.int32)
        j0 = fg.astype(np.int32)
        ti = (fr - i0).astype(np.float32)
        tj = (fg - j0).astype(np.float32)
        flat = self.values.ravel()
        base = i0 * self.n_angle + j0
        v00 = flat[base]
        v01 = flat[base + 1]
        v10 = flat[base + self.n_angle]
        v11 = flat[base + self.n_angle + 1]
        top = v00 + (v01 - v00) * tj
        bottom = v10 + (v11 - v10) * tj
        return np.where(inside, top + (bottom - top) * ti, 0.0).astype(np.float32)


def modal_field_table(
    response: np.ndarray,
    response_ell: np.ndarray,
    radius: np.ndarray,
    angular_concentration: float,
    *,
    n_radius: int = 3072,
    n_angle: int = 2048,
    radius_max: float | None = None,
    spectral_filter: float | None = None,
) -> FieldTable:
    """Build a uniform ``(r, cos gamma)`` table from stored radial responses.

    ``spectral_filter`` applies an exponential (Vandeven-type) filter
    ``exp(-alpha (ell/ell_max)^{2s})`` with ``alpha = spectral_filter``.  A
    caustic is a genuine geometric-optics singularity, so a truncated Legendre
    sum rings near the focus; the filter suppresses that ringing for display.
    It is off by default and every figure that uses it says so.
    """

    response = np.asarray(response, dtype=float)
    response_ell = np.asarray(response_ell, dtype=int)
    radius = np.asarray(radius, dtype=float)
    weights = angular_spectral_weights(
        float(angular_concentration), int(response_ell[-1])
    )
    coefficients = (
        weights[response_ell] * (2.0 * response_ell + 1.0) / (4.0 * np.pi)
    )
    if spectral_filter is not None:
        ell_max = float(response_ell[-1])
        coefficients = coefficients * np.exp(
            -float(spectral_filter) * (response_ell / ell_max) ** 8
        )

    inner = float(radius[0])
    outer = float(radius[-1]) if radius_max is None else float(radius_max)
    uniform_radius = np.linspace(inner, outer, n_radius)
    cosine = np.linspace(-1.0, 1.0, n_angle)

    legendre = np.empty((response_ell.size, n_angle))
    for index, ell in enumerate(response_ell):
        legendre[index] = special.eval_legendre(int(ell), cosine)
    legendre *= coefficients[:, None]

    # Resample each radial response onto the uniform grid before contracting.
    resampled = np.empty((response_ell.size, n_radius))
    for index in range(response_ell.size):
        resampled[index] = np.interp(uniform_radius, radius, response[index])

    values = (resampled.T @ legendre).astype(np.float32)
    return FieldTable(values=values, radius_min=inner, radius_max=outer)


# --------------------------------------------------------------------- camera


@dataclass
class Camera:
    """A pinhole camera looking at ``target`` from ``position``."""

    position: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_degrees: float = 34.0
    width: int = 1600
    height: int = 1100

    def rays(self, supersample: int = 1) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Return the origin and one unit direction per (supersampled) pixel."""

        origin = np.asarray(self.position, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        forward = target - origin
        forward /= np.linalg.norm(forward)
        up = np.asarray(self.up, dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        width = self.width * supersample
        height = self.height * supersample
        aspect = width / height
        half = np.tan(np.radians(self.fov_degrees) / 2.0)
        xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
        ys = 1.0 - (np.arange(height) + 0.5) / height * 2.0
        grid_x, grid_y = np.meshgrid(xs * half * aspect, ys * half)
        directions = (
            forward[None, None, :]
            + grid_x[:, :, None] * right[None, None, :]
            + grid_y[:, :, None] * true_up[None, None, :]
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        return origin, directions.reshape(-1, 3), (height, width)

    def _frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        origin = np.asarray(self.position, dtype=np.float64)
        forward = np.asarray(self.target, dtype=np.float64) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(self.up, dtype=np.float64))
        right /= np.linalg.norm(right)
        return origin, forward, right, np.cross(right, forward)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project world points to fractional image coordinates in [0, 1].

        The origin is the top-left corner, matching ``imshow`` axes, so
        annotations can be anchored to physical directions instead of being
        positioned by hand.
        """

        origin, forward, right, true_up = self._frame()
        offset = np.atleast_2d(np.asarray(points, dtype=np.float64)) - origin
        depth = offset @ forward
        half = np.tan(np.radians(self.fov_degrees) / 2.0)
        aspect = self.width / self.height
        ndc_x = (offset @ right) / (depth * half * aspect)
        ndc_y = (offset @ true_up) / (depth * half)
        return np.stack(((ndc_x + 1.0) / 2.0, (1.0 - ndc_y) / 2.0), axis=-1)


# ----------------------------------------------------------------- appearance


def symlog(
    values: np.ndarray,
    scale: float,
    linear_fraction: float,
    noise_floor: float = 0.0,
) -> np.ndarray:
    """Map a signed field to [-1, 1], compressing several decades.

    ``noise_floor`` is a fraction of ``scale`` below which the field is treated
    as exactly zero.  A logarithmic transfer function otherwise assigns visible
    colour to round-off: in these archives the region the wave has not reached
    sits near ``1e-9`` of the frame maximum and differs between the Dedalus and
    finite-difference codes, so rendering it would show numerical floor as if it
    were structure.
    """

    threshold = scale * linear_fraction
    magnitude = np.abs(values) / threshold
    compressed = np.log1p(magnitude) / np.log1p(1.0 / linear_fraction)
    compressed = np.clip(compressed, 0.0, 1.0)
    if noise_floor > 0.0:
        compressed = np.where(np.abs(values) < noise_floor * scale, 0.0, compressed)
    return np.sign(values) * compressed


#: Emission colour ramps.  Negative values run cool, positive values warm, and
#: zero is transparent, so the wavefront reads as light emitted by the field.
#: Both ramps start from the same neutral dark value.  If the two ramps
#: disagreed at zero, every point the wave has not reached would be tinted with
#: whichever sign won the comparison, painting the undisturbed region a colour.
_ZERO = [0.035, 0.045, 0.070]
_NEGATIVE_RAMP = np.array(
    [_ZERO, [0.05, 0.28, 0.62], [0.15, 0.62, 0.92],
     [0.55, 0.88, 1.00], [0.90, 0.98, 1.00]]
)
_POSITIVE_RAMP = np.array(
    [_ZERO, [0.55, 0.16, 0.06], [0.92, 0.42, 0.05],
     [1.00, 0.75, 0.28], [1.00, 0.97, 0.85]]
)


def _ramp(ramp: np.ndarray, t: np.ndarray) -> np.ndarray:
    position = np.clip(t, 0.0, 1.0) * (ramp.shape[0] - 1)
    index = np.clip(position.astype(np.int32), 0, ramp.shape[0] - 2)
    frac = (position - index)[:, None]
    return ramp[index] * (1.0 - frac) + ramp[index + 1] * frac


def signed_colour(normalized: np.ndarray) -> np.ndarray:
    """Return an RGB emission colour for a value already in [-1, 1]."""

    magnitude = np.abs(normalized)
    warm = _ramp(_POSITIVE_RAMP, magnitude)
    cool = _ramp(_NEGATIVE_RAMP, magnitude)
    return np.where((normalized >= 0.0)[:, None], warm, cool)


# ------------------------------------------------------------------- geometry


def _sphere_span(
    origin: np.ndarray, directions: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the entry and exit parameters of each ray on a centred sphere."""

    b = directions @ origin
    c = float(origin @ origin) - radius * radius
    discriminant = b * b - c
    hit = discriminant > 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    return hit, -b - root, -b + root


@dataclass
class Scene:
    """Everything the renderer needs about one snapshot."""

    table: FieldTable
    horizon_radius: float
    outer_radius: float
    #: Predicate marking removed material, evaluated on (n, 3) positions.
    cutaway: Callable[[np.ndarray], np.ndarray] | None = None
    #: Optional map from display radius to areal radius.  Rays march in display
    #: space; only the radial coordinate is remapped, so every angular feature
    #: and every sphere is reproduced exactly.  Used to give the horizon visible
    #: extent while still showing the domain out to the cosmological horizon.
    display_to_physical: Callable[[np.ndarray], np.ndarray] | None = None
    colour_scale: float = 1.0
    linear_fraction: float = 3e-3
    opacity: float = 0.55
    opacity_gamma: float = 1.35
    #: Fraction of ``colour_scale`` treated as exactly zero when shading.
    noise_floor: float = 1e-5
    background: tuple[float, float, float] = (0.024, 0.028, 0.042)
    slice_ambient: float = 0.030
    horizon_colour: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Optional faint shell drawn where the cosmological horizon sits.
    boundary_radius: float | None = None
    boundary_colour: tuple[float, float, float] = (0.45, 0.42, 0.58)
    boundary_opacity: float = 0.16
    steps: int = 720
    metadata: dict = dataclass_field(default_factory=dict)


def _physical(scene: "Scene", radius: np.ndarray) -> np.ndarray:
    if scene.display_to_physical is None:
        return radius
    return scene.display_to_physical(radius)


def power_display(exponent: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
    """Return the display-to-areal map for ``R = r**exponent``.

    With ``exponent = 1/2`` the horizon of a Schwarzschild-de Sitter bridge with
    ``L/M = 80`` occupies about a sixth of the displayed radius instead of a
    fortieth, so the black hole is visible in the same frame as the
    cosmological horizon.
    """

    def to_physical(display: np.ndarray) -> np.ndarray:
        return np.power(np.maximum(display, 0.0), 1.0 / exponent)

    return to_physical


def render(scene: Scene, camera: Camera, supersample: int = 2) -> np.ndarray:
    """Render one scene, returning an ``(height, width, 3)`` image in [0, 1]."""

    origin, directions, (height, width) = camera.rays(supersample)
    n_rays = directions.shape[0]
    image = np.tile(np.asarray(scene.background, dtype=np.float32), (n_rays, 1))

    hit, t_near, t_far = _sphere_span(origin, directions, scene.outer_radius)
    active = hit & (t_far > 0.0)
    if not np.any(active):
        return image.reshape(height, width, 3)

    index = np.flatnonzero(active)
    dirs = directions[index]
    start = np.maximum(t_near[index], 0.0)
    stop = t_far[index]
    step = (stop - start) / scene.steps

    colour = np.zeros((index.size, 3), dtype=np.float32)
    alpha = np.zeros(index.size, dtype=np.float32)
    done = np.zeros(index.size, dtype=bool)
    # A ray that enters the volume outside the wedge must not be treated as
    # having just crossed a cut face on its first sample.
    previous_removed = np.zeros(index.size, dtype=bool)

    boundary_band = (
        None
        if scene.boundary_radius is None
        else 0.0075 * scene.outer_radius
    )

    for sample in range(scene.steps):
        live = ~done & (alpha < 0.996)
        if not np.any(live):
            break
        where = np.flatnonzero(live)
        t = start[where] + (sample + 0.5) * step[where]
        points = origin[None, :] + dirs[where] * t[:, None]
        radius = np.sqrt(np.einsum("ij,ij->i", points, points))

        removed = (
            np.zeros(where.size, dtype=bool)
            if scene.cutaway is None
            else scene.cutaway(points)
        )
        was_removed = previous_removed[where]
        previous_removed[where] = removed

        # 1. The black hole terminates any ray that reaches it.
        inside = (radius <= scene.horizon_radius) & ~removed
        if np.any(inside):
            target = where[inside]
            colour[target] += (1.0 - alpha[target])[:, None] * np.asarray(
                scene.horizon_colour, dtype=np.float32
            )
            alpha[target] = 1.0
            done[target] = True

        # 2. Crossing out of the removed wedge exposes an opaque cut face.
        face = was_removed & ~removed & ~inside
        if np.any(face):
            target = where[face]
            value = scene.table.sample(
                _physical(scene, radius[face]), points[face, 0] / radius[face]
            )
            normalized = symlog(
                value, scene.colour_scale, scene.linear_fraction, scene.noise_floor
            )
            base = signed_colour(normalized) + scene.slice_ambient
            colour[target] += (1.0 - alpha[target])[:, None] * base.astype(np.float32)
            alpha[target] = 1.0
            done[target] = True

        # 3. Everything still live accumulates emission from the volume.
        volume = ~removed & ~inside & ~face
        if np.any(volume):
            target = where[volume]
            local_radius = radius[volume]
            value = scene.table.sample(
                _physical(scene, local_radius), points[volume, 0] / local_radius
            )
            normalized = symlog(
                value, scene.colour_scale, scene.linear_fraction, scene.noise_floor
            )
            magnitude = np.abs(normalized)
            density = scene.opacity * magnitude**scene.opacity_gamma
            local_alpha = 1.0 - np.exp(-density * step[where][volume])
            emission = signed_colour(normalized).astype(np.float32)
            if boundary_band is not None:
                shell = np.abs(local_radius - scene.boundary_radius) < boundary_band
                if np.any(shell):
                    local_alpha = local_alpha.copy()
                    emission = emission.copy()
                    blend = scene.boundary_opacity
                    emission[shell] += np.asarray(
                        scene.boundary_colour, dtype=np.float32
                    ) * blend
                    local_alpha[shell] = np.maximum(local_alpha[shell], blend)
            transmit = (1.0 - alpha[target]) * local_alpha
            colour[target] += transmit[:, None] * emission
            alpha[target] += transmit

    image[index] = image[index] * (1.0 - alpha[:, None]) + colour
    image = image.reshape(height, width, 3)
    if supersample > 1:
        image = image.reshape(
            height // supersample, supersample, width // supersample, supersample, 3
        ).mean(axis=(1, 3))
    return np.clip(image, 0.0, 1.0)


def tone_map(image: np.ndarray, exposure: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Apply a gentle filmic roll-off so bright cores keep their hue."""

    scaled = image * exposure
    mapped = scaled / (1.0 + scaled)
    mapped = mapped / (1.0 / (1.0 + 1.0))
    if gamma != 1.0:
        mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / gamma)
    return np.clip(mapped, 0.0, 1.0)


@dataclass
class AngularShell:
    """One extraction sphere painted with the angular field it carries."""

    display_radius: float
    #: The field sampled on a uniform ``cos gamma`` grid in [-1, 1].
    profile: np.ndarray
    opacity: float = 0.62
    label: str = ""

    def sample(self, cosine: np.ndarray) -> np.ndarray:
        position = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0) * (self.profile.size - 1)
        index = np.clip(position.astype(np.int32), 0, self.profile.size - 2)
        frac = position - index
        return self.profile[index] * (1.0 - frac) + self.profile[index + 1] * frac


def render_shells(
    shells: list[AngularShell],
    camera: Camera,
    *,
    horizon_radius: float,
    colour_scale: float,
    linear_fraction: float = 0.05,
    noise_floor: float = 1e-5,
    cutaway: Callable[[np.ndarray], np.ndarray] | None = None,
    background: tuple[float, float, float] = (0.043, 0.055, 0.086),
    supersample: int = 2,
    chunk: int = 400_000,
) -> np.ndarray:
    """Render nested painted spheres with exact ray-sphere intersections.

    Each shell contributes two hits per ray.  All hits, plus the opaque black
    hole, are sorted by depth and composited front to back, so the nesting order
    is resolved geometrically rather than by drawing order.
    """

    origin, directions, (height, width) = camera.rays(supersample)
    image = np.empty((directions.shape[0], 3), dtype=np.float32)
    ordered = sorted(shells, key=lambda shell: shell.display_radius)

    for begin in range(0, directions.shape[0], chunk):
        block = directions[begin : begin + chunk]
        count = block.shape[0]
        n_events = 2 * len(ordered) + 1
        depth = np.full((count, n_events), np.inf, dtype=np.float64)
        colour = np.zeros((count, n_events, 3), dtype=np.float32)
        alpha = np.zeros((count, n_events), dtype=np.float32)

        for shell_index, shell in enumerate(ordered):
            hit, t_near, t_far = _sphere_span(origin, block, shell.display_radius)
            for side, parameter in enumerate((t_near, t_far)):
                slot = 2 * shell_index + side
                valid = hit & (parameter > 0.0)
                if not np.any(valid):
                    continue
                points = origin[None, :] + block * parameter[:, None]
                radius = np.sqrt(np.einsum("ij,ij->i", points, points))
                if cutaway is not None:
                    valid = valid & ~cutaway(points)
                if not np.any(valid):
                    continue
                value = shell.sample(points[:, 0] / np.maximum(radius, 1e-12))
                normalized = symlog(value, colour_scale, linear_fraction, noise_floor)
                depth[valid, slot] = parameter[valid]
                colour[valid, slot] = signed_colour(normalized)[valid]
                # A ray meeting the shell near its limb crosses more material
                # than one striking it head on.  Weighting the opacity by the
                # true grazing angle is what makes a translucent sphere read as
                # a curved surface rather than a flat disc.
                normal = points / np.maximum(radius[:, None], 1e-12)
                facing = np.abs(np.einsum("ij,ij->i", normal, block))
                grazing = np.clip(1.0 - facing, 0.0, 1.0)
                alpha[valid, slot] = shell.opacity * (0.45 + 0.55 * grazing[valid])

        hit, t_near, _ = _sphere_span(origin, block, horizon_radius)
        valid = hit & (t_near > 0.0)
        points = origin[None, :] + block * t_near[:, None]
        if cutaway is not None and np.any(valid):
            valid = valid & ~cutaway(points)
        depth[valid, -1] = t_near[valid]
        # A black sphere on a dark ground is invisible except where it happens
        # to occlude something bright, so the horizon carries a grazing-angle
        # rim that gives it an edge without lightening the disc.
        normal = points / np.maximum(
            np.linalg.norm(points, axis=1, keepdims=True), 1e-12
        )
        facing = np.abs(np.einsum("ij,ij->i", normal, block))
        rim = np.clip(1.0 - facing, 0.0, 1.0) ** 3
        colour[valid, -1] = (
            rim[valid, None] * np.asarray([0.42, 0.50, 0.68], np.float32)
        )
        alpha[valid, -1] = 1.0

        order = np.argsort(depth, axis=1)
        alpha = np.take_along_axis(alpha, order, axis=1)
        colour = np.take_along_axis(colour, order[:, :, None], axis=1)

        accumulated = np.zeros(count, dtype=np.float32)
        result = np.zeros((count, 3), dtype=np.float32)
        for slot in range(n_events):
            weight = (1.0 - accumulated) * alpha[:, slot]
            result += weight[:, None] * colour[:, slot]
            accumulated += weight
        image[begin : begin + chunk] = (
            result + (1.0 - accumulated)[:, None] * np.asarray(background, np.float32)
        )

    image = image.reshape(height, width, 3)
    if supersample > 1:
        image = image.reshape(
            height // supersample, supersample, width // supersample, supersample, 3
        ).mean(axis=(1, 3))
    return np.clip(image, 0.0, 1.0)


def wedge_cutaway(
    first_normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
    second_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Callable[[np.ndarray], np.ndarray]:
    """Remove the material on the positive side of two planes through the origin.

    The exposed faces are half-planes that both contain the source axis, so the
    cut always reveals a meridional section of the field.
    """

    first = np.asarray(first_normal, dtype=float)
    second = np.asarray(second_normal, dtype=float)

    def predicate(points: np.ndarray) -> np.ndarray:
        return (points @ first > 0.0) & (points @ second > 0.0)

    return predicate
