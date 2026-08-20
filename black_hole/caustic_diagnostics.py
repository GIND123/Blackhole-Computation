"""Diagnostic sections through the antipodal caustic.

The rendered candidates showed a caustic without ever establishing that the
caustic is resolved, so these diagnostics work in physical areal radius on a
symmetric linear scale and measure the focus rather than illustrating it.
Three questions are answered here:

* where the wavefront actually is, as a ridge traced from the data;
* when the field is most strongly focused on the antipodal axis, measured
  from the spatial field and not from an outer observer time series;
* what sets the width of the focus, separating the angular truncation of the
  evolution from the finite angular width of the emitter.

The convention throughout is that ``Phi = u / r`` is the scalar field, the
emitter sits at ``gamma = 0`` and the antipodal axis is ``gamma = pi``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import special

from .localized_source import angular_spectral_weights
from .source_evolution import SourcedSimulationResult, load_sourced_result


HORIZON_RADIUS = 2.0
PHOTON_SPHERE_RADIUS = 3.0
DEFAULT_OUTER_RADIUS = 20.0
# The axial band used to define focusing.  It is a band rather than the exact
# axis so the measurement survives the finite angular sampling.
AXIAL_HALF_WIDTH_DEGREES = 1.0
# Focusing is measured in the wave zone.  Inside this radius the field is
# dominated by the strong field pile up around the photon sphere, which is not
# a caustic: it has no interior maximum along the axis and it would otherwise
# capture the search simply by being large.
WAVE_ZONE_RADIUS = 4.0


@dataclass(frozen=True)
class EquatorialSection:
    """One equatorial snapshot in physical areal radius."""

    bridge_time: float
    radius: np.ndarray
    angle: np.ndarray
    field: np.ndarray

    def cartesian(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.outer(self.radius, np.cos(self.angle)),
            np.outer(self.radius, np.sin(self.angle)),
        )


def _legendre_coefficients(
    result: SourcedSimulationResult, cosine: np.ndarray, ell_max: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the retained ell indices and their angular reconstruction weights."""

    ells = np.asarray(result.response_ell, dtype=int)
    if ell_max is not None:
        ells = ells[ells <= int(ell_max)]
    if ells.size == 0:
        raise ValueError("No angular responses survive the requested truncation.")
    concentration = float(result.metadata["source"]["angular_concentration"])
    weights = angular_spectral_weights(concentration, int(ells[-1]))
    legendre = np.stack([special.eval_legendre(int(ell), cosine) for ell in ells])
    coefficients = (
        weights[ells] * (2.0 * ells + 1.0) / (4.0 * np.pi)
    )[:, None] * legendre
    return ells, coefficients


def equatorial_section(
    result: SourcedSimulationResult,
    snapshot_index: int,
    *,
    outer_radius: float = DEFAULT_OUTER_RADIUS,
    angles: int = 1441,
    ell_max: int | None = None,
) -> EquatorialSection:
    """Return ``Phi`` on the equatorial plane out to ``outer_radius``.

    The emitter lies in this plane, so the section carries the full caustic
    structure and the angle from the emitter is the azimuth itself.
    """

    radius = np.asarray(result.snapshot_areal_radius, dtype=float)
    keep = (radius >= HORIZON_RADIUS) & (radius <= outer_radius)
    if not keep.any():
        raise ValueError("No archived radii lie inside the requested crop.")
    angle = np.linspace(0.0, np.pi, int(angles))
    ells, coefficients = _legendre_coefficients(result, np.cos(angle), ell_max)
    lookup = {int(value): index for index, value in enumerate(result.response_ell)}
    rows = np.asarray([lookup[int(ell)] for ell in ells])
    snapshot = np.asarray(result.response_snapshots[snapshot_index], dtype=float)
    reduced = np.einsum("lr,lp->rp", snapshot[rows][:, keep], coefficients)
    return EquatorialSection(
        bridge_time=float(result.snapshot_times[snapshot_index]),
        radius=radius[keep],
        angle=angle,
        field=reduced / radius[keep][:, None],
    )


def axial_focus(
    section: EquatorialSection, *, wave_zone_radius: float = WAVE_ZONE_RADIUS
) -> dict:
    """Return the strongest focus on the antipodal axis of one section.

    A focus is accepted only when the axial profile has an interior maximum
    inside the wave zone.  A maximum sitting on either edge of the band means
    the converging wavefront has not arrived yet or has already passed through,
    and the value there describes the band edge rather than a caustic.
    """

    axial = section.angle >= np.pi - np.deg2rad(AXIAL_HALF_WIDTH_DEGREES)
    off = (section.angle > np.deg2rad(60.0)) & (section.angle < np.deg2rad(120.0))
    profile = np.abs(section.field[:, axial]).max(axis=1)
    transverse = np.abs(section.field[:, off]).mean(axis=1)

    wave = section.radius >= wave_zone_radius
    radius = section.radius[wave]
    banded = profile[wave]
    index = int(np.argmax(banded))
    interior = 0 < index < banded.size - 1
    reference = float(transverse[wave][index])
    amplification = (
        float(banded[index] / reference) if reference > 0.0 else float("nan")
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(transverse[wave] > 0.0, banded / transverse[wave], 0.0)
    peak = int(np.argmax(ratio))
    return {
        "bridge_time": section.bridge_time,
        "radius_over_M": float(radius[index]),
        "peak_abs_field": float(banded[index]),
        "transverse_reference": reference,
        "amplification": amplification,
        "interior_maximum": bool(interior),
        "amplification_peak": float(ratio[peak]),
        "amplification_peak_radius_over_M": float(radius[peak]),
        "radial_index": int(np.flatnonzero(wave)[index]),
    }


def locate_focus(result: SourcedSimulationResult, **kwargs) -> dict:
    """Return the snapshot of strongest axial focusing.

    Selection uses the spatial field on the antipodal axis.  An outer observer
    time series peaks later, when the focused wavefront reaches that observer,
    and is therefore the wrong clock for choosing a snapshot.
    """

    best: dict | None = None
    for index in range(len(result.snapshot_times)):
        section = equatorial_section(result, index, **kwargs)
        record = axial_focus(section)
        record["snapshot_index"] = index
        if not record["interior_maximum"]:
            continue
        if best is None or record["amplification"] > best["amplification"]:
            best = record
    if best is None:
        raise ValueError(
            "No snapshot shows an interior axial maximum in the wave zone."
        )
    return best


def wavefront_ridge(section: EquatorialSection, *, minimum_radius: float = 2.2) -> dict:
    """Return the per-angle radial maximum of ``|Phi|``.

    This traces where the wavefront is; it does not attempt to identify a
    mathematical cusp.  The point of the trace that reaches the antipodal axis
    is reported separately as the axial focus.
    """

    usable = section.radius >= minimum_radius
    field = np.abs(section.field[usable])
    radius = section.radius[usable]
    rows = np.argmax(field, axis=0)
    return {
        "angle": section.angle,
        "radius": radius[rows],
        "value": field[rows, np.arange(field.shape[1])],
    }


def transverse_profile(
    result: SourcedSimulationResult,
    snapshot_index: int,
    radius_over_M: float,
    *,
    ell_max: int | None = None,
    angles: int = 4001,
    span_degrees: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Phi`` across the antipodal axis at one areal radius."""

    radii = np.asarray(result.snapshot_areal_radius, dtype=float)
    index = int(np.argmin(np.abs(radii - radius_over_M)))
    angle = np.linspace(np.pi - np.deg2rad(span_degrees), np.pi, int(angles))
    ells, coefficients = _legendre_coefficients(result, np.cos(angle), ell_max)
    lookup = {
        int(value): position for position, value in enumerate(result.response_ell)
    }
    rows = np.asarray([lookup[int(ell)] for ell in ells])
    snapshot = np.asarray(result.response_snapshots[snapshot_index], dtype=float)
    reduced = np.einsum("l,lp->p", snapshot[rows, index], coefficients)
    return angle, reduced / radii[index]


def full_width_half_maximum(angle: np.ndarray, profile: np.ndarray) -> float:
    """Return the angular full width at half maximum in degrees."""

    magnitude = np.abs(profile)
    peak = float(magnitude.max())
    if peak <= 0.0:
        return float("nan")
    above = np.flatnonzero(magnitude >= 0.5 * peak)
    if above.size < 2:
        return float("nan")
    # The profile is sampled on half of a symmetric feature centred on the
    # axis, so the measured half width is doubled.
    return float(2.0 * np.rad2deg(angle[-1] - angle[above[0]]))


def truncation_study(
    result: SourcedSimulationResult,
    snapshot_index: int,
    radius_over_M: float,
    truncations: tuple[int, ...] = (20, 30, 40, 50),
) -> list[dict]:
    """Return the focus width and peak against the angular truncation.

    A focus that does not move as ``ell_max`` is lowered is limited by the
    angular width of the emitter rather than by the truncation, which decides
    whether a sharper caustic needs a narrower source or a longer sum.
    """

    available = int(np.asarray(result.response_ell, dtype=int).max())
    rows = []
    for ell_max in truncations:
        if ell_max > available:
            continue
        angle, profile = transverse_profile(
            result, snapshot_index, radius_over_M, ell_max=ell_max
        )
        rows.append(
            {
                "ell_max": int(ell_max),
                "peak_abs_field": float(np.abs(profile).max()),
                "fwhm_degrees": full_width_half_maximum(angle, profile),
            }
        )
    if rows:
        finest = rows[-1]
        for row in rows:
            row["relative_peak_change"] = (
                abs(row["peak_abs_field"] - finest["peak_abs_field"])
                / finest["peak_abs_field"]
            )
    return rows


def load(path: str | Path) -> SourcedSimulationResult:
    """Return one archived snapshot case."""

    return load_sourced_result(Path(path))
