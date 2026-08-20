"""Tests for the caustic diagnostic measurements.

The estimators are driven with analytic fields whose focus position, width,
and angular content are known in advance, so a wrong answer is a failed test
rather than a plausible looking picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from black_hole import caustic_diagnostics as diagnostics


@dataclass
class _Archive:
    """A minimal stand in for one archived snapshot case."""

    snapshot_areal_radius: np.ndarray
    snapshot_times: np.ndarray
    response_ell: np.ndarray
    response_snapshots: np.ndarray
    metadata: dict


def _archive_with_single_ell(ell: int, radii: np.ndarray, amplitude: float = 1.0):
    """Return an archive carrying exactly one angular response."""

    ells = np.arange(0, ell + 1)
    snapshots = np.zeros((1, ells.size, radii.size))
    snapshots[0, ell, :] = amplitude
    return _Archive(
        snapshot_areal_radius=radii,
        snapshot_times=np.asarray([10.0]),
        response_ell=ells,
        response_snapshots=snapshots,
        metadata={"source": {"angular_concentration": 64.0}},
    )


class TestSectionGeometry:
    def test_section_is_cropped_to_the_requested_radius(self) -> None:
        radii = np.linspace(1.0, 40.0, 400)
        archive = _archive_with_single_ell(2, radii)
        section = diagnostics.equatorial_section(archive, 0, outer_radius=20.0)
        assert section.radius.min() >= diagnostics.HORIZON_RADIUS
        assert section.radius.max() <= 20.0

    def test_cartesian_grid_matches_the_polar_sampling(self) -> None:
        radii = np.linspace(2.0, 20.0, 64)
        archive = _archive_with_single_ell(2, radii)
        section = diagnostics.equatorial_section(archive, 0, angles=181)
        x, y = section.cartesian()
        assert x.shape == section.field.shape
        radius = np.hypot(x, y)
        assert np.allclose(radius[:, 0], section.radius)

    def test_a_single_multipole_reproduces_its_legendre_shape(self) -> None:
        """The reconstruction must be the addition theorem, not a fit."""

        radii = np.linspace(2.0, 20.0, 32)
        ell = 6
        archive = _archive_with_single_ell(ell, radii)
        section = diagnostics.equatorial_section(archive, 0, angles=361)
        from scipy import special

        from black_hole.localized_source import angular_spectral_weights

        weights = angular_spectral_weights(64.0, ell)
        expected = (
            weights[ell]
            * (2.0 * ell + 1.0)
            / (4.0 * np.pi)
            * special.eval_legendre(ell, np.cos(section.angle))
        )
        assert np.allclose(section.field[0], expected / section.radius[0])


class TestFocusSelection:
    def _focused_archive(self, focus_radius: float, width: float = 0.6):
        """Return an archive whose field is a ring peaked on the antipodal axis."""

        radii = np.linspace(2.0, 20.0, 361)
        angle = np.linspace(0.0, np.pi, 721)
        radial = np.exp(-((radii - focus_radius) ** 2) / (2.0 * width**2))
        angular = np.exp(-((np.pi - angle) ** 2) / (2.0 * np.deg2rad(6.0) ** 2))
        field = np.outer(radial, angular) * radii[:, None]
        section = diagnostics.EquatorialSection(
            bridge_time=1.0, radius=radii, angle=angle, field=field / radii[:, None]
        )
        return section

    def test_interior_focus_is_found_at_the_planted_radius(self) -> None:
        section = self._focused_archive(9.0)
        record = diagnostics.axial_focus(section)
        assert record["interior_maximum"]
        assert record["radius_over_M"] == pytest.approx(9.0, abs=0.1)
        assert record["amplification"] > 10.0

    def test_a_focus_on_the_band_edge_is_rejected(self) -> None:
        """A maximum at the wave zone edge is not a caustic."""

        section = self._focused_archive(20.0)
        record = diagnostics.axial_focus(section)
        assert not record["interior_maximum"]

    def test_strong_field_pile_up_does_not_count_as_a_focus(self) -> None:
        """A monotonic profile rising toward the hole has no interior maximum."""

        radii = np.linspace(2.0, 20.0, 361)
        angle = np.linspace(0.0, np.pi, 721)
        field = np.outer(radii**-3.0, np.ones_like(angle))
        section = diagnostics.EquatorialSection(
            bridge_time=1.0, radius=radii, angle=angle, field=field
        )
        record = diagnostics.axial_focus(section)
        assert not record["interior_maximum"]


class TestWidthMeasurement:
    def test_full_width_half_maximum_recovers_a_known_gaussian(self) -> None:
        angle = np.linspace(np.pi - np.deg2rad(60.0), np.pi, 4001)
        sigma = np.deg2rad(5.0)
        profile = np.exp(-((np.pi - angle) ** 2) / (2.0 * sigma**2))
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.rad2deg(sigma)
        measured = diagnostics.full_width_half_maximum(angle, profile)
        assert measured == pytest.approx(expected, rel=2.0e-3)

    def test_width_of_a_flat_profile_is_not_a_number(self) -> None:
        angle = np.linspace(np.pi - np.deg2rad(60.0), np.pi, 101)
        assert np.isnan(
            diagnostics.full_width_half_maximum(angle, np.zeros_like(angle))
        )


class TestTruncation:
    def test_truncation_is_reported_against_the_finest_available_sum(self) -> None:
        radii = np.linspace(2.0, 20.0, 64)
        archive = _archive_with_single_ell(30, radii)
        rows = diagnostics.truncation_study(
            archive, 0, 8.0, truncations=(10, 20, 30)
        )
        assert [row["ell_max"] for row in rows] == [10, 20, 30]
        assert rows[-1]["relative_peak_change"] == 0.0

    def test_truncations_above_the_archive_are_skipped(self) -> None:
        radii = np.linspace(2.0, 20.0, 64)
        archive = _archive_with_single_ell(12, radii)
        rows = diagnostics.truncation_study(
            archive, 0, 8.0, truncations=(10, 12, 40)
        )
        assert [row["ell_max"] for row in rows] == [10, 12]
