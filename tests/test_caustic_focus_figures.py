"""The focus alternatives must not saturate or mislocate the peak."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole import caustic_focus_figures as figures


class TransferFunctionTests(unittest.TestCase):
    """The display range has to contain the data, whatever the data is."""

    def test_limits_contain_every_sample(self) -> None:
        rng = np.random.default_rng(11)
        for scale in (1.0, 1e-3, 1e3):
            field = scale * rng.normal(size=4096)
            field[7] = -12.0 * scale
            norm, record = figures._unclipped_norm(field)
            self.assertLessEqual(float(np.abs(field).max()), float(norm.vmax))
            self.assertGreaterEqual(float(field.min()), float(norm.vmin))
            self.assertFalse(record["peak_is_clipped"])
            self.assertEqual(record["colour_limit"], float(np.abs(field).max()))

    def test_the_peak_lands_on_the_end_of_the_colour_bar(self) -> None:
        field = np.array([-5.0, -1.0, 0.0, 0.25, 3.0])
        norm, _ = figures._unclipped_norm(field)
        self.assertAlmostEqual(float(norm(field.min())), 0.0, places=12)
        self.assertAlmostEqual(float(norm(-field.min())), 1.0, places=12)

    def test_the_stretch_is_monotone_and_signed(self) -> None:
        field = np.concatenate(
            [np.linspace(-2e-2, 2e-2, 501), [1e-9, -1e-9]]
        )
        norm, _ = figures._unclipped_norm(field)
        ordered = np.sort(field)
        mapped = np.asarray(norm(ordered))
        self.assertTrue(np.all(np.diff(mapped) >= -1e-12))
        # Zero sits at the centre of a signed scale, so the neutral colour is
        # the zero of the field rather than an arbitrary offset.
        self.assertAlmostEqual(float(norm(np.array([0.0]))[0]), 0.5, places=12)

    def test_the_linear_width_ignores_the_undisturbed_region(self) -> None:
        disturbed = np.full(200, 1.0)
        quiet = np.full(20000, 1e-6)
        width = figures._linear_width(np.concatenate([disturbed, quiet]))
        self.assertAlmostEqual(width, 1.0, places=12)


@unittest.skipUnless(
    figures.NARROW_ARCHIVE.exists(), "narrow-source archive is not present"
)
class ExtremumSelectionTests(unittest.TestCase):
    """A raw argmax is not the focus the diagnostics report."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = figures.load(figures.NARROW_ARCHIVE)
        archived = np.asarray(cls.result.snapshot_times, dtype=float)
        cls.indices = [
            int(index)
            for index in np.flatnonzero((archived >= 46.0) & (archived <= 50.0))
        ]
        _, cls.radius, cls.profiles = figures._axial_profiles(
            cls.result, cls.indices, angles=361
        )
        inside = cls.radius >= figures.SURFACE_INNER_RADIUS
        cls.radius = cls.radius[inside]
        cls.profiles = cls.profiles[:, inside]

    def test_the_selection_returns_the_reported_focus(self) -> None:
        row, column, focus = figures._selected_focus(
            self.result, self.indices, self.radius, angles=361
        )
        self.assertTrue(focus["interior_maximum"])
        self.assertAlmostEqual(focus["bridge_time"], 48.0, places=6)
        self.assertAlmostEqual(self.radius[column], 6.2525, places=3)
        self.assertLess(abs(self.profiles[row, column]) - 2.0818e-2, 1e-5)

    def test_the_raw_wave_zone_argmax_is_a_different_and_rejected_point(self) -> None:
        wave = self.radius >= figures.WAVE_ZONE_RADIUS
        banded = np.abs(self.profiles[:, wave])
        raw_row, raw_column = np.unravel_index(
            int(np.argmax(banded)), banded.shape
        )
        raw_radius = float(self.radius[wave][int(raw_column)])
        # The raw extremum sits on the inner edge of the wave zone, which is
        # exactly the configuration axial_focus rejects.
        self.assertAlmostEqual(raw_radius, figures.WAVE_ZONE_RADIUS, places=2)
        _, column, _ = figures._selected_focus(
            self.result, self.indices, self.radius, angles=361
        )
        self.assertNotEqual(int(raw_column), int(column))
        self.assertGreater(float(self.radius[column]), raw_radius)


if __name__ == "__main__":
    unittest.main()
