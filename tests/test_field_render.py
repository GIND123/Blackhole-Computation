"""Unit checks for the ray marching renderer used by the figure suite."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import special

from black_hole.field_render import (
    AngularShell,
    Camera,
    FieldTable,
    Scene,
    modal_field_table,
    power_display,
    render,
    render_shells,
    signed_colour,
    symlog,
    wedge_cutaway,
)
from black_hole.localized_source import angular_spectral_weights


CONCENTRATION = 64.0


def _synthetic(ell_max: int = 12, n_radius: int = 64):
    generator = np.random.default_rng(20260818)
    ell = np.arange(ell_max + 1)
    radius = np.linspace(2.0, 10.0, n_radius)
    response = generator.normal(size=(ell.size, n_radius))
    return ell, radius, response


class ModalTableTest(unittest.TestCase):
    def test_table_matches_the_direct_legendre_sum(self) -> None:
        """The tabulated volume equals the reconstruction it stands in for."""

        ell, radius, response = _synthetic()
        table = modal_field_table(
            response, ell, radius, CONCENTRATION,
            n_radius=radius.size, n_angle=97, radius_max=float(radius[-1]),
        )
        weights = angular_spectral_weights(CONCENTRATION, int(ell[-1]))
        cosine = np.linspace(-1.0, 1.0, 97)
        expected = np.zeros((radius.size, cosine.size))
        for index, order in enumerate(ell):
            expected += np.outer(
                response[index],
                weights[order] * (2.0 * order + 1.0) / (4.0 * np.pi)
                * special.eval_legendre(int(order), cosine),
            )
        np.testing.assert_allclose(table.values, expected, rtol=0, atol=1e-5)

    def test_spectral_filter_only_attenuates_high_orders(self) -> None:
        ell, radius, response = _synthetic()
        plain = modal_field_table(
            response, ell, radius, CONCENTRATION, n_radius=radius.size,
            n_angle=65, radius_max=float(radius[-1]),
        )
        filtered = modal_field_table(
            response, ell, radius, CONCENTRATION, n_radius=radius.size,
            n_angle=65, radius_max=float(radius[-1]), spectral_filter=8.0,
        )
        self.assertLess(
            float(np.abs(filtered.values).max()),
            float(np.abs(plain.values).max()) * 1.001,
        )
        self.assertGreater(float(np.abs(filtered.values).max()), 0.0)


class SamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
        self.table = FieldTable(values=values, radius_min=2.0, radius_max=6.0)

    def test_sampling_is_exact_at_grid_nodes(self) -> None:
        radii = np.linspace(2.0, 6.0, 5)
        cosines = np.linspace(-1.0, 1.0, 7)
        for i, r in enumerate(radii):
            for j, c in enumerate(cosines):
                self.assertAlmostEqual(
                    float(self.table.sample(np.array([r]), np.array([c]))[0]),
                    float(self.table.values[i, j]),
                    places=4,
                )

    def test_sampling_is_bilinear_between_nodes(self) -> None:
        # Node (i, j) holds 7*i + j; r=3.5 sits midway between rows 1 and 2.
        midpoint = self.table.sample(np.array([3.5]), np.array([0.0]))[0]
        self.assertAlmostEqual(float(midpoint), 13.5, places=4)

    def test_sampling_vanishes_outside_the_radial_range(self) -> None:
        outside = self.table.sample(np.array([1.0, 9.0]), np.array([0.0, 0.0]))
        np.testing.assert_array_equal(outside, np.zeros(2, dtype=np.float32))


class TransferTest(unittest.TestCase):
    def test_symlog_is_odd_and_saturates_at_the_scale(self) -> None:
        values = np.array([-1.0, -0.1, 0.0, 0.1, 1.0])
        mapped = symlog(values, 1.0, 0.05)
        np.testing.assert_allclose(mapped, -mapped[::-1], atol=1e-12)
        self.assertAlmostEqual(float(mapped[-1]), 1.0, places=12)
        self.assertAlmostEqual(float(mapped[2]), 0.0, places=12)

    def test_symlog_is_monotone_in_magnitude(self) -> None:
        magnitudes = np.linspace(0.0, 1.0, 64)
        mapped = symlog(magnitudes, 1.0, 0.05)
        self.assertTrue(np.all(np.diff(mapped) > 0.0))

    def test_noise_floor_zeroes_values_below_the_threshold(self) -> None:
        values = np.array([1e-9, 1e-3, 1.0])
        mapped = symlog(values, 1.0, 0.05, noise_floor=1e-5)
        self.assertEqual(float(mapped[0]), 0.0)
        self.assertGreater(float(mapped[1]), 0.0)

    def test_both_ramps_agree_at_zero(self) -> None:
        """Otherwise the undisturbed region takes the colour of a sign."""

        negative = signed_colour(np.array([-1e-18]))
        positive = signed_colour(np.array([1e-18]))
        np.testing.assert_allclose(negative, positive, atol=1e-12)


class GeometryTest(unittest.TestCase):
    def test_camera_projects_its_target_to_the_centre(self) -> None:
        camera = Camera(position=(0.0, 10.0, 0.0), target=(0.0, 0.0, 0.0),
                        width=64, height=48)
        centre = camera.project(np.array([0.0, 0.0, 0.0]))[0]
        np.testing.assert_allclose(centre, [0.5, 0.5], atol=1e-12)

    def test_projection_agrees_with_the_rays_that_are_marched(self) -> None:
        """Annotations are only anchored correctly if the two agree."""

        camera = Camera(position=(3.0, 9.0, 4.0), width=81, height=61)
        origin, directions, (height, width) = camera.rays(1)
        for row, column in ((0, 0), (30, 40), (60, 80)):
            point = origin + directions[row * width + column] * 7.5
            fraction = camera.project(point)[0]
            self.assertAlmostEqual(
                float(fraction[0]) * width - 0.5, column, places=6
            )
            self.assertAlmostEqual(
                float(fraction[1]) * height - 0.5, row, places=6
            )

    def test_wedge_removes_exactly_one_quadrant(self) -> None:
        predicate = wedge_cutaway()
        points = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        )
        np.testing.assert_array_equal(predicate(points), [True, False, False, False])

    def test_power_display_inverts_the_radial_map(self) -> None:
        to_physical = power_display(0.5)
        radii = np.array([2.0, 17.0, 79.0])
        np.testing.assert_allclose(to_physical(radii**0.5), radii, rtol=1e-12)


class RenderTest(unittest.TestCase):
    def _scene(self, values: np.ndarray) -> Scene:
        return Scene(
            table=FieldTable(values=values, radius_min=2.0, radius_max=20.0),
            horizon_radius=2.0,
            outer_radius=20.0,
            colour_scale=1.0,
            steps=48,
        )

    def test_a_vanishing_field_emits_nothing(self) -> None:
        """Only the opaque horizon may differ from the background."""

        scene = self._scene(np.zeros((16, 16), dtype=np.float32))
        camera = Camera(position=(0.0, 70.0, 0.0), width=32, height=24)
        image = render(scene, camera, supersample=1)
        background = np.asarray(scene.background, dtype=np.float32)
        np.testing.assert_allclose(image[0, 0], background, atol=1e-6)
        self.assertLessEqual(float(image.max()), float(background.max()) + 1e-6)

    def test_the_horizon_occludes_the_field_behind_it(self) -> None:
        values = np.full((16, 16), 0.5, dtype=np.float32)
        camera = Camera(position=(0.0, 70.0, 0.0), width=41, height=41)
        # The volume has to stay optically thin, or a central ray saturates
        # before it reaches the horizon and occlusion changes nothing.
        opaque = self._scene(values)
        opaque.opacity = 0.01
        with_horizon = render(opaque, camera, supersample=1)
        without = self._scene(values)
        without.opacity = 0.01
        without.horizon_radius = 1e-3
        bare = render(without, camera, supersample=1)
        # A central ray keeps the emission in front of the horizon and loses
        # everything behind it, so removing the horizon can only brighten it.
        self.assertLess(float(with_horizon[20, 20].sum()), float(bare[20, 20].sum()))
        # A ray that misses the horizon entirely is unaffected by it.
        np.testing.assert_allclose(with_horizon[20, 1], bare[20, 1], atol=1e-6)

    def test_output_stays_inside_the_unit_range(self) -> None:
        generator = np.random.default_rng(7)
        values = generator.normal(size=(24, 24)).astype(np.float32)
        image = render(self._scene(values),
                       Camera(position=(10.0, 60.0, 20.0), width=32, height=32),
                       supersample=2)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)


class ShellRenderTest(unittest.TestCase):
    def test_nesting_is_resolved_by_depth_not_by_drawing_order(self) -> None:
        profile = np.full(32, 1.0)
        inner = AngularShell(display_radius=4.0, profile=profile, opacity=0.7)
        outer = AngularShell(display_radius=9.0, profile=-profile, opacity=0.7)
        camera = Camera(position=(0.0, 60.0, 0.0), width=24, height=24)
        forward = render_shells([inner, outer], camera, horizon_radius=1.0,
                                colour_scale=1.0, supersample=1)
        reversed_order = render_shells([outer, inner], camera, horizon_radius=1.0,
                                       colour_scale=1.0, supersample=1)
        np.testing.assert_allclose(forward, reversed_order, atol=1e-6)

    def test_an_inner_shell_changes_the_image_it_appears_in(self) -> None:
        profile = np.full(32, 1.0)
        inner = AngularShell(display_radius=4.0, profile=profile, opacity=0.7)
        outer = AngularShell(display_radius=9.0, profile=-profile, opacity=0.7)
        camera = Camera(position=(0.0, 60.0, 0.0), width=24, height=24)
        both = render_shells([inner, outer], camera, horizon_radius=1.0,
                             colour_scale=1.0, supersample=1)
        alone = render_shells([outer], camera, horizon_radius=1.0,
                              colour_scale=1.0, supersample=1)
        self.assertFalse(np.allclose(both, alone, atol=1e-6))

    def test_shells_are_removed_inside_the_cutaway(self) -> None:
        profile = np.full(32, 1.0)
        shell = AngularShell(display_radius=6.0, profile=profile, opacity=1.0)
        camera = Camera(position=(0.0, 60.0, 0.0), width=24, height=24)
        full = render_shells([shell], camera, horizon_radius=1.0,
                             colour_scale=1.0, supersample=1)
        cut = render_shells([shell], camera, horizon_radius=1.0,
                            colour_scale=1.0, cutaway=wedge_cutaway(),
                            supersample=1)
        self.assertFalse(np.allclose(full, cut))


if __name__ == "__main__":
    unittest.main()
