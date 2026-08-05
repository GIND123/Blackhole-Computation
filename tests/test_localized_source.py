"""Tests for the localized emitter and its exact angular decomposition."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.localized_source import (
    LocalizedSourceParameters,
    angular_profile,
    angular_spectral_weights,
    build_mode_catalogue,
    compact_bump,
    minimum_ell_max,
    radial_profile,
    retained_angular_fraction,
    source_normalizations,
    time_profile,
    verify_angular_expansion,
    weak_source_integral,
)
from black_hole.three_d_solver import real_spherical_harmonic


def _sphere_quadrature(points: int = 200):
    cosine, weights = np.polynomial.legendre.leggauss(points)
    phi = 2.0 * np.pi * np.arange(2 * points) / (2 * points)
    theta = np.arccos(cosine)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    area = np.broadcast_to(
        weights[:, None] * (2.0 * np.pi / (2 * points)), theta_grid.shape
    )
    return theta_grid, phi_grid, area


class CompactProfileTests(unittest.TestCase):
    def test_bump_is_supported_only_inside_the_unit_interval(self) -> None:
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
        values = compact_bump(x)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 0.0)
        self.assertEqual(values[5], 0.0)
        self.assertEqual(values[6], 0.0)
        self.assertAlmostEqual(values[3], 1.0, places=15)
        self.assertGreater(values[2], 0.0)

    def test_bump_derivatives_vanish_at_the_support_boundary(self) -> None:
        x = np.linspace(0.9, 1.0, 400)
        values = compact_bump(x)
        slope = np.gradient(values, x)
        self.assertLess(abs(values[-1]), 1e-300)
        self.assertLess(abs(slope[-1]), 1e-6)

    def test_factors_respect_the_declared_support(self) -> None:
        source = LocalizedSourceParameters()
        left, right = source.radial_support
        self.assertEqual(radial_profile(np.array([left, right]), source).tolist(), [0.0, 0.0])
        start, end = source.killing_time_support
        self.assertEqual(time_profile(np.array([start, end]), source).tolist(), [0.0, 0.0])
        temporal_normalization, radial_normalization = source_normalizations(source)
        self.assertAlmostEqual(
            float(radial_profile(np.asarray(source.center_radius), source)),
            radial_normalization,
        )
        self.assertAlmostEqual(
            float(time_profile(np.asarray(source.time_center), source)),
            temporal_normalization,
        )

    def test_temporal_and_radial_factors_have_covariant_unit_integrals(self) -> None:
        source = LocalizedSourceParameters()
        time = np.linspace(*source.killing_time_support, 20001)
        radius = np.linspace(*source.radial_support, 20001)
        self.assertAlmostEqual(float(np.trapezoid(time_profile(time, source), time)), 1.0, places=10)
        self.assertAlmostEqual(
            float(np.trapezoid(radial_profile(radius, source) * radius**2, radius)),
            1.0,
            places=10,
        )

    def test_legacy_peak_normalization_remains_available_for_old_archives(self) -> None:
        source = LocalizedSourceParameters(normalization="peak")
        self.assertAlmostEqual(
            float(time_profile(np.asarray(source.time_center), source)), 1.0
        )
        self.assertAlmostEqual(
            float(radial_profile(np.asarray(source.center_radius), source)), 1.0
        )
        self.assertNotAlmostEqual(source.as_dict()["covariant_integral"], 1.0)

    def test_weak_source_family_converges_to_fixed_strength_delta(self) -> None:
        widths = (1.0, 0.5, 0.25)
        errors = []
        for scale in widths:
            source = LocalizedSourceParameters(
                amplitude=2.5,
                radial_half_width=1.5 * scale,
                time_half_width=4.0 * scale,
                angular_concentration=16.0 / scale**2,
            )
            measured = weak_source_integral(
                source,
                lambda time, radius, cosine: (
                    1.0 + 0.01 * time**2 + 0.02 * radius**2 + 0.03 * cosine
                ),
                quadrature_order=64,
            )
            exact = source.amplitude * (
                1.0
                + 0.01 * source.time_center**2
                + 0.02 * source.center_radius**2
                + 0.03
            )
            errors.append(abs(measured - exact))
        self.assertTrue(np.all(np.diff(errors) < 0.0))
        self.assertLess(errors[-1], 0.3 * errors[0])


class AngularDecompositionTests(unittest.TestCase):
    def test_profile_integrates_to_one_over_the_sphere(self) -> None:
        source = LocalizedSourceParameters()
        theta, phi, area = _sphere_quadrature()
        direction = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )
        emitter = np.array(
            [
                np.sin(source.source_theta) * np.cos(source.source_phi),
                np.sin(source.source_theta) * np.sin(source.source_phi),
                np.cos(source.source_theta),
            ]
        )
        cosine = np.einsum("i...,i->...", direction, emitter)
        total = float(np.sum(angular_profile(cosine, source) * area))
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_closed_form_weights_match_gauss_legendre_projection(self) -> None:
        source = LocalizedSourceParameters()
        report = verify_angular_expansion(source, ell_max=20)
        self.assertLess(report["maximum_weight_error"], 1e-11)
        self.assertLess(report["maximum_relative_reconstruction_error"], 1e-4)
        self.assertGreater(report["retained_angular_fraction"], 1.0 - 1e-9)

    def test_weights_decrease_and_start_at_one(self) -> None:
        weights = angular_spectral_weights(16.0, 24)
        self.assertAlmostEqual(weights[0], 1.0, places=15)
        self.assertTrue(np.all(np.diff(weights) < 0.0))
        self.assertGreater(weights[-1], 0.0)

    def test_narrower_emitter_needs_more_harmonics(self) -> None:
        broad = retained_angular_fraction(16.0, 8)
        narrow = retained_angular_fraction(49.0, 8)
        self.assertGreater(broad, narrow)

    def test_automatic_cutoff_meets_omitted_power_target(self) -> None:
        cutoff = minimum_ell_max(49.0, omitted_power_tolerance=1e-10)
        self.assertLessEqual(1.0 - retained_angular_fraction(49.0, cutoff), 1e-10)
        if cutoff:
            self.assertGreater(1.0 - retained_angular_fraction(49.0, cutoff - 1), 1e-10)


class ModeCatalogueTests(unittest.TestCase):
    def test_equatorial_emitter_obeys_the_reflection_selection_rule(self) -> None:
        catalogue = build_mode_catalogue(LocalizedSourceParameters(), 10)
        self.assertTrue(np.all(catalogue.m >= 0))
        self.assertTrue(np.all((catalogue.ell + catalogue.m) % 2 == 0))
        self.assertLess(
            catalogue.discarded_maximum_amplitude,
            1e-14 * float(np.max(np.abs(catalogue.amplitude))),
        )

    def test_catalogue_reconstructs_the_angular_profile(self) -> None:
        source = LocalizedSourceParameters()
        catalogue = build_mode_catalogue(source, 20)
        theta = np.array([0.3, 1.2, np.pi / 2, 2.7])
        phi = np.array([0.0, 1.1, 3.9, 5.5])
        reconstruction = sum(
            amplitude * real_spherical_harmonic(int(ell), int(m), theta, phi)
            for ell, m, amplitude in zip(
                catalogue.ell, catalogue.m, catalogue.amplitude
            )
        )
        emitter = np.array([1.0, 0.0, 0.0])
        direction = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )
        exact = angular_profile(emitter @ direction, source)
        # Normalize by the peak of the profile itself, at gamma = 0.
        peak = float(angular_profile(np.asarray(1.0), source))
        self.assertLess(float(np.max(np.abs(reconstruction - exact))) / peak, 1e-5)

    def test_off_equatorial_emitter_excites_sine_harmonics(self) -> None:
        source = LocalizedSourceParameters(source_theta=1.0, source_phi=0.7)
        catalogue = build_mode_catalogue(source, 6)
        self.assertTrue(np.any(catalogue.m < 0))


if __name__ == "__main__":
    unittest.main()
