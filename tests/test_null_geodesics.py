"""Tests for exact Schwarzschild and SdS null ray timing."""

from __future__ import annotations

import unittest

import numpy as np

from black_hole.null_geodesics import generic_target_angle, trace_null_ray


class NullRayTests(unittest.TestCase):
    def test_radial_direct_ray_has_known_schwarzschild_retarded_time(self) -> None:
        ray = trace_null_ray(
            source_radius=6.0,
            observer_radius=None,
            target_angle=0.0,
            emission_time=30.0,
        )
        expected = 30.0 - (2.0 + 2.0 * np.log(2.0))
        self.assertAlmostEqual(ray.arrival_u, expected, places=12)

    def test_high_winding_sds_interval_approaches_photon_orbit_prediction(self) -> None:
        first = trace_null_ray(
            source_radius=6.0,
            observer_radius=12.0,
            target_angle=7.0 * np.pi,
            emission_time=30.0,
            cosmological_length=40.0,
            winding=7,
        )
        second = trace_null_ray(
            source_radius=6.0,
            observer_radius=12.0,
            target_angle=8.0 * np.pi,
            emission_time=30.0,
            cosmological_length=40.0,
            winding=8,
        )
        expected = np.pi / first.photon_frequency
        self.assertAlmostEqual(second.arrival_u - first.arrival_u, expected, delta=2e-3)

    def test_direct_generic_ray_uses_inward_branch_when_required(self) -> None:
        ray = trace_null_ray(
            source_radius=6.0,
            observer_radius=None,
            target_angle=np.pi / 2.0,
            emission_time=30.0,
            cosmological_length=12.0,
            winding=0,
        )
        self.assertTrue(np.isfinite(ray.arrival_u))
        self.assertTrue(np.isfinite(ray.turning_radius))
        self.assertGreater(ray.turning_radius, 3.0)

    def test_generic_targets_have_alternating_asymptotic_intervals(self) -> None:
        gamma = np.pi / 3.0
        targets = [generic_target_angle(gamma, pulse) for pulse in range(4)]
        np.testing.assert_allclose(
            np.diff(targets),
            [2.0 * (np.pi - gamma), 2.0 * gamma, 2.0 * (np.pi - gamma)],
        )


if __name__ == "__main__":
    unittest.main()
