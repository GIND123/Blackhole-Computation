"""Checks for the caustic phase measurement."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from black_hole.caustic_phase import (
    arrival_time,
    direction_trace,
    null_ray_consistency,
    phase_fit,
    truncation_sensitivity,
)
from black_hole.caustic_study import direction_waveform
from black_hole.source_evolution import load_sourced_result


ARCHIVE = Path("results/regulator_production_v3/raw/source/fine/sds_L80.npz")


class PhaseFitTests(unittest.TestCase):
    """The estimator is checked against signals whose answer is known."""

    def setUp(self) -> None:
        self.times = np.linspace(0.0, 120.0, 24001)
        centre, width = 30.0, 2.0
        self.pulse = -np.exp(-(((self.times - centre) / width) ** 2))

    def test_unrotated_copy_returns_zero_phase(self) -> None:
        delayed = np.interp(
            self.times, self.times + 12.0, self.pulse, left=0.0, right=0.0
        )
        fit = phase_fit(self.times, self.pulse, 2.5 * delayed, (34.0, 56.0))
        self.assertAlmostEqual(fit.phase_degrees, 0.0, places=3)
        self.assertAlmostEqual(fit.amplitude, 2.5, places=3)
        self.assertAlmostEqual(fit.delay_over_M, 12.0, places=6)
        self.assertGreater(fit.variance_explained, 0.999)

    def test_sign_reversal_returns_one_hundred_and_eighty_degrees(self) -> None:
        fit = phase_fit(self.times, self.pulse, -self.pulse, (20.0, 44.0))
        self.assertAlmostEqual(abs(fit.phase_degrees), 180.0, places=3)
        self.assertGreater(fit.variance_explained, 0.999)

    def test_hilbert_transform_returns_ninety_degrees(self) -> None:
        quadrature = np.imag(hilbert(self.pulse))
        fit = phase_fit(self.times, self.pulse, quadrature, (20.0, 44.0))
        self.assertAlmostEqual(abs(fit.phase_degrees), 90.0, places=2)
        self.assertGreater(fit.variance_explained, 0.999)


@unittest.skipUnless(ARCHIVE.exists(), "final localized-source archive is absent")
class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = load_sourced_result(ARCHIVE)
        cls.times = np.asarray(cls.result.retarded_time, dtype=float)

    def test_direction_trace_matches_the_established_reconstruction(self) -> None:
        observer = self.result.outer_index()
        for gamma in (0.0, np.pi / 3.0, np.pi):
            trace = direction_trace(self.result, gamma, observer)
            # direction_waveform takes the equatorial azimuth, which equals the
            # angle from the emitter for a source at theta = pi/2, phi = 0.
            _, established = direction_waveform(self.result, gamma, observer)
            self.assertTrue(np.allclose(trace, established, atol=1e-12))

    def test_antipodal_rotation_is_neither_a_copy_nor_a_sign_reversal(self) -> None:
        direct = direction_trace(self.result, 0.0)
        antipode = direction_trace(self.result, np.pi)
        peak = arrival_time(self.times, antipode, 20.0)
        fit = phase_fit(
            self.times,
            direct,
            antipode,
            (peak - 8.0, min(float(self.times[-1]), peak + 14.0)),
        )
        self.assertGreater(fit.variance_explained, 0.9)
        self.assertGreater(fit.phase_degrees, 30.0)
        self.assertLess(fit.phase_degrees, 60.0)

    def test_rotation_is_absent_away_from_the_axis(self) -> None:
        direct = direction_trace(self.result, 0.0)
        for degrees in (30.0, 60.0, 90.0):
            signal = direction_trace(self.result, float(np.radians(degrees)))
            peak = arrival_time(self.times, signal, 20.0)
            fit = phase_fit(
                self.times, direct, signal, (peak - 8.0, peak + 14.0)
            )
            self.assertLess(abs(fit.phase_degrees), 10.0)

    def test_arrivals_agree_with_inward_turning_null_rays(self) -> None:
        rows = {row["arrival"]: row for row in null_ray_consistency(ARCHIVE)}
        # The envelope maximum of a pulse of finite width is not the
        # geometrical optics arrival, so a few tenths of M is the expected
        # offset on arrivals of 26 and 44.
        self.assertLess(abs(rows["direct"]["difference_over_M"]), 0.2)
        self.assertLess(abs(rows["antipodal"]["difference_over_M"]), 0.6)
        self.assertLess(abs(rows["delay"]["difference_over_M"]), 0.6)
        # The wrapping ray turns inside the region set by the photon sphere.
        self.assertLess(rows["antipodal"]["turning_radius_over_M"], 6.0)
        self.assertGreater(rows["antipodal"]["turning_radius_over_M"], 3.0)

    def test_measurement_is_converged_in_the_angular_truncation(self) -> None:
        rows = truncation_sensitivity(ARCHIVE, orders=(30, 40, 50))
        phases = [row["phase_degrees"] for row in rows]
        peaks = [row["antipodal_peak_amplitude"] for row in rows]
        self.assertLess(max(phases) - min(phases), 1e-4)
        self.assertLess((max(peaks) - min(peaks)) / max(peaks), 1e-4)


if __name__ == "__main__":
    unittest.main()
