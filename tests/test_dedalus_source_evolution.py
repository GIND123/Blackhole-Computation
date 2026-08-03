"""Cross-discretization tests for the optional Dedalus source backend."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from black_hole.localized_source import LocalizedSourceParameters
from black_hole.source_evolution import (
    SourcedNumericalParameters,
    run_sourced_simulation,
)

DEDALUS_AVAILABLE = importlib.util.find_spec("dedalus") is not None

if DEDALUS_AVAILABLE:
    from black_hole.dedalus_source_evolution import (
        run_sourced_dedalus_simulation,
    )


class GreenFunctionCliTests(unittest.TestCase):
    def test_backend_option_parses_without_importing_dedalus_solver(self) -> None:
        from black_hole.__main__ import build_parser

        arguments = build_parser().parse_args(
            ["green-function-run", "--backend", "dedalus", "schwarzschild"]
        )
        self.assertEqual(arguments.backend, "dedalus")
        self.assertEqual(arguments.cases, ["schwarzschild"])


@unittest.skipUnless(DEDALUS_AVAILABLE, "Dedalus 3 is not installed")
class DedalusSourceEvolutionTests(unittest.TestCase):
    def test_zero_field_is_exact_before_source_activation(self) -> None:
        numerical = SourcedNumericalParameters(
            radial_resolution=128,
            angular_ell_max=1,
            timestep=0.01,
            end_time=0.1,
            signal_dt=0.05,
            diagnostic_dt=0.05,
            snapshot_dt=0.1,
            snapshot_radial_points=32,
        )
        result = run_sourced_dedalus_simulation(
            background="schwarzschild",
            source=LocalizedSourceParameters(),
            numerical=numerical,
            dealias=1.0,
        )
        self.assertEqual(float(np.max(np.abs(result.modal_signals))), 0.0)
        self.assertEqual(float(np.max(result.field_linf)), 0.0)
        self.assertEqual(float(np.max(result.source_activity)), 0.0)
        self.assertEqual(float(np.max(result.constraint_linf)), 0.0)

    def test_active_source_matches_finite_difference_backend(self) -> None:
        source = LocalizedSourceParameters(time_center=6.0, time_half_width=0.75)
        numerical = SourcedNumericalParameters(
            radial_resolution=128,
            angular_ell_max=2,
            timestep=0.01,
            end_time=10.0,
            signal_dt=0.05,
            diagnostic_dt=0.25,
            snapshot_dt=5.0,
            snapshot_end_time=0.0,
            snapshot_radial_points=32,
            observer_radii=(8.0, None),
        )
        for background in ("schwarzschild", "sds"):
            with self.subTest(background=background):
                finite_difference = run_sourced_simulation(
                    background=background,
                    source=source,
                    numerical=numerical,
                    cosmological_length=80.0,
                )
                dedalus = run_sourced_dedalus_simulation(
                    background=background,
                    source=source,
                    numerical=numerical,
                    cosmological_length=80.0,
                )

                np.testing.assert_array_equal(
                    finite_difference.mode_ell, dedalus.mode_ell
                )
                np.testing.assert_array_equal(
                    finite_difference.mode_m, dedalus.mode_m
                )
                self.assertGreater(float(np.max(dedalus.source_activity)), 0.99)
                self.assertGreater(float(np.max(dedalus.field_linf)), 0.1)
                self.assertLess(float(np.max(dedalus.constraint_linf)), 1e-7)

                active = dedalus.signal_times >= 3.0
                difference = (
                    finite_difference.modal_signals[active]
                    - dedalus.modal_signals[active]
                )
                relative_l2 = float(
                    np.linalg.norm(difference)
                    / np.linalg.norm(dedalus.modal_signals[active])
                )
                self.assertLess(relative_l2, 3e-3)

                ell_two = np.flatnonzero(dedalus.mode_ell == 2)
                self.assertGreaterEqual(ell_two.size, 2)
                first, second = ell_two[:2]
                normalized_first = (
                    dedalus.modal_signals[..., first]
                    / dedalus.mode_source_amplitude[first]
                )
                normalized_second = (
                    dedalus.modal_signals[..., second]
                    / dedalus.mode_source_amplitude[second]
                )
                np.testing.assert_allclose(
                    normalized_first,
                    normalized_second,
                    rtol=0.0,
                    atol=1e-14,
                )


if __name__ == "__main__":
    unittest.main()
