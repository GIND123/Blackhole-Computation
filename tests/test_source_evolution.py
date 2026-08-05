"""Tests for the sourced hyperboloidal evolution and its static reference."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from black_hole.localized_source import (
    LocalizedSourceParameters,
    build_mode_catalogue,
)
from black_hole.schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    minimal_height,
)
from black_hole.source_evolution import (
    SourcedNumericalParameters,
    load_sourced_result,
    run_sourced_simulation,
)
from black_hole.caustic_study import direction_waveform
from black_hole.static_reference import (
    StaticReferenceGrid,
    reflection_free_time,
    solve_static_mode,
)

SOURCE = LocalizedSourceParameters()


def _settings(**overrides) -> SourcedNumericalParameters:
    base = dict(
        radial_resolution=384,
        angular_ell_max=2,
        timestep=0.01,
        end_time=60.0,
        signal_dt=0.25,
        diagnostic_dt=2.0,
        snapshot_dt=10.0,
        snapshot_end_time=0.0,
        snapshot_radial_points=64,
        observer_radii=(8.0, None),
    )
    base.update(overrides)
    return SourcedNumericalParameters(**base)


class CausalityTests(unittest.TestCase):
    def test_field_is_exactly_zero_before_the_emitter_switches_on(self) -> None:
        result = run_sourced_simulation(
            background="schwarzschild", source=SOURCE, numerical=_settings()
        )
        activation = result.metadata["source_support"]["bridge_time_window"][0]
        self.assertGreater(activation, 0.0)
        early = result.signal_times < activation
        self.assertTrue(early.any())
        self.assertEqual(float(np.max(np.abs(result.modal_signals[early]))), 0.0)

    def test_emitter_that_would_be_active_at_the_initial_slice_is_rejected(self) -> None:
        early = LocalizedSourceParameters(time_center=0.0, time_half_width=1.0)
        with self.assertRaises(ValueError):
            run_sourced_simulation(
                background="schwarzschild", source=early, numerical=_settings()
            )


class StructureTests(unittest.TestCase):
    def test_compact_archive_preserves_full_angular_reconstruction(self) -> None:
        settings = _settings(end_time=10.0, compact_modal_storage=False)
        expanded = run_sourced_simulation(
            background="schwarzschild", source=SOURCE, numerical=settings
        )
        compact = run_sourced_simulation(
            background="schwarzschild",
            source=SOURCE,
            numerical=_settings(end_time=10.0, compact_modal_storage=True),
        )
        self.assertEqual(compact.modal_signals.shape[-1], 0)
        for angle in (0.0, np.pi / 3.0, np.pi):
            np.testing.assert_allclose(
                direction_waveform(compact, angle)[1],
                direction_waveform(expanded, angle)[1],
                rtol=0.0,
                atol=1e-14,
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "compact.npz"
            compact.save(path)
            loaded = load_sourced_result(path)
            np.testing.assert_allclose(
                direction_waveform(loaded, np.pi / 3.0)[1],
                direction_waveform(expanded, np.pi / 3.0)[1],
                rtol=1e-7,
                atol=1e-10,
            )

    def test_response_is_linear_in_the_source_amplitude(self) -> None:
        settings = _settings()
        single = run_sourced_simulation(
            background="schwarzschild", source=SOURCE, numerical=settings
        )
        doubled = run_sourced_simulation(
            background="schwarzschild",
            source=LocalizedSourceParameters(amplitude=2.0),
            numerical=settings,
        )
        scale = float(np.max(np.abs(single.modal_signals)))
        difference = np.max(
            np.abs(doubled.modal_signals - 2.0 * single.modal_signals)
        )
        self.assertLess(difference, 1e-12 * scale)

    def test_only_the_selected_angular_modes_are_carried(self) -> None:
        result = run_sourced_simulation(
            background="schwarzschild", source=SOURCE, numerical=_settings()
        )
        self.assertTrue(np.all((result.mode_ell + result.mode_m) % 2 == 0))
        self.assertTrue(np.all(result.mode_m >= 0))

    def test_reduction_constraint_stays_small_on_both_backgrounds(self) -> None:
        for background in ("schwarzschild", "sds"):
            result = run_sourced_simulation(
                background=background,
                source=SOURCE,
                numerical=_settings(),
                cosmological_length=40.0,
            )
            scale = float(np.max(np.abs(result.modal_signals)))
            self.assertGreater(scale, 1e-3)
            self.assertLess(float(np.max(result.constraint_linf)), 1e-6 * scale)

    def test_schwarzschild_offset_is_the_analytic_value(self) -> None:
        result = run_sourced_simulation(
            background="schwarzschild", source=SOURCE, numerical=_settings()
        )
        self.assertAlmostEqual(
            result.metadata["retarded_time_offset"]["q"],
            4.0 * np.log(2.0),
            places=12,
        )


class StaticReferenceTests(unittest.TestCase):
    def test_hyperboloidal_and_static_solves_agree_at_a_finite_observer(self) -> None:
        settings = _settings(
            radial_resolution=768, timestep=0.005, end_time=120.0, signal_dt=0.1
        )
        result = run_sourced_simulation(
            background="schwarzschild", source=SOURCE, numerical=settings
        )
        catalogue = build_mode_catalogue(SOURCE, settings.angular_ell_max)
        parameters = SchwarzschildScalarParameters(mass=1.0, ell=0)
        observer = 8.0
        height = float(minimal_height(np.asarray(observer), parameters, 4.0))
        grid = StaticReferenceGrid(points=4201)
        limit = reflection_free_time(grid, SOURCE, observer)
        killing = result.signal_times - height
        for index in (0, 2):
            reference = solve_static_mode(
                ell=int(catalogue.ell[index]),
                mode_amplitude=float(catalogue.amplitude[index]),
                source=SOURCE,
                observer_radii=(observer,),
                end_time=min(float(killing[-1]), limit),
                grid=grid,
            )
            times = reference["times"]
            static = reference["signals"][:, 0]
            window = (times >= 15.0) & (times <= min(float(killing[-1]), limit))
            hyperboloidal = np.interp(
                times[window], killing, result.modal_signals[:, 0, index]
            )
            relative = float(
                np.linalg.norm(hyperboloidal - static[window])
                / np.linalg.norm(static[window])
            )
            self.assertLess(relative, 5e-3)

    def test_tortoise_inversion_is_accurate_including_the_deep_horizon(self) -> None:
        grid = StaticReferenceGrid(points=1001)
        tortoise, radius, lapse = grid.coordinates()
        # r-2M underflows as a difference of doubles long before the lapse
        # does, so the inversion is audited through lambda = ln[(r-2M)/2M].
        lam = np.log(lapse / (1.0 - lapse))
        rebuilt = 2.0 * (1.0 + np.exp(lam) + lam)
        self.assertLess(float(np.max(np.abs(rebuilt - tortoise))), 1e-9)
        self.assertTrue(np.all((lapse > 0.0) & (lapse < 1.0)))
        self.assertTrue(np.all(np.diff(lapse) > 0.0))
        # r itself saturates at 2M once r-2M drops below one double ulp; the
        # radius must still be monotone and never fall inside the horizon.
        self.assertTrue(np.all(np.diff(radius) >= 0.0))
        self.assertTrue(np.all(radius >= 2.0))


if __name__ == "__main__":
    unittest.main()
