"""Checks for endpoint-factored physical characteristic variables."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


DEDALUS_AVAILABLE = importlib.util.find_spec("dedalus") is not None

if DEDALUS_AVAILABLE:
    from black_hole.exterior_sds_model import ExteriorSdSParameters
    from black_hole.sds_model import (
        ArealVelocityBumpInitialData,
        ScalarInitialData,
    )
    from black_hole.sds_solver import (
        SdSNumericalParameters,
        run_exterior_sds_simulation,
    )


@unittest.skipUnless(DEDALUS_AVAILABLE, "Dedalus 3 is not installed")
class EndpointFactoredCharacteristicTests(unittest.TestCase):
    @staticmethod
    def short_numerical() -> "SdSNumericalParameters":
        return SdSNumericalParameters(
            resolution=48,
            timestep=0.005,
            end_time=0.05,
            signal_dt=0.005,
            snapshot_dt=0.025,
            observers=(0.25, 0.75, 1.0),
            timestepper="RK222",
            dealias=1.5,
        )

    def test_short_waveform_matches_standard_reduction(self) -> None:
        model = ExteriorSdSParameters(cosmological_length=80.0, ell=1)
        initial = ScalarInitialData(center_fraction=0.5, width=0.08)
        standard = run_exterior_sds_simulation(
            model,
            initial,
            self.short_numerical(),
            explicit_potential=True,
        )
        factored = run_exterior_sds_simulation(
            model,
            initial,
            self.short_numerical(),
            explicit_potential=True,
            endpoint_factored_characteristic_variables=True,
            characteristic_constraint_damping=1.0,
        )

        np.testing.assert_allclose(
            factored.signals,
            standard.signals,
            rtol=0.0,
            atol=5.0e-8,
        )
        self.assertEqual(factored.u_snapshots.shape[1], factored.rho.size)
        audit = factored.metadata["factored_coefficient_audit"]
        self.assertTrue(audit["finite_and_positive"])
        self.assertLess(audit["alpha_plus_chebyshev_tail_ratio"], 0.01)
        self.assertLess(audit["alpha_minus_chebyshev_tail_ratio"], 0.001)
        self.assertEqual(
            factored.metadata["constraint"]["definition"],
            "C=H/(2*alpha_plus)-J/(2*alpha_minus)-d_rho(u)",
        )
        self.assertEqual(factored.metadata["constraint"]["damping_rate"], 1.0)
        self.assertEqual(
            factored.metadata["constraint"]["continuum_propagation"],
            "d_tau(C)=-gamma*C",
        )

    def test_physical_waveform_is_independent_of_damping_rate(self) -> None:
        """Constraint damping must not change a compatible scalar solution."""

        model = ExteriorSdSParameters(cosmological_length=80.0, ell=1)
        initial = ScalarInitialData(center_fraction=0.5, width=0.08)
        reference = run_exterior_sds_simulation(
            model,
            initial,
            self.short_numerical(),
            explicit_potential=True,
            endpoint_factored_characteristic_variables=True,
            characteristic_constraint_damping=0.5,
        )
        for damping in (1.0, 2.0):
            result = run_exterior_sds_simulation(
                model,
                initial,
                self.short_numerical(),
                explicit_potential=True,
                endpoint_factored_characteristic_variables=True,
                characteristic_constraint_damping=damping,
            )
            np.testing.assert_allclose(
                result.signals,
                reference.signals,
                rtol=0.0,
                atol=5.0e-8,
            )

    def test_conservative_characteristics_match_standard_reduction(self) -> None:
        model = ExteriorSdSParameters(cosmological_length=80.0, ell=1)
        initial = ScalarInitialData(center_fraction=0.5, width=0.08)
        standard = run_exterior_sds_simulation(
            model,
            initial,
            self.short_numerical(),
            explicit_potential=True,
        )
        conservative = run_exterior_sds_simulation(
            model,
            initial,
            self.short_numerical(),
            explicit_potential=True,
            conservative_characteristic_variables=True,
        )
        np.testing.assert_allclose(
            conservative.signals,
            standard.signals,
            rtol=0.0,
            atol=1.0e-7,
        )
        self.assertEqual(conservative.u_snapshots.shape[1], conservative.rho.size)
        self.assertLessEqual(
            np.max(conservative.constraint_linf),
            1.01 * conservative.constraint_linf[0] + 1.0e-12,
        )
        constraint = conservative.metadata["constraint"]
        self.assertEqual(constraint["definition"], "C=(h-j)/2-d_rho(u)")
        self.assertIn("identical Fplus", constraint["semidiscrete_propagation"])

    def test_characteristic_variable_choices_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "either endpoint-factored"):
            run_exterior_sds_simulation(
                ExteriorSdSParameters(cosmological_length=80.0, ell=1),
                ScalarInitialData(center_fraction=0.5, width=0.08),
                self.short_numerical(),
                endpoint_factored_characteristic_variables=True,
                conservative_characteristic_variables=True,
            )

    def test_damping_requires_factored_variables(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires endpoint-factored"):
            run_exterior_sds_simulation(
                ExteriorSdSParameters(cosmological_length=80.0, ell=1),
                ScalarInitialData(center_fraction=0.5, width=0.08),
                self.short_numerical(),
                characteristic_constraint_damping=1.0,
            )

    def test_damping_rate_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            run_exterior_sds_simulation(
                ExteriorSdSParameters(cosmological_length=80.0, ell=1),
                ScalarInitialData(center_fraction=0.5, width=0.08),
                self.short_numerical(),
                endpoint_factored_characteristic_variables=True,
                characteristic_constraint_damping=-1.0,
            )

    def test_factored_checkpoint_reloads_exactly(self) -> None:
        model = ExteriorSdSParameters(cosmological_length=80.0, ell=1)
        initial = ScalarInitialData(center_fraction=0.5, width=0.08)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "endpoint_factored.npz"
            first = run_exterior_sds_simulation(
                model,
                initial,
                self.short_numerical(),
                checkpoint_path=checkpoint,
                checkpoint_dt=0.025,
                explicit_potential=True,
                endpoint_factored_characteristic_variables=True,
                characteristic_constraint_damping=1.0,
            )
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertIn("u", saved)
                self.assertIn("H", saved)
                self.assertIn("J", saved)
                self.assertNotIn("psi", saved)
                self.assertNotIn("pi", saved)
                self.assertIn("state_field_scales", saved)
            resumed = run_exterior_sds_simulation(
                model,
                initial,
                self.short_numerical(),
                checkpoint_path=checkpoint,
                checkpoint_dt=0.025,
                explicit_potential=True,
                endpoint_factored_characteristic_variables=True,
                characteristic_constraint_damping=1.0,
            )

        self.assertTrue(resumed.metadata["checkpoint_restart"]["resumed"])
        for name in (
            "signal_times",
            "signals",
            "snapshot_times",
            "u_snapshots",
            "constraint_linf",
            "constraint_l2",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(resumed, name))

    def test_conservative_checkpoint_reloads_exactly(self) -> None:
        model = ExteriorSdSParameters(cosmological_length=80.0, ell=1)
        initial = ScalarInitialData(center_fraction=0.5, width=0.08)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "conservative.npz"
            first = run_exterior_sds_simulation(
                model,
                initial,
                self.short_numerical(),
                checkpoint_path=checkpoint,
                checkpoint_dt=0.025,
                explicit_potential=True,
                conservative_characteristic_variables=True,
            )
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertIn("u", saved)
                self.assertIn("h", saved)
                self.assertIn("j", saved)
                self.assertNotIn("psi", saved)
                self.assertNotIn("pi", saved)
                self.assertEqual(saved["u_snapshots"].shape[1], 48)
            resumed = run_exterior_sds_simulation(
                model,
                initial,
                self.short_numerical(),
                checkpoint_path=checkpoint,
                checkpoint_dt=0.025,
                explicit_potential=True,
                conservative_characteristic_variables=True,
            )

        self.assertTrue(resumed.metadata["checkpoint_restart"]["resumed"])
        for name in (
            "signal_times",
            "signals",
            "snapshot_times",
            "u_snapshots",
            "constraint_linf",
            "constraint_l2",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(resumed, name))

    @unittest.skipUnless(
        os.environ.get("RUN_LONG_DEDALUS_TESTS") == "1",
        "set RUN_LONG_DEDALUS_TESTS=1 for the 300M endpoint stress test",
    )
    def test_l640_outer_horizon_remains_bounded_through_300M(self) -> None:
        """Catch endpoint-flux and compatibility-mode regressions."""

        numerical = SdSNumericalParameters(
            resolution=128,
            timestep=0.005,
            end_time=300.0,
            signal_dt=1.0,
            snapshot_dt=25.0,
            observers=(1.0,),
            timestepper="RK222",
            dealias=1.5,
        )
        result = run_exterior_sds_simulation(
            ExteriorSdSParameters(cosmological_length=640.0, ell=1),
            ArealVelocityBumpInitialData(),
            numerical,
            explicit_potential=True,
            endpoint_factored_characteristic_variables=True,
            characteristic_constraint_damping=1.0,
        )
        horizon = result.signals[:, 0]
        late = result.signal_times >= 80.0
        self.assertTrue(np.all(np.isfinite(horizon)))
        self.assertLess(np.max(np.abs(horizon[late])), 0.02)
        self.assertLess(abs(horizon[-1]), 0.01)
        late_constraints = result.constraint_linf[result.snapshot_times >= 80.0]
        self.assertLess(
            np.max(late_constraints),
            5.0e-3,
            msg=(
                f"constraint snapshots={result.constraint_linf.tolist()} at "
                f"times={result.snapshot_times.tolist()}"
            ),
        )
        self.assertLess(late_constraints[-1], late_constraints[0])


if __name__ == "__main__":
    unittest.main()
