"""Checks for atomic Dedalus scalar checkpoints."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


DEDALUS_AVAILABLE = importlib.util.find_spec("dedalus") is not None

if DEDALUS_AVAILABLE:
    from black_hole.sds_model import ScalarInitialData
    from black_hole.sds_solver import (
        SdSNumericalParameters,
        run_schwarzschild_scalar_simulation,
    )
    from black_hole.schwarzschild_scalar import SchwarzschildScalarParameters


@unittest.skipUnless(DEDALUS_AVAILABLE, "Dedalus 3 is not installed")
class ScalarCheckpointTests(unittest.TestCase):
    def test_saved_state_and_output_histories_reload_exactly(self) -> None:
        numerical = SdSNumericalParameters(
            resolution=32,
            timestep=0.01,
            end_time=0.1,
            signal_dt=0.01,
            snapshot_dt=0.05,
            observers=(0.25, 1.0),
            timestepper="RK222",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "scalar_checkpoint.npz"
            first = run_schwarzschild_scalar_simulation(
                SchwarzschildScalarParameters(ell=1),
                ScalarInitialData(),
                numerical,
                checkpoint_path=checkpoint,
                checkpoint_dt=0.05,
            )
            resumed = run_schwarzschild_scalar_simulation(
                SchwarzschildScalarParameters(ell=1),
                ScalarInitialData(),
                numerical,
                checkpoint_path=checkpoint,
                checkpoint_dt=0.05,
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
            np.testing.assert_array_equal(
                getattr(first, name), getattr(resumed, name)
            )


if __name__ == "__main__":
    unittest.main()
