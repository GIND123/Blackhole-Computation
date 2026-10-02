"""Passive instrumentation regression for the matched velocity-data audit."""

import numpy as np
import pytest


@pytest.mark.parametrize("background", ["sds", "schwarzschild"])
def test_passive_rhs_and_history_hook_leaves_waveform_unchanged(background):
    from black_hole.large_l_tail import INITIAL_DATA, TailCase, _observer_coordinates
    from black_hole.sds_model import SdSParameters
    from black_hole.schwarzschild_scalar import SchwarzschildScalarParameters
    from black_hole.sds_solver import (SdSNumericalParameters, run_sds_simulation,
                                      run_schwarzschild_scalar_simulation)
    from black_hole.uniform_cutoff_audit import instrument

    case = TailCase(background, 64, .0025, .1, 3072.0)
    numerical = SdSNumericalParameters(resolution=64, timestep=.0025,
        end_time=.1, signal_dt=.025, snapshot_dt=.05,
        observers=_observer_coordinates(case), timestepper="RK222",
        bridge="minimal", dealias=1.5)
    model = (SdSParameters(mass=1, ell=1, cosmological_length=3072)
             if background == "sds" else SchwarzschildScalarParameters(mass=1, ell=1))
    evolve = run_sds_simulation if background == "sds" else run_schwarzschild_scalar_simulation
    baseline = evolve(model, INITIAL_DATA, numerical, explicit_potential=True)
    with instrument(ncc_cutoff=1e-6, entry_cutoff=1e-12,
                    history_dt=.01, rhs_times=(.01, .05, .075)) as audit:
        candidate = evolve(model, INITIAL_DATA, numerical, explicit_potential=True)
    np.testing.assert_array_equal(candidate.signal_times, baseline.signal_times)
    np.testing.assert_allclose(candidate.signals, baseline.signals, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(candidate.u_snapshots, baseline.u_snapshots, rtol=1e-10, atol=1e-12)
    assert audit["fixed_velocity_scale"] > 0
    assert audit["normalization"].startswith("M*||G_v||")
    assert audit["rhs"][0]["constraint_linf"] == 0
    assert all(row["explicit_constraint_rhs_linf"] == 0 for row in audit["rhs"])
    assert len(audit["history"]) >= 10
