"""Checks that the constraint audit observes, rather than repairs, a run."""

import numpy as np
import pytest

from black_hole.exterior_sds_model import ExteriorSdSParameters
from black_hole.regulator_suite import flat_initial_data
from black_hole.ringdown_constraint_diagnostic import instrument_solver
from black_hole.sds_solver import SdSNumericalParameters, run_exterior_sds_simulation


@pytest.mark.parametrize("shared", [False, True])
def test_audit_preserves_evolution_and_records_stages(shared):
    model = ExteriorSdSParameters(cosmological_length=640, ell=2)
    initial = flat_initial_data()
    numerical = SdSNumericalParameters(
        resolution=32, timestep=.0025, end_time=.01,
        signal_dt=.0025, snapshot_dt=.01, observers=(0., .5, 1.),
    )
    kwargs = dict(explicit_potential=True,
                  conservative_characteristic_variables=shared)
    control = run_exterior_sds_simulation(model, initial, numerical, **kwargs)
    with instrument_solver(history_dt=.005) as audit:
        observed = run_exterior_sds_simulation(model, initial, numerical, **kwargs)
    np.testing.assert_allclose(observed.signals, control.signals, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(observed.u_snapshots, control.u_snapshots, rtol=1e-10, atol=1e-12)
    assert audit["initial_constraint_scale"] > 0
    assert len(audit["stages"]) == 6
    assert audit["history"][-1]["tau"] == pytest.approx(.01)
    assert [row["tau"] for row in audit["stages"]] == sorted(row["tau"] for row in audit["stages"])
    for row in audit["rhs"]:
        assert row["explicit_constraint_rhs_linf"] == 0
        if shared:
            assert row["complete_constraint_rhs_linf"] < 1e-10
