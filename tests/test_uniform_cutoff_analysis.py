"""Fail-closed guards for the isolated matched cutoff comparison."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from black_hole.sds_result import load_sds_result
from black_hole.uniform_cutoff_analysis import validate_pair
import black_hole.uniform_cutoff_analysis as analysis


@pytest.fixture(scope="module")
def archives():
    production = Path("results/large_l_tail/raw/final_sds_L3072_N2048_dt0p0025.npz")
    tightened = Path("results/uniform_cutoff_revision_v1/raw/sds_N2048_tight.npz")
    if not production.exists() or not tightened.exists():
        pytest.skip("isolated cutoff archives are not installed")
    return load_sds_result(production), load_sds_result(tightened)


def test_matching_completed_configuration_and_padding(archives):
    result = validate_pair(*archives)
    assert result["normal_completion"]
    assert result["end_U"] == 450
    assert result["padding_beyond_fixed_interval"] > result["required_composed_padding"]


@pytest.mark.parametrize("field,value", [
    ("timestep", .00125), ("dealias", 2), ("bridge", "linear"),
    ("signal_dt", .1), ("resolution", 3072)])
def test_reject_changed_discretization(archives, field, value):
    reference, candidate = archives
    metadata = deepcopy(candidate.metadata)
    metadata["numerical"][field] = value
    with pytest.raises(ValueError, match="Unmatched numerical"):
        validate_pair(reference, replace(candidate, metadata=metadata))


def test_reject_incomplete_run(archives):
    reference, candidate = archives
    metadata = deepcopy(candidate.metadata)
    metadata["final_time"] -= 1
    with pytest.raises(ValueError, match="did not finish"):
        validate_pair(reference, replace(candidate, metadata=metadata))


def test_reject_changed_clock(archives):
    reference, candidate = archives
    metadata = deepcopy(candidate.metadata)
    metadata["retarded_time_offset"]["q"] += .1
    with pytest.raises(ValueError, match="clock"):
        validate_pair(reference, replace(candidate, metadata=metadata))


def test_reject_nonfinite_signal(archives):
    reference, candidate = archives
    signals = candidate.signals.copy()
    signals[100, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_pair(reference, replace(candidate, signals=signals))


def test_reject_displacement_normalization(archives):
    reference, candidate = archives
    metadata = deepcopy(candidate.metadata)
    metadata["uniform_cutoff_audit"]["fixed_velocity_scale"] = 0
    with pytest.raises(ValueError, match="velocity-normalized"):
        validate_pair(reference, replace(candidate, metadata=metadata))


@pytest.mark.parametrize("failure", [None, "sds_band", "reference_band", "mutual_band",
                                     "sds_refinement", "reference_refinement", "duration", "anchor"])
def test_unchanged_acceptance_rule_keeps_every_condition(monkeypatch, failure):
    times = np.arange(0, 450.05, .05)
    sds_power = np.full_like(times, 3.0)
    ref_power = np.full_like(times, 3.0)
    if failure == "sds_band":
        sds_power[:] = 3.151
    if failure == "reference_band":
        ref_power[:] = 3.151
    if failure == "mutual_band":
        sds_power[:], ref_power[:] = 3.1, 2.9
    if failure == "duration":
        sds_power[(times < 250) | (times > 350)] = np.nan
    if failure == "anchor":
        sds_power[times > 250] = np.nan
    floors = {}
    for bg in ("sds", "schwarzschild"):
        fail = failure == ("sds_refinement" if bg == "sds" else "reference_refinement")
        floors[bg] = {"times": times, "amplitude": np.ones_like(times),
                      "floor": np.full_like(times, .001),
                      "spatial_fine": np.full_like(times, .011 if fail else .001)}
    fake = SimpleNamespace(signal_times=times, signals=np.ones((times.size, 3)),
                           metadata={"retarded_time_offset": {"q": 0.0}})
    monkeypatch.setattr(analysis, "measure_transition", lambda *a, **k: {"power": sds_power})
    monkeypatch.setattr(analysis, "effective_rates", lambda *a, **k: (np.ones_like(times), ref_power, np.zeros_like(times)))
    result = analysis.classify(fake, fake, floors)
    assert result["classification"] == ("passes" if failure is None else "fails")
