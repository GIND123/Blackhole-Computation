"""Small regression checks for the independent pilot comparison pipeline."""

from decimal import Decimal
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from black_hole.sbp_pilot_analysis import (
    Run, control_type, exact_difference, metrics, read_exclusions,
)


def sample_run():
    return Run("sample", Path("sample"),
               dict(background="schwarzschild", ell=1, q="0", degree=32,
                    dt="0.1", layout="standard", integrator="radau3"),
               dict(precision="double-double", integrator="radau3"),
               [Decimal("0.1"), Decimal("0.2")],
               [Decimal("1"), Decimal("2")],
               np.array([.1,.2]), np.array([1.,2.]), np.array([0.,0.]))


def test_decimal_comparison_preserves_sub_binary64_differences():
    first, second = sample_run(), sample_run()
    first.outer_decimal[0] = Decimal("1.0000000000000000000000001")
    second.tau_decimal[0] = Decimal("0.10000000000000000000000000001")
    times, difference, reference = exact_difference(first,second)
    np.testing.assert_array_equal(times,[.1,.2])
    assert difference[0] == 1e-25
    assert difference[1] == 0
    np.testing.assert_array_equal(reference,[1.,2.])


def test_controls_do_not_mix_integrator_with_spatial_refinement():
    first, second = sample_run(), sample_run()
    second.configuration["degree"] = 40
    assert control_type(first,second) == "spatial"
    second.configuration["integrator"] = "hermite"
    second.backend["integrator"] = "hermite"
    assert control_type(first,second) is None
    second.configuration["degree"] = 32
    assert control_type(first,second) == "integrator"


def test_quarantine_requires_explicit_nonempty_reasons():
    with tempfile.TemporaryDirectory(prefix="sbp-exclusions-") as temporary:
        root=Path(temporary)
        assert read_exclusions(root) == {}
        path=root/"excluded_runs.json"
        path.write_text(json.dumps({"bad-run":"invalid preliminary input parser"}))
        assert "bad-run" in read_exclusions(root)
        path.write_text(json.dumps({"bad-run":""}))
        with pytest.raises(ValueError):
            read_exclusions(root)


def test_relative_norm_and_full_window_requirement():
    times=np.linspace(0,10,101)
    measured=metrics(times,np.full(101,.1),np.full(101,2.),0,10)
    assert measured["relative_L2"] == pytest.approx(.05)
    assert metrics(times,np.ones(101),np.ones(101),0,20) is None
