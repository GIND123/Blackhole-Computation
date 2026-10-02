"""Independent high-precision geometry agrees with frozen scalar backgrounds."""

import mpmath
import numpy as np
import pytest

from black_hole import exterior_sds_model as exterior
from black_hole import schwarzschild_scalar as schwarzschild
from black_hole import sds_model as uniform
from black_hole.sbp_geometry import Geometry


@pytest.mark.parametrize("background", ["schwarzschild", "uniform", "exterior"])
@pytest.mark.parametrize("xi", ["0", "1/6"])
@pytest.mark.parametrize("length", [80, 640, 3072])
def test_matches_archived_coefficient_definitions(background, xi, length):
    geom = Geometry(background, length, ell=1, xi=xi)
    nodes = [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]
    if background == "exterior":
        nodes += [float(geom.rho0), float((geom.rho0 + geom.rho1) / 2), float(geom.rho1)]
    rho = np.array(sorted(nodes))
    actual = np.array([[float(x) for x in geom.coefficients(str(r))] for r in rho]).T
    coupling = float(geom.xi)
    if background == "schwarzschild":
        params = schwarzschild.SchwarzschildScalarParameters(mass=1.0, ell=1)
        A = schwarzschild.propagation_coefficient(rho, params)
        B = schwarzschild.minimal_boost(rho)
        p = rho * (1 - rho)**2 / 2
        P = schwarzschild.rescaled_scalar_potential(rho, params)
    elif background == "uniform":
        params = uniform.SdSParameters(1.0, length, 1, coupling)
        A = uniform.propagation_coefficient(rho, params, "minimal")
        B = uniform.bridge_boost(rho, params, "minimal")
        p = uniform.tortoise_grid_speed(rho, params)
        P = uniform.rescaled_scalar_potential(rho, params)
    else:
        params = exterior.ExteriorSdSParameters(1.0, length, 1, coupling)
        A = exterior.propagation_coefficient(rho, params)
        B = exterior.bridge_boost(rho, params)
        radius = exterior.areal_radius(rho, params)
        p = exterior.metric_f(radius, params) * exterior.compactification_derivative(radius, params)
        P = exterior.rescaled_scalar_potential(rho, params)
    # The production uniform evaluation loses digits through 1+B near H_c;
    # compare at its binary64 accuracy, not at the adapter's working precision.
    np.testing.assert_allclose(actual, np.array([A, B, p, P]), rtol=2e-9, atol=2e-12)


@pytest.mark.parametrize("background", ["schwarzschild", "uniform", "exterior"])
def test_initial_data_and_clock_match_frozen_tail_protocol(background):
    geom = Geometry(background, 640, ell=1)
    radius = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 20.0])
    rho = [geom.compact_radius(str(r)) for r in radius]
    # At uniform SdS the left endpoint lies just above 2M; avoid sampling
    # outside that physical domain, although the bump would be zero there.
    values = np.array([[float(x) for x in geom.initial(r)] for r in rho])
    expected = uniform.compact_areal_velocity_profile(radius, uniform.ArealVelocityBumpInitialData())
    np.testing.assert_allclose(values[:, 0], 0)
    np.testing.assert_allclose(values[:, 1], expected, rtol=1e-14, atol=1e-15)
    if background == "schwarzschild":
        expected_q = schwarzschild.retarded_time_offset(schwarzschild.SchwarzschildScalarParameters(), 4)
    elif background == "uniform":
        expected_q = uniform.retarded_time_offset(uniform.SdSParameters(cosmological_length=640), 4)
    else:
        expected_q = exterior.retarded_time_offset(exterior.ExteriorSdSParameters(cosmological_length=640), 4)
    assert float(geom.clock_offset()) == pytest.approx(expected_q, rel=2e-11, abs=2e-12)


@pytest.mark.parametrize("background", ["schwarzschild", "uniform", "exterior"])
@pytest.mark.parametrize("xi", ["0", "1/6"])
def test_high_precision_endpoints_and_identity(background, xi):
    geom = Geometry(background, 640, xi=xi, dps=70)
    mp = geom.mp
    for rho in [mp.zero, mp.mpf("1e-35"), mp.mpf("0.73"), 1 - mp.mpf("1e-35"), mp.one]:
        A, B, p, P = geom.coefficients(rho)
        assert all(mp.isfinite(x) for x in (A, B, p, P))
        assert A > 0
        assert abs(p - A * (1 - B) * (1 + B)) < mp.mpf("1e-65")
    assert geom.coefficients(0)[1:3] == (mp.one, mp.zero)
    assert geom.coefficients(1)[1:3] == (-mp.one, mp.zero)


def test_conformal_uniform_potential_cancellation():
    geom = Geometry("uniform", 640, xi="1/6", dps=70)
    for rho in [0, "0.995", 1]:
        r = geom.areal_radius(rho)
        expected = geom.D / geom.rb * (2 + 2 * geom.mass / r)
        assert abs(geom.coefficients(rho)[3] - expected) < geom.mp.mpf("1e-65")


@pytest.mark.parametrize("background", ["schwarzschild", "uniform", "exterior"])
@pytest.mark.parametrize("xi", ["0", "1/6"])
@pytest.mark.parametrize("dps", [30, 50, 70])
def test_compact_support_endpoints_are_exactly_zero(background, xi, dps):
    geom = Geometry(background, 640, xi=xi, dps=dps)
    left, right = geom.support_rho
    for rho in [0, left, right, 1]:
        assert geom.initial(rho) == (geom.mp.zero, geom.mp.zero)
    # The support check must not erase valid interior data.
    middle = geom.compact_radius(6 * geom.mass)
    assert geom.initial(middle) == (geom.mp.zero, geom.mp.one)
    for rho in [(left + middle) / 2, (middle + right) / 2]:
        assert 0 < geom.initial(rho)[1] < 1


def test_context_is_local_and_transition_derivatives_are_analytic():
    original_dps = mpmath.mp.dps
    geom = Geometry("exterior", 640, xi="1/6", dps=70)
    assert mpmath.mp.dps == original_dps
    rho = (geom.rho0 + geom.rho1) / 2
    chi, first, second = geom.transition(rho)
    assert 0 < chi < 1
    for derivative, order in [(first, 1), (second, 2)]:
        numeric = geom.mp.diff(lambda x: geom.transition(x)[0], rho, order)
        assert abs(derivative / numeric - 1) < geom.mp.mpf("1e-60")


@pytest.mark.parametrize("background", ["schwarzschild", "uniform", "exterior"])
@pytest.mark.parametrize("length", [320, 640])
def test_displacement_data_matches_frozen_ringdown(background, length):
    geom = Geometry(background, length, ell=2, initial_data="displacement")
    radii = np.array([2.5, 3.0, 4.0, 5.0, 5.5, 7.0])
    actual = np.array([[float(v) for v in geom.initial(geom.compact_radius(str(r)))]
                       for r in radii])
    expected, _ = uniform.compact_areal_profile(radii, uniform.ArealBumpInitialData())
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-15, rtol=1e-14)
    np.testing.assert_array_equal(actual[:, 1], 0)
    for endpoint in geom.support_rho:
        assert geom.initial(endpoint) == (geom.mp.zero, geom.mp.zero)
    assert geom.initial(geom.compact_radius(4)) == (geom.mp.one, geom.mp.zero)



@pytest.mark.parametrize("kwargs", [{"background": "unknown"}, {"background": "uniform", "L": 5}, {"background": "uniform", "L": None}, {"background": "schwarzschild", "ell": -1}, {"background": "schwarzschild", "dps": 20}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        Geometry(**kwargs)
