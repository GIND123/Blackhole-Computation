"""Preserve the matched-tail estimator and mask when changing discretization."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from black_hole.curvature_coupling_tail_analysis import rate_diagnostic
from black_hole.sbp_matched_tail_figure import FAMILIES, create_figure, diagnose, interval_row
from black_hole.sbp_pilot_analysis import Run


def sample_run(times, signal, degree=40, dt=.05):
    return Run("sample", Path("sample"),
               dict(background="schwarzschild", ell=1, q="0", degree=degree,
                    dt=str(dt), layout="standard", integrator="radau3"),
               dict(precision="double-double", method="Radau"), [], [],
               times, signal, np.gradient(signal, times))


def test_same_estimator_and_640_gate_as_archived_pipeline():
    times = np.arange(0., 1000.025, .05)
    signal = (times+20.)**-3
    run = sample_run(times, signal)
    new = diagnose(dict(coarse=run, medium=run, fine=run, half_dt=run))
    archive = SimpleNamespace(signal_times=times,
                              signals=np.column_stack([signal]*3),
                              metadata={"retarded_time_offset": {"q": 0.}})
    old = rate_diagnostic({(1536,.0025):archive, (2048,.0025):archive,
                           (3072,.0025):archive, (2048,.00125):archive})
    np.testing.assert_allclose(new["amplitude"],old["amplitude"],equal_nan=True)
    np.testing.assert_allclose(new["power"],old["power"],equal_nan=True)
    np.testing.assert_array_equal(new["power_supported"],old["power_supported"])
    assert new["price_interval"] == old["price_interval"]


def test_zero_crossings_retain_half_fit_width_exclusion():
    times = np.arange(0.,1000.025,.05)
    signal = (times-500.)*(times+20.)**-4
    run = sample_run(times,signal)
    diagnostic = diagnose(dict(coarse=run,medium=run,fine=run,half_dt=run))
    selected = (times >= 480.1) & (times <= 519.9)
    assert not np.any(diagnostic["power_supported"][selected])
    assert np.any(diagnostic["power_supported"][(times > 550) & (times < 600)])


def test_unresolved_rate_is_not_accepted_even_when_fine_power_looks_correct():
    times = np.arange(0.,1000.025,.05)
    fine = sample_run(times,(times+20.)**-3)
    medium = sample_run(times,(times+20.)**-2)
    diagnostic = diagnose(dict(coarse=medium,medium=medium,fine=fine,half_dt=fine))
    assert diagnostic["price_interval"] is None
    assert not interval_row("synthetic",diagnostic)["passes_price_criterion"]


def test_figure_annotation_uses_table_interval_without_changing_curves(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    times = np.array([100., 200., 300., 400., 975.])
    values = np.array([3., 3., np.nan, 3.1, 3.2])
    diagnostics = {family.stem: dict(times=times, amplitude=np.full(5, 1e-7),
                                    power=values, power_supported=np.isfinite(values),
                                    price_interval=(201., 249.))
                   for family in FAMILIES}
    # Deliberately differ from the diagnostic interval: annotations read table rows.
    rows = [dict(family="uniform640_conformal", start_U_over_M="155.0",
                 end_U_over_M="297.0", duration_over_M="142.0")]
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: captured.append(figure))
    create_figure(diagnostics, tmp_path, rows)
    figure = captured[0]
    assert tuple(figure.get_size_inches()) == (3.4, 3.5)
    assert "Uniform conformal: $142M$" in [text.get_text() for text in figure.axes[0].texts]
    assert figure.axes[1].get_ylim() == (-4., 12.)
    for axis in figure.axes:
        shading = axis.patches[-1]
        assert shading.get_x() == 155.
        assert shading.get_width() == 142.
        for line in axis.lines[:5]:
            np.testing.assert_array_equal(line.get_xdata(), times)
    for line in figure.axes[1].lines[:5]:
        np.testing.assert_array_equal(line.get_ydata(), values)
    plt.close(figure)
