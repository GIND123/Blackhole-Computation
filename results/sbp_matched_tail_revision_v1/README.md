# Matched dipole-tail replacement figure

All six additional runs in the predeclared `PLAN.md` completed normally.
Together with the existing, verified SBP--Radau runs, they provide the same
five physical signals as the original matched `L/M=640` comparison, with
three spatial degrees and a halved-step control for each family. Original
production archives and paper files were not modified by this analysis.

The figure is `analysis/matched_tail_comparison.pdf`, with a PNG preview.
It has the displaced two-panel figure's 3.4-by-3.5-inch footprint. Panel (a)
shows the actual 10M RMS envelopes, including the deep uniform-minimal
minimum and the exterior secondary structure. Panel (b) shows the 40M fitted
indices only where the original numerical and zero-crossing masks allow
them. Some physical excursions exceed the plotted -4 to 12 ordinate range.
The gray band is `p=3 +/- 10%`; the gold segment identifies the accepted
uniform-conformal interval. Schwarzschild is a single common reference.

## Results

| Family | Primary interval, U/M | Duration, M | Meets 40M criterion |
|---|---:|---:|---|
| Schwarzschild | 140.40–975.00 | at least 834.60 | yes |
| Uniform minimal | 134.50–144.10 | 9.60 | no |
| Uniform conformal | 153.25–294.70 | 141.45 | yes |
| Exterior minimal | 299.00–312.30 | 13.30 | no |
| Exterior conformal | 617.60–626.10 | 8.50 | no |

The original nine estimator-width choices retain these classifications.
Neither uniform case has a resolved cosmological-entry interval before
U=1000M under the unchanged criterion. The exterior results concern this
profile and length; they do not rule out other exterior-supported regulators.

The finest displayed signals use p=40 and dt=0.05M. Spatial comparisons use
p=24,32,40 at dt=0.1M; the half-step control is at p=40. Across 150–300,
300–500, 500–750, and 750–950M, the largest p=32-to-40 waveform change is
3.52e-6 in relative L2 norm (exterior conformal), while the largest timestep
change is 2.18e-7 (uniform minimal). The corresponding coarse spatial change
is larger in every family. No Richardson model or absolute-error bound is
asserted. The largest half-output-cadence interpolation sensitivity is
2.46e-10 in relative L2 norm.

The old-versus-new exterior signed-waveform discrepancies range from 0.83%
to 5.46% on these four windows, despite the unchanged qualitative conclusion.
This range differs from the earlier pilot's merged-window summary: splitting
500–950M at 750M exposes a larger relative discrepancy in the weaker
late minimally coupled exterior signal. The old numerical intervals and
convergence numbers must not accompany the new curves unchanged.

## Derived records

- `matched_tail_intervals.csv`: exact primary intervals.
- `matched_tail_estimator_sweep.csv`: all 45 estimator choices.
- `matched_tail_refinement.csv`: signed-waveform spatial, time, and
  output-cadence comparisons on the four windows.
- `matched_tail_accepted_interval_refinement.csv`: controls on passing
  intervals, including the uniform-conformal accepted interval.
- `matched_tail_frozen_comparison.csv`: separate discrepancies from the
  finest old archives, normalized by the corresponding frozen signal.
- `matched_tail_audit_summary.json`: compact per-family and per-window
  values for manuscript editing.
- `matched_tail_curves.npz`: actual masked plotted arrays, not reconstructed
  curves from interval endpoints.
- `matched_tail_manifest.json`: inputs, source hashes, and exact estimator
  settings. New and reused raw runs remain separate.

The standalone sources are `black_hole/sbp_matched_tail_figure.py` and
`black_hole/sbp_matched_tail_audit.py`. Three new regression tests verify
sample-for-sample agreement with the archived estimator/gate on common
synthetic data, the half-fit-width zero-crossing exclusions, and rejection
of numerically unresolved indices. Together with the existing comparison
tests, seven focused tests pass.
