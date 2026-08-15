# Artificial-cosmology regulator: final analysis

## Paper framing and supported conclusion

The Physical Review D paper will center on the quantitative demonstration of
Misner's artificial cosmology regulator in a black hole spacetime, including
controlled recovery of Schwarzschild waveforms as `L` increases. The
Schwarzschild de Sitter propagation and caustic echo results provide
supporting physical evidence and do not constitute a competing main claim.

Schwarzschild de Sitter is supported as an artificial cosmology regulator
for the fixed data and localized source observables tested here, at a
conservative **1% extrapolated waveform level**. This statement applies to
the cumulative prompt dominated pure `ell = 2` norms and to the localized
source sphere integrated modal norm on the common archived interval. It is
not a claim of uniform 1% late time accuracy.

For the flat sequence, `0.0313%` is the largest **central cumulative
extrapolant residual**. It is not a resolved-accuracy estimate. The directly
observed medium-to-fine changes and the propagated fine-grid estimates are
reported beside every central value. Across the cumulative extrapolant
comparisons these ranges are:

| quantity | range |
|---|---:|
| central extrapolant residual | 0.0109--0.0313% |
| directly observed medium to fine change | 0.0190 to 0.4092% |
| propagated Richardson fine grid estimate | 0.00914 to 0.1189% |

The `0.409%` value is the observed refinement change. The `0.119%` value is a
separate propagated Richardson estimate. Neither value is a relabeling of
the other. Both numerical scales are below the 1% acceptance scale, but they
do not resolve the central residual as an accuracy measurement. No `L/M = 1280`
simulation was run because the two extrapolants agree within 1% in the
declared cumulative test and in the independent localized-source norm.

## Frozen physical contract and archives

Every finite-`L` case and its Schwarzschild reference uses `M = 1`, the
minimal gauge, the same physical datum or source, the same compactification
family, height normalization at `r/M = 4`, and analytic retarded time
`U = tau - q`. The flat sequence uses the same time-symmetric `ell = 2`
areal bump centered at `r/M = 4` with half-width `1.5M`. The localized-source
sequence uses vanishing initial field and velocity and one fixed normalized
source:

```text
r_source/M = 6
radial half-width/M = 0.75
time center/M = 30
time half-width/M = 2
angular concentration = 64
```

Source width is not varied in the regulator comparison. Historical width
variation is retained as physical source dependence, never as numerical
error. All 36 raw archives were produced from the clean simulation-only
commit `2460d976fd023f7bcae892d760436248d32d0290`; analysis does not modify
them. The physical-contract hashes and provenance are in
[`manifest.json`](../results/regulator_production_v3/manifest.json).

The flat refinement ladder is `(N, dt/M) = (384, 0.005), (512, 0.00375),
(768, 0.0025)`. The localized-source ladder is `(N_r, dt/M, ell_max) =
(1024, 0.001, 42), (1536, 1/1500, 46), (2048, 0.0005, 50)`.

## Direct pure ell equals 2 fixed data comparison

For every window,

```text
E2(L) = ||W_L - W_Schw||_2 / ||W_Schw||_2.
```

No relative time translation is fitted. `Einf` is normalized by the maximum
Schwarzschild amplitude in the same window; amplitude and phase are computed
from the zero-lag analytic-signal overlap.

### Cumulative windows

| L/M | U/M window | E2 | Einf | amplitude difference | phase difference |
|---:|---:|---:|---:|---:|---:|
| 320 | 0--40 | 2.7161% | 2.8160% | 0.0200% | -0.02649 rad |
| 320 | 0--80 | 2.7154% | 2.8160% | 0.1173% | -0.02840 rad |
| 320 | 0--160 | 2.7154% | 2.8160% | 0.1496% | -0.02806 rad |
| 640 | 0--40 | 1.3406% | 1.3937% | 0.0011% | -0.01307 rad |
| 640 | 0--80 | 1.3402% | 1.3937% | 0.0682% | -0.01400 rad |
| 640 | 0--160 | 1.3402% | 1.3937% | 0.0837% | -0.01382 rad |

These thresholds apply only to the pure `ell = 2` fixed data sequence. Using
`E2 +` the case specific conservative numerical margin, direct 5%
agreement first occurs at `L/M = 320`, direct 2% agreement first occurs at
`L/M = 640`, and direct 1% agreement is not attained through 640. They do not
apply to the localized source values reported below.

For the representative endpoints, the largest cumulative medium-to-fine
paired change is `0.1478%` at `L/M = 320` and `0.04648%` at `L/M = 640`;
the corresponding Richardson estimates are `0.01384%` and `0.00347%`.

### Disjoint windows and late-time limitation

The disjoint `40--80M` and `80--160M` norms are reported separately because
the cumulative norms are dominated by the prompt waveform.

| L/M | disjoint window | E2 | observed refinement change | status |
|---:|---:|---:|---:|---|
| 320 | 40--80 | 2.4145% | 1.2126% | diagnostic |
| 640 | 40--80 | 1.1547% | 0.4380% | diagnostic |
| 320 | 80--160 | 5.5766% | 145.16% | diagnostic |
| 640 | 80--160 | 7.0146% | 43.48% | diagnostic |

The nested-extrapolant central residuals are `0.0957--0.2105%` on `40--80M`,
but their observed refinement changes are `1.016--3.144%`. On `80--160M`,
the central residuals span `0.852--24.616%` and refinement dominates. These
disjoint results are therefore diagnostics, not resolved accuracy claims.
The analysis supports no uniform late-time 1% statement.

Complete values are in
[`flat_waveform_metrics.csv`](../results/regulator_production_v3/tables/flat_waveform_metrics.csv),
[`flat_numerical_errors.csv`](../results/regulator_production_v3/tables/flat_numerical_errors.csv),
and
[`flat_successive_L.csv`](../results/regulator_production_v3/tables/flat_successive_L.csv).

## Tested expansion and flat nested extrapolants

The requested expansion is tested, not assumed, with

```text
W_inf^(L) = (W_L - 6 W_2L + 8 W_4L) / 3
```

at base lengths 80 and 160. All nine cumulative central comparisons are
below 1%. Their observed refinement changes and propagated fine-grid scales
are separately recorded in
[`flat_extrapolant_comparisons.csv`](../results/regulator_production_v3/tables/flat_extrapolant_comparisons.csv).
The disjoint rows in the same table are explicitly marked diagnostic where
refinement is not subdominant.

## Localized source extrapolation on the common archived interval

The waveform extrapolation is applied directly to the existing compact modal
archives on their common interval. The primary norm uses Parseval
orthogonality,

```text
||W||^2 = integral dU sum_(ell,m) |u_ellm(U)|^2,
```

at the outer boundary on the common fixed `U` grid, from the first
nonnegative stored sample `U/M = 0.000411` through the common archive endpoint
`57.2274`. This is the common archived interval, not a complete late time
signal. No time translation is fitted.

| comparison | sphere-integrated E2 | observed medium-to-fine change |
|---|---:|---:|
| direct L/M = 320 vs Schwarzschild | 5.26710% | 0.0000110% |
| direct L/M = 640 vs Schwarzschild | 2.62569% | 0.0000374% |
| W_inf^(80) vs Schwarzschild | 0.058378% | 0.0000278% |
| W_inf^(160) vs Schwarzschild | 0.006720% | 0.0000972% |
| W_inf^(80) vs W_inf^(160) | 0.051706% | 0.0001012% |

The archived modal pipeline reproduces the `5.26%`, `2.62%`, and `0.052%`
values independently. The sphere integrated norm is the primary result. The `gamma = 0` and `gamma = pi` caustic directions are
secondary diagnostics only; the extrapolant differences there are `0.0186%`
and `0.1184%`, respectively.

See
[`localized_source_waveform_metrics.csv`](../results/regulator_production_v3/tables/localized_source_waveform_metrics.csv),
[`localized_source_extrapolant_comparisons.csv`](../results/regulator_production_v3/tables/localized_source_extrapolant_comparisons.csv),
and
[`localized_source_direction_diagnostics.csv`](../results/regulator_production_v3/tables/localized_source_direction_diagnostics.csv).

## Caustic timing: deterministic sensitivities

The primary `D1` estimator is the matched-template lag. Five effects remain
separate:

1. PDE discretization from the three-level paired refinement;
2. estimator sensitivity from matched-template versus analytic-envelope
   arrivals;
3. fixed-window sensitivity;
4. analysis-cadence sensitivity from `0.0005M`, `0.001M`, and `0.002M` grids;
5. source-width dependence, classified as physical dependence and available
   only in the historical `L/M = 80` width study.

The combined fixed-source sensitivity is the conservative linear sum of the
first four absolute sensitivities. It is a deterministic sensitivity bound,
not a standard deviation, confidence interval, or other statistical error.
Source-width dependence is not included in that fixed-source target test.

| L/M | observer | D1/M | PDE | estimator | window | cadence | combined | target | met? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 320 | r=8M | 0.00412123 | 6.80e-8 | 3.24e-4 | 2.52e-5 | 3.04e-7 | 3.49e-4 | 1e-4 | no |
| 320 | r=12M | 0.00390526 | 7.86e-7 | 7.83e-4 | 1.59e-5 | 4.47e-8 | 8.00e-4 | 1e-4 | no |
| 320 | outer | 0.06736804 | 3.83e-9 | 4.03e-3 | 2.45e-4 | 2.21e-6 | 4.28e-3 | 5e-4 | no |
| 640 | outer | 0.03280627 | 1.04e-8 | 2.11e-3 | 1.27e-4 | 1.56e-6 | 2.24e-3 | 5e-4 | no |

Thus the PDE-discretization components alone meet the requested values, but
the requested **combined timing-sensitivity targets do not**. Both local
`L/M = 640` timings remain diagnostic because `|D1|` is below three times
the combined deterministic sensitivity. The `D1` scaling curves are
presented only as consistency evidence for the expected local and outer
orders. They are not precision coefficient measurements and do not provide
an independent regulator claim. The consistency guides use inverse squared
deterministic sensitivity scales only as numerical weights; their residual
sums are not chi squared statistics.

See
[`D1_measurements.csv`](../results/regulator_production_v3/tables/D1_measurements.csv),
[`D1_estimator_window_sensitivity.csv`](../results/regulator_production_v3/tables/D1_estimator_window_sensitivity.csv),
and
[`D1_scaling_fits.csv`](../results/regulator_production_v3/tables/D1_scaling_fits.csv).

## Phase cleanup

All six `L/M = 12` generic-angle phase pairs are excluded from quantitative
phase analysis. At least one extracted pulse in each pair differs from its
null-ray arrival by more than the declared `1M` tolerance; the largest
residual is `15.666M`. No corrected phase value is emitted. The audit is
[`L12_phase_cleanup.csv`](../results/regulator_production_v3/tables/L12_phase_cleanup.csv).

## Paper-ready artifacts

Raster previews and vector PDFs are regenerated together:

- [`flat_waveform_sequence.pdf`](../results/regulator_production_v3/flat_waveform_sequence.pdf)
- [`flat_window_errors.pdf`](../results/regulator_production_v3/flat_window_errors.pdf)
- [`nested_extrapolants.pdf`](../results/regulator_production_v3/nested_extrapolants.pdf)
- [`localized_source_regulator.pdf`](../results/regulator_production_v3/localized_source_regulator.pdf)
- [`D1_error_separation.pdf`](../results/regulator_production_v3/D1_error_separation.pdf)
- [`D1_scaling.pdf`](../results/regulator_production_v3/D1_scaling.pdf)
- [`L12_phase_exclusion.pdf`](../results/regulator_production_v3/L12_phase_exclusion.pdf)

Booktabs-ready tables are
[`paper_flat_windows.tex`](../results/regulator_production_v3/tables/paper_flat_windows.tex),
[`paper_flat_extrapolants.tex`](../results/regulator_production_v3/tables/paper_flat_extrapolants.tex),
[`paper_localized_source.tex`](../results/regulator_production_v3/tables/paper_localized_source.tex),
and
[`paper_timing_sensitivities.tex`](../results/regulator_production_v3/tables/paper_timing_sensitivities.tex).

## Reproduction

```text
python -m black_hole.regulator_analysis --output-dir results/regulator_production_v3
python -m black_hole.regulator_manifest --output-dir results/regulator_production_v3
python -m black_hole.regulator_manifest --output-dir results/regulator_production_v3 --verify
```

The manifest canonicalizes CRLF/CR to LF before hashing UTF-8 text while
also retaining byte hashes. It records distinct simulation and analysis
commits, every case command, the physical-contract hash, and the raw inputs
for each derived artifact.
