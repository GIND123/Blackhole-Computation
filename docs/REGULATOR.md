# Artificial-cosmology regulator test

## Scope and decision

This production stage tests whether Schwarzschild--de Sitter (SdS) can be used
as an artificial-cosmology regulator for fixed-time, asymptotically flat
waveform physics. It implements the requested `L/M = 320, 640` extensions of
both the minimal-gauge flat-limit sequence and the fixed localized-source
calculation.

The regulator test succeeds at the requested 1% extrapolated-waveform level.
Both nested extrapolants agree with the directly evolved Schwarzschild
waveform and with each other by at most `0.0314%` in relative L2 norm over all
three predeclared windows. Their largest propagated fine-grid numerical error
is `0.119%`, well below the 1% acceptance scale. Therefore `L/M = 1280` was not
run.

Direct agreement is less stringent: the smallest length giving 5% agreement
is `L/M = 320`, the smallest giving 2% is `L/M = 640`, and direct 1% agreement
is not attained through `L/M = 640`. These thresholds require the measured
E2 plus the case-specific conservative numerical margin to pass.

## Frozen physical contract

Every finite-L member and its Schwarzschild reference uses `M = 1`, the
minimal gauge, the same physical datum or source, the same compactification
family, the same height normalization at `r/M = 4`, and the analytic retarded
time `U = tau - q`. The flat sequence uses the same time-symmetric `ell = 2`
areal bump centered at `r/M = 4` with half-width `1.5M`. The source sequence
uses vanishing initial field and velocity and one fixed normalized source:

```text
r_source/M = 6
radial half-width/M = 0.75
time center/M = 30
time half-width/M = 2
angular concentration = 64
```

Source width is not varied in the regulator comparison. The historical width
sequence is retained only as a physical source-dependence diagnostic and is
explicitly excluded from numerical uncertainty.

The machine-readable contract hashes and archive provenance are in
[`manifest.json`](../results/regulator_production_v3/manifest.json).

## Simulation and refinement design

All 36 raw archives were produced afresh from simulation-only commit
`2460d976fd023f7bcae892d760436248d32d0290`. No raw archive is overwritten or
modified by analysis.

The boundary-waveform ladder is:

| level | Chebyshev modes | timestep/M | signal cadence/M |
|---|---:|---:|---:|
| coarse | 384 | 0.005 | 0.03 |
| medium | 512 | 0.00375 | 0.03 |
| fine | 768 | 0.0025 | 0.03 |

It covers Schwarzschild and `L/M = 20, 40, 80, 160, 320, 640` through
`U/M = 160`. The fixed-source ladder is:

| level | radial points | timestep/M | ell_max |
|---|---:|---:|---:|
| coarse | 1024 | 0.001 | 42 |
| medium | 1536 | 1/1500 | 46 |
| fine | 2048 | 0.0005 | 50 |

It covers Schwarzschild and `L/M = 80, 160, 320, 640`. All timing levels are
interpolated to one fixed `0.001M` analysis grid before arrival extraction, so
output sampling is not conflated with PDE discretization.

## Direct fixed-window waveform comparison

For each predeclared window,

```text
E2(L) = ||W_L - W_Schw||_2 / ||W_Schw||_2.
```

No relative time translation is fitted. Amplitude and phase come from the
zero-lag complex analytic-signal overlap. `Einf` is normalized by the maximum
absolute Schwarzschild signal in the same window.

| L/M | U/M window | E2 | Einf | amplitude difference | phase difference |
|---:|---:|---:|---:|---:|---:|
| 320 | 0--40 | 2.7161% | 2.8160% | 0.0200% | -0.02649 rad |
| 320 | 0--80 | 2.7154% | 2.8160% | 0.1173% | -0.02840 rad |
| 320 | 0--160 | 2.7154% | 2.8160% | 0.1496% | -0.02806 rad |
| 640 | 0--40 | 1.3406% | 1.3937% | 0.0011% | -0.01307 rad |
| 640 | 0--80 | 1.3402% | 1.3937% | 0.0682% | -0.01400 rad |
| 640 | 0--160 | 1.3402% | 1.3937% | 0.0837% | -0.01382 rad |

The complete table, including every smaller L and a case-specific numerical
margin for every point, is
[`flat_waveform_metrics.csv`](../results/regulator_production_v3/tables/flat_waveform_metrics.csv).
Successive-L norm, maximum norm, amplitude, and phase differences are in
[`flat_successive_L.csv`](../results/regulator_production_v3/tables/flat_successive_L.csv).

The raw medium-to-fine boundary error at the representative endpoints stays
below 0.2% in every window. Its worst value is `0.1478%` for `L/M = 320` on
`0 <= U/M <= 160`; the corresponding Richardson fine-grid estimate is
`0.01384%`. For `L/M = 640`, the worst values are `0.04648%` and `0.00347%`.
Thus the conservative check meets 0.2%, and the estimated fine error meets the
preferred 0.1% target.

## Tested expansion and nested extrapolants

The expansion is tested with exactly the requested combinations

```text
W_inf^(L) = (W_L - 6 W_2L + 8 W_4L) / 3,
```

for base lengths 80 and 160. The comparison is:

| comparison | E2 range over the three windows | propagated numerical E2 range |
|---|---:|---:|
| W_inf^(80) vs Schwarzschild | 0.0217--0.0222% | 0.0271--0.0777% |
| W_inf^(160) vs Schwarzschild | 0.0109--0.0265% | 0.00914--0.0413% |
| W_inf^(80) vs W_inf^(160) | 0.0186--0.0314% | 0.0362--0.1189% |

All nine comparisons are below 1%, and all nine propagated numerical bounds
are below 0.2%. The detailed L2, Linf, amplitude, phase, and refinement fields
are in
[`flat_extrapolant_comparisons.csv`](../results/regulator_production_v3/tables/flat_extrapolant_comparisons.csv).

## Caustic timing and separated uncertainties

The primary `D1` estimator is the matched-template lag because it converges
cleanly across the three PDE levels. The tapered analytic-envelope maximum is
retained as an independent estimator choice, not averaged into the central
value. The uncertainty table keeps three separate case-specific terms:

1. PDE discretization from the three-level paired SdS-minus-Schwarzschild
   refinement;
2. estimator sensitivity, the matched-template/envelope difference; and
3. fixed-window sensitivity under inset, expansion, and left/right shifts.

Their quadrature is used only for fit weights and for the explicit resolution
test. Source-width dependence is a separate physical dependence and is never
included in this uncertainty.

| L/M | observer | D1/M | discretization/M | estimator/M | window/M | status |
|---:|---|---:|---:|---:|---:|---|
| 320 | r=8M | 0.00412123 | 6.80e-8 | 3.24e-4 | 2.52e-5 | quantitative |
| 320 | r=12M | 0.00390526 | 7.86e-7 | 7.83e-4 | 1.59e-5 | quantitative |
| 320 | outer | 0.06736804 | 3.83e-9 | 4.03e-3 | 2.45e-4 | quantitative |
| 640 | r=8M | 0.00101275 | 1.74e-7 | 4.09e-4 | 1.37e-5 | diagnostic |
| 640 | r=12M | 0.00096203 | 1.76e-6 | 3.45e-4 | 9.60e-6 | diagnostic |
| 640 | outer | 0.03280627 | 1.04e-8 | 2.11e-3 | 1.27e-4 | quantitative |

The requested numerical targets are met: both local `L/M = 320` errors are
below `1e-4M`, and both outer `L/M = 320, 640` errors are below `5e-4M`.
The two local `L/M = 640` values are deliberately diagnostic: their signal is
less than three times the combined timing uncertainty, even though their PDE
refinement differences are small.

The local weighted fits therefore use `L/M = 80, 160, 320` and explicitly
exclude diagnostic `L/M = 640`. The outer fit uses all four lengths. Every fit
point has its own combined error assembled from the three separately recorded
terms. See
[`D1_measurements.csv`](../results/regulator_production_v3/tables/D1_measurements.csv),
[`D1_numerical_errors.csv`](../results/regulator_production_v3/tables/D1_numerical_errors.csv),
and
[`D1_scaling_fits.csv`](../results/regulator_production_v3/tables/D1_scaling_fits.csv).

## Phase cleanup

All six `L/M = 12` generic-angle phase pairs are excluded from quantitative
phase analysis. For every pair, at least one extracted pulse differs from its
null-ray arrival by more than the declared `1M` consistency tolerance. The
largest residual is `15.666M`. No corrected phase value is emitted for an
excluded pair. The complete audit is
[`L12_phase_cleanup.csv`](../results/regulator_production_v3/tables/L12_phase_cleanup.csv).

## Figures and reproduction

- [`flat_waveform_sequence.png`](../results/regulator_production_v3/flat_waveform_sequence.png)
- [`flat_window_errors.png`](../results/regulator_production_v3/flat_window_errors.png)
- [`nested_extrapolants.png`](../results/regulator_production_v3/nested_extrapolants.png)
- [`D1_scaling.png`](../results/regulator_production_v3/D1_scaling.png)
- [`D1_error_separation.png`](../results/regulator_production_v3/D1_error_separation.png)
- [`L12_phase_exclusion.png`](../results/regulator_production_v3/L12_phase_exclusion.png)

List simulation cases:

```text
python -m black_hole.regulator_suite
```

Run one case into a fresh output directory:

```text
python -m black_hole.regulator_suite flat_sds_L640_fine --output-dir <new-directory>
```

Generate the analysis and verify its manifest:

```text
python -m black_hole.regulator_analysis --output-dir results/regulator_production_v3
python -m black_hole.regulator_manifest --output-dir results/regulator_production_v3
python -m black_hole.regulator_manifest --output-dir results/regulator_production_v3 --verify
```

The manifest hashes UTF-8 text after canonical CRLF/CR-to-LF conversion and
also records byte hashes. It records distinct simulation and analysis commits,
the exact command and physical-contract hash for every archive, and the raw
inputs for every derived artifact.
