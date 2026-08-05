# Normalized localized retarded response and caustic echoes

## Scope

This report answers one question. How does a positive cosmological constant change photon sphere caustic echoes near the black hole, and how does that differ from propagation to the cosmological horizon?

The calculation supports a sharp answer at fixed normalized source width. Local delay shifts approach Schwarzschild approximately as `(M/L)^2`. The outer delay shift is larger and approaches more slowly, approximately between `M/L` and a steeper finite range correction. The measured powers over `L/M = 20, 40, 80, 160` are `2.066` at `r = 8M`, `2.065` at `r = 12M`, and `1.304` at the outer boundary.

The source width study does not establish a point source field limit. The correct name for the result is a normalized localized retarded response. All background comparisons use the same narrow source, so the fixed width local versus outer conclusion remains meaningful. The width dependence is archived and reported rather than hidden.

## Geometry and analytic prediction

The metric function is

```text
f(r) = 1 minus 2M/r minus r^2/L^2
```

The photon orbit remains at `r = 3M`. Its frequency and Lyapunov exponent are

```text
Omega_ph = lambda_ph = sqrt(1 minus 27 M^2/L^2) / (3 sqrt(3) M)
```

The geometric half orbit time is

```text
T_half = pi / Omega_ph
```

Thus the local clock correction begins at order `(M/L)^2`. The eikonal field amplitude reference is `exp(minus pi/2) = 0.2078796`, and the simple caustic phase reference is minus `pi/2`. These are comparison values, not fixed laws imposed on the finite width numerical result.

## Covariantly normalized source

The source is centered at Killing time `30M`, areal radius `6M`, and equatorial angle zero. It is written as a temporal factor, a radial factor, and a normalized angular factor. The factors obey

```text
integral T dt = 1
integral R r^2 dr = 1
integral Omega dOmega = 1
integral sqrt(minus g) S d4x = A
```

Three width scales are used. Their radial half widths are `1.5M`, `1.05M`, and `0.75M`. Their temporal half widths are `4M`, `2.8M`, and `2M`. Their angular widths are `0.25`, `0.175`, and `0.125` radians.

The minimum angular cutoffs for omitted source power below `1e-10` are `19`, `27`, and `38`. Further cutoff ladders are `(19, 23, 27)`, `(27, 31, 35)`, and `(38, 42, 46)`.

The weak source test uses constant, linear, and quadratic smooth functions. The exact CSV values are reproduced here.

```csv
width_scale,test_function,integral,delta_limit,absolute_error
1.0,constant,1.0000000000040983,1.0,4.098277273101303e-12
1.0,linear,37.05492482324065,37.0,0.05492482324064696
1.0,quadratic,941.181380691794,937.0,4.181380691793947
0.7,constant,1.0000000000040328,1.0,4.032774114648419e-12
0.7,linear,37.027201751042604,37.0,0.027201751042603917
0.7,quadratic,939.0494182177401,937.0,2.04941821774014
0.5,constant,1.0000000000039877,1.0,3.987699059848637e-12
0.5,linear,37.013948245488685,37.0,0.013948245488684563
0.5,quadratic,938.0457560230874,937.0,1.0457560230873924
```

The exact source validation data are in `results/green_function/tables/normalized_source_weak_convergence.csv`.

## Local pulse estimators

Every background uses identical pulse windows and identical tapers. Two independent estimators are applied inside each local window.

1. A tapered analytic signal estimator finds the local envelope maximum with interpolation.

2. A complex matched template fit varies amplitude, time shift, constant background, and linear background.

The reported pulse time is the midpoint of the two estimates. Their full difference is recorded as the estimator systematic. Half the output cadence is recorded separately. The production output cadence is `0.001M`, and interpolation is supported by the temporal ladder.

Each pulse row contains arrival time, matched amplitude, integrated field energy, integrated flux energy, delay, amplitude ratio, energy ratio, local phase, cadence uncertainty, and estimator systematic. The sequence is the direct pulse plus three caustic echoes.

## Reanalysis of existing archives

The older archives have cadence `0.1M`, so their local estimator uncertainties prevent subcadence claims. The reanalysis gives the following first delay shifts in units of `M`.

```csv
L_over_M,r8,r12,outer
20,1.3369294495,1.2601732808,2.1704442552
40,0.2913877911,0.2568550461,0.6673344009
80,0.0656400361,0.0580914894,0.2669277629
160,0.0175664325,0.0157874894,0.1200504951
```

The corresponding uncertainty table is in `results/green_function/tables/local_delay_scaling.csv`. The `L=80` and `160` local shifts are unresolved at the old cadence. This is why the high cadence production suite was necessary.

The reanalysis products include identical window local pulse tables, local phase tables, exact ray timing, generic angle reconstruction, weak source convergence, fixed radius versus outer scaling, ray residuals, clock collapse, and damping and phase figures.

## Production suite

The selected narrow production setting is radial resolution `1536`, timestep `0.001M`, signal cadence `0.001M`, and angular cutoff `42`. Every signal timestep is saved. The run ends at `110M`.

The main backgrounds are Schwarzschild and `L/M = 20, 40, 80, 160`. The stronger case `L/M = 12` is also included. For `L=12`, the black hole horizon is `2.0607756558M`, the cosmological horizon is `10.8361577003M`, and the full narrow source support is `5.25M` through `6.75M`. The `r=8M` observer is safe. The invalid `r=12M` observer is excluded.

Three widths are run on Schwarzschild, `L=20`, and `L=80`. The narrow width is run through the complete length sequence. The small length analysis includes full and monopole subtracted signals.

## Observable convergence

The spatial ladder is `N = 768, 1024, 1536, 2048`. The temporal ladder is `0.004M`, `0.002M`, and `0.001M`. Each source width has three angular cutoffs.

For `N=1536` against `2048`, the sphere integrated relative waveform error is `3.573e-6`. The largest relative amplitude error is `1.729e-5`. The largest phase error is `1.971e-4` radians. The largest arrival difference is `9.262e-4 M`.

For timestep `0.002M` against `0.001M`, the sphere integrated error is `1.966e-7`, the amplitude error is `6.515e-6`, and the phase error is `1.747e-4` radians. Arrival interpolation reaches a nonmonotone floor near `0.0026M`. No convergence order is fitted to that floor.

For the narrow source, cutoff `42` against `46` gives sphere integrated error `2.274e-9`, amplitude error `6.763e-10`, phase error `2.787e-8` radians, and arrival difference `6.696e-8 M`.

All fixed source waveform, amplitude, and phase targets are met. The full table is `results/caustic_production/tables/observable_convergence.csv`.

## Source width sensitivity

Absolute pulse waveforms change substantially as the normalized source narrows. On Schwarzschild, width `0.7` versus `0.5` changes the generic `gamma = pi/2` waveform norm by `0.272`. At `L=80`, the corresponding value is `0.260`. This is not a numerical failure. It is finite regularization dependence of a singular response.

The primary finite cosmological constant delay shift is more stable because it is a within run delay and a same width background difference. Medium versus narrow changes the local shift by `0.00803M` at `L=20`, and by `0.00236M` at `L=80` for `r=8M`. Outer sensitivities are `0.0501M` and `0.0201M`.

The full data are in `results/caustic_production/tables/source_width_delay_sensitivity.csv`. Absolute pulse error budgets include a conservative source width sensitivity column. Fixed source and source inclusive totals are both retained.

## Local versus outer scaling

The narrow production first delay shifts are

```csv
L_over_M,r8,r12,outer
20,1.1800413776,1.1174635993,2.0955243972
40,0.2713284854,0.2560657452,0.7395256543
80,0.0659039400,0.0620446053,0.3060844584
160,0.0159901628,0.0151963858,0.1383378931
```

The fitted powers over all four lengths are `2.066`, `2.065`, and `1.304`. Over `L=20` through `80`, they are `2.081`, `2.085`, and `1.388`. Both candidate powers are plotted in each panel of `results/caustic_production/production_timing_scaling.png`.

The `L=160` local correction is shown but is excluded from the preferred precision fit when transferred source width sensitivity is included. The all length fit remains available as a descriptive comparison in `production_scaling_fits.csv`.

## Exact null rays

The separate ray tracer solves the actual source radius, observer radius, angular separation, turning point, impact parameter, and winding. It uses the same retarded clock normalization as the simulation. It selects an outward direct branch when geometrically available and an initially inward branch otherwise.

For Schwarzschild and `L=20` through `160`, simulation minus ray residuals stay within about `0.44M`. At `L=12`, the largest residual is `3.38M`, consistent with strong finite width and cosmological transition effects. These residuals are measured wave effects, not replaced by the asymptotic photon orbit interval.

The exact values are in `production_null_rays.csv` and `production_generic_angles.csv`. The residual figure is `production_ray_residuals.png`.

## Angular structure and phase

Waveforms are reconstructed on both caustic axes and at `gamma = pi/3` and `pi/2`. The axes display the degenerate twofold structure. Generic angles test the fourfold Maslov cycle through local complex comparisons.

For Schwarzschild and `L=20` through `160`, generic consecutive pulse phases remain near minus `pi/2`, with finite width and finite frequency residuals. The `L=12` sequence is strongly deformed and includes a phase reversal in one late pair. Generic amplitude ratios vary by pulse and are not asserted to equal a fixed damping constant.

The rescaled clock is

```text
U_hat = Omega_ph(L) times [U minus U_ref(L)]
```

The residual collapse is plotted in `production_clock_collapse.png`. Damping and phase are compared with `exp(minus pi/2)` and minus `pi/2` in `production_damping_phase.png`.

## Cross code validation

Finite difference and Dedalus evolve the same narrow source through the direct pulse plus two echoes for Schwarzschild and `L=80`. The timestep and signal cadence are `0.002M`. The angular cutoff is `42`. Finite difference uses converged radial resolution `768`. Dedalus uses Chebyshev resolution `512`.

For Schwarzschild, the sphere integrated error is `5.33e-5`, the generic angle error is `3.56e-5`, the maximum arrival error is `1.40e-4 M`, the amplitude error is `8.16e-5`, and the phase error is `1.25e-4` radians.

For `L=80`, the sphere integrated error is `5.58e-4`, the generic angle error is `7.25e-5`, the maximum arrival error is `2.22e-4 M`, the amplitude error is `8.22e-5`, and the phase error is `1.99e-4` radians.

Every required cross code target is met, including the preferred `1e-3` sphere integrated norm.

## Physical monopole conversion

The scalar field convention is

```text
Phi = sum_lm u_lm Y_lm / r
```

For the monopole at the cosmological horizon,

```text
Phi_00 = Y_00 u_00 / r_c
Y_00 = 1 / sqrt(4 pi)
```

The factor `Y_00` is included in the production monopole subtraction. No conversion uses `u_00/r_c` alone.

## Reproduction

List every production case.

```powershell
python -m black_hole.production_suite
```

Run a named finite difference case.

```powershell
python -m black_hole.production_suite sds_L80 --output-dir results/caustic_production
```

Run a Dedalus cross case in the pinned WSL environment.

```bash
python -m black_hole.production_suite cross_sds_L80 --output-dir results/caustic_production --backend dedalus
```

Rebuild convergence and cross code products.

```powershell
python -m black_hole.production_analysis --output-dir results/caustic_production --include-cross-code
```

Rebuild all final tables and figures.

```powershell
python -m black_hole.production_report --output-dir results/caustic_production
```

## Reproducibility

The native environment uses Python `3.12.10`, NumPy `2.5.1`, SciPy `1.18.0`, Microsoft compiler `19.43`, and Windows `11 build 26200`.

The independent spectral environment uses Python `3.14.6`, Dedalus `3.0.5`, NumPy `2.5.1`, SciPy `1.18.0`, mpi4py `4.1.2`, Open MPI `5.0.10`, GCC `14.3.0`, and WSL Linux `6.6.87.2`.

Every production archive stores Python, package, compiler, operating system, MPI, Git commit, and tracked worktree state. JSON summaries are strict JSON and map unresolved nonfinite values to `null`.

The archive audit retains two historical dirty state records. `radial_N768` was saved while the compact error bound metadata change was pending at revision `724724c`. The de Sitter Dedalus archive at revision `d1e14ad` saw tracked line ending differences under WSL. Their numerical arrays are finite and preserved exactly. The restamp note is stored inside each affected archive, and `archive_audit.json` reports both records.

## Data products

1. `results/green_function` contains the existing archive reanalysis and corrected source validation.

2. `results/caustic_production/pilots/raw` contains radial, temporal, angular, and source width archives.

3. `results/caustic_production/raw` contains Schwarzschild and five de Sitter production archives.

4. `results/caustic_production/cross_code` contains finite difference and Dedalus archives.

5. `results/caustic_production/tables` contains every pulse observable, exact ray timing, phase, scaling, source width sensitivity, convergence, cross code comparison, monopole subtraction, and full error budget.

6. `results/caustic_production/production_summary.json` and `production_analysis.json` contain strict machine readable summaries.

## Conclusion

At fixed normalized source width, positive cosmological constant preserves the local caustic organization while stretching its clock at order `(M/L)^2`. Propagation to the moving cosmological horizon produces a larger and more slowly converging deformation. Exact rays explain the geometric clock, while residuals isolate finite width and wave effects. Generic observers recover the local Maslov phase pattern except in the strongly deformed `L=12` regime.

The calculation is numerically converged at fixed source and independently validated with Dedalus. The source width study does not establish a point source field limit, so no point source Green function claim or universal fixed damping law is made.
