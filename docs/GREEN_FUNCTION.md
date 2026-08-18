# Normalized localized retarded response and caustic echoes

## Scope

This report records the corrected caustic echo study requested by Professor Zenginoglu. The calculation is a normalized localized retarded response. The source width sequence is a sensitivity study. It does not establish a point source Green function limit.

All final raw archives were generated in a fresh directory from clean Git commit `c0ec6a69c80f69b87c65a51c8e4b77a4760a7af2`. No v1 archive was copied into v2. Postprocessing reads raw NPZ files without saving or modifying them.

The final archive directory is [`results/caustic_production_v2`](../results/caustic_production_v2). The machine readable provenance record is [`manifest.json`](../results/caustic_production_v2/manifest.json).

## Primary observable

The principal observable is

```text
D1(L,w;r_o) = [U1 - U0]_SdS(L,w;r_o) - [U1 - U0]_Schw(w;r_o).
```

Every primary arrival time is the maximum of the tapered analytic signal envelope in its fixed local pulse window. This convention is used without alteration in production, source width, convergence, and cross code calculations.

Matched template timing is not averaged with the primary timing. It is reported separately as an estimator sensitivity check. The estimator sensitivity is the absolute difference between analytic envelope D1 and matched template D1.

The complete primary table is [`production_delay_scaling.csv`](../results/caustic_production_v2/tables/production_delay_scaling.csv).

## Central timing results

The analytic envelope values of D1 in units of M are:

```text
L/M       r_o=8M             r_o=12M            outer
20        1.1651680212       1.1015045109        2.0689221905
40        0.2678177855       0.2520715228        0.7270260932
80        0.0646050126       0.0605111408        0.2991613043
160       0.0153330621       0.0146373771        0.1348276073
```

The narrow width table reproduces these values exactly at every stored digit for L/M equal to 20 and 80. In particular, the earlier disagreement between `1.18004M` and `1.16517M` is removed. Both analyses now give `1.1651680212M` for L/M equal to 20 at r equal to 8M.

## Direct D1 error budget

Errors are evaluated by recomputing the full four arrival time D1 combination at each numerical setting. Arrival errors are not assigned independently to U0 and U1. This preserves cancellation and correlation among the four times.

Representative SdS convergence suites were run independently at L/M equal to 20 and 80. Each suite has its own radial, temporal, and angular comparison paired with the corresponding Schwarzschild comparison. Source width sensitivity is used only at the same L value.

The largest direct D1 convergence differences among the three observers are:

```text
L/M       radial             temporal           angular
20        2.2331e-4 M        7.6058e-4 M        5.6907e-8 M
80        3.1440e-4 M        4.7617e-4 M        6.8694e-8 M
```

The resulting complete primary D1 uncertainty totals are:

```text
L/M       r_o=8M             r_o=12M            outer
20        0.00806893 M       0.00056146 M       0.05007618 M
80        0.00241237 M       0.00273262 M       0.02006741 M
```

All six representative D1 values are resolved against these totals. The L/M equal to 40 and 160 rows are marked `budget_complete=False`. No convergence or width error from another L value is assigned to them.

The detailed records are [`D1_convergence.csv`](../results/caustic_production_v2/tables/D1_convergence.csv), [`D1_convergence.png`](../results/caustic_production_v2/D1_convergence.png), and [`full_error_budget.csv`](../results/caustic_production_v2/tables/full_error_budget.csv).

## Source width sensitivity

The width scale sequence is `1.0`, `0.7`, and `0.5`. Every row uses the analytic envelope timing pipeline. The difference column is measured relative to the narrow `0.5` archive at the same L and observer.

At L/M equal to 20, the intermediate to narrow D1 differences are `0.00803253M`, `0.00047973M`, and `0.05007354M` at r equal to 8M, r equal to 12M, and the outer observer. At L/M equal to 80, they are `0.00236129M`, `0.00271333M`, and `0.02006407M`.

These values quantify finite source sensitivity only. They do not demonstrate convergence to a point source distribution.

The complete table is [`source_width_delay_sensitivity.csv`](../results/caustic_production_v2/tables/source_width_delay_sensitivity.csv).

## Scaling analysis

Fixed radius data are fitted to the motivated local expansion

```text
D_local(L) = a2 (M/L)^2 + a4 (M/L)^4.
```

Using L/M equal to 20 through 160 gives

```text
r_o=8M:     a2 = 415.3041,   a4 = 20307.9010,   RMS residual = 6.2490e-4 M
r_o=12M:    a2 = 390.1443,   a4 = 20185.8611,   RMS residual = 5.9336e-4 M
```

Outer data are fitted to

```text
D_outer(L) = b1 M/L + b2 (M/L)^2.
```

Using L/M equal to 20 through 160 gives `b1 = 17.4456`, `b2 = 478.0696`, and an RMS residual of `0.0062494M`.

Log power fits are retained only as finite interval diagnostics. The outer effective exponent is `1.3100` over L/M equal to 20 through 160. It is not an established asymptotic power.

The coefficients and finite interval diagnostics are in [`production_scaling_fits.csv`](../results/caustic_production_v2/tables/production_scaling_fits.csv). The requested expansion fits are plotted in [`production_timing_scaling.png`](../results/caustic_production_v2/production_timing_scaling.png).

## Phase analysis

Each phase comparison records whether the optimized lag reaches the permitted lag boundary. Boundary saturation is evaluated with a tolerance tied to the local sample cadence. A saturated fit has `phase_resolved=False`; its primary phase fields are null, while its raw diagnostic phase and lag remain available for audit.

All generic phase fits for Schwarzschild and L/M equal to 20, 40, 80, and 160 are unsaturated. Four L/M equal to 12 fits are unresolved. The apparent phase reversal at gamma equal to pi over 2 for pulse pair 1 to 2 has lag `minus 1.9949999M` against a permitted magnitude of `2M`. It is therefore removed from the physical interpretation.

The corrected phase table is [`production_generic_phase.csv`](../results/caustic_production_v2/tables/production_generic_phase.csv). The phase figure is [`production_damping_phase.png`](../results/caustic_production_v2/production_damping_phase.png).

## Cross code validation

Fresh finite difference and Dedalus archives were generated for Schwarzschild and SdS with L/M equal to 80. Primary timing uses the analytic envelope in both backends.

Both calculations carry the same normalized source, modes through
\(\ell_{\max}=42\), observers at \(r=8M\), \(r=12M\), and the outer endpoint,
output spacing \(0.002M\), timestep \(0.002M\), and final bridge time
\(72M\). The finite difference calculation uses 768 uniform radial points,
the eighth order stencil, and classical RK4. The Dedalus calculation uses 512
Chebyshev T modes with a 3/2 dealias factor and RK443. Dedalus evaluates the
source at every Runge Kutta stage through a general function. These are
independently resolved configurations rather than nominally identical grids.

The maximum sphere integrated relative L2 disagreement is `5.5840e-4`. The maximum individual arrival difference is `4.1363e-4M`. The maximum relative analytic envelope amplitude difference is `1.2344e-4`. The maximum resolved phase difference is `2.0072e-4` radians.

The direct D1 backend differences are `7.0207e-5M`, `2.9720e-4M`, and `5.3119e-5M` at r equal to 8M, r equal to 12M, and the outer observer.

The complete records are [`cross_code_observables.csv`](../results/caustic_production_v2/tables/cross_code_observables.csv) and [`cross_code_D1.csv`](../results/caustic_production_v2/tables/cross_code_D1.csv).

## Other figures

The null ray residual comparison is [`production_ray_residuals.png`](../results/caustic_production_v2/production_ray_residuals.png).

The normalized clock collapse comparison is [`production_clock_collapse.png`](../results/caustic_production_v2/production_clock_collapse.png).

The observable convergence summary is [`observable_convergence.png`](../results/caustic_production_v2/observable_convergence.png).

## Reproduction

The runner refuses every existing archive destination. It never silently reuses or overwrites a raw archive. A simulation is first saved with an incomplete suffix and renamed to its final NPZ path only after successful completion.

List all finite difference case names:

```text
python -m black_hole.production_suite
```

Run one finite difference case into a new directory:

```text
python -m black_hole.production_suite sds_L80 --output-dir results/caustic_production_v2
```

Run one Dedalus case after activating a Dedalus 3 environment:

```text
OMP_NUM_THREADS=1 python -m black_hole.production_suite cross_sds_L80 --output-dir results/caustic_production_v2 --backend dedalus
```

Generate convergence and cross code tables:

```text
python -m black_hole.production_analysis --output-dir results/caustic_production_v2 --include-cross-code
```

Generate final tables and figures:

```text
python -m black_hole.production_report --output-dir results/caustic_production_v2
```

Generate and verify the artifact manifest:

```text
python -m black_hole.production_manifest --output-dir results/caustic_production_v2
```

Every archive records Python, NumPy, SciPy, Dedalus when applicable, operating system, compiler, Git commit, and worktree state. The manifest records SHA256 checksums, exact case commands, and the raw input archives for every final table and figure.

## Conclusion

The corrected primary estimator gives a consistent D1 in production and width analyses. The representative direct D1 convergence checks support the L/M equal to 20 and 80 claims without transferring numerical errors across geometries. The local data support the motivated even power expansion over the simulated interval. The outer data support a mixed first and second order expansion, while the effective exponent is only descriptive. The L/M equal to 12 phase reversal claim is withdrawn because the relevant fit is boundary saturated.
