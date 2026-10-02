# Far horizon-supported regulator pilot

This package tests one exterior-supported background at `L/M=80` against the
unchanged Schwarzschild and uniform-SdS controls in
`results/regulator_production_v3`.  The production runs use the same initial
data, resolutions, timesteps, gauge normalization, and waveform windows as the
controls.

## Background

The switch is defined in the Chebyshev endpoint angle

```text
theta = acos(2 rho - 1),   theta_0 = 2 theta_1,
R_1 = 0.9 r_c,             chi = S((theta_0-theta)/(theta_0-theta_1)).
```

For `L/M=80`, `r_c/M=78.98059813`.  The metric is exactly Schwarzschild for
`r/M <= 54.72748594`, transitions over
`54.72748594 < r/M < 71.08253832`, and is exactly SdS from there to the
cosmological horizon.  The transition and outer cap have equal Chebyshev-angle
widths.

## Outer waveform result

The direct observable is the unshifted reduced `ell=2` waveform at the
cosmological horizon compared with the Schwarzschild waveform at scri on
`0 <= U/M <= 80`.  No time translation or amplitude rescaling is fitted.

| observable | uniform SdS | far exterior SdS | relative reduction |
|---|---:|---:|---:|
| raw outer-boundary `E2` | 11.7458% | 9.6288% | 18.02% |
| leading-transfer-corrected `E2` | 1.8522% | 0.6219% | 66.42% |

The raw improvement is resolved but does not meet the predeclared 25%
substantial-improvement criterion.  The medium-to-fine paired change of the far
raw residual is 0.0101 percentage points.

Most of the raw error is the common transfer difference between a finite-area
cosmological horizon and Schwarzschild scri.  The fit-free first-Born
coefficient is obtained from the background potential integrals,

```text
I = integral [ell(ell+1)/r^2 + f'(r)/r] dr,
C = -(I_finite-I_Schwarzschild)/2.
```

For the far profile, `C r_c=3.59122717`.  Applying the leading causal inverse

```text
W_corrected(U) = W_Hc(U) - C integral^U W_Hc(U') dU'
```

gives the second row of the table.  Its coefficient is fixed by the background,
not fitted to the waveform.  The far corrected medium-to-fine paired change is
0.0299 percentage points, so the factor-2.98 advantage over corrected uniform
SdS is resolved.  This correction is only the leading transfer approximation;
an exact frequency-domain transfer function would be its systematic extension.

## Inner ringdown diagnostic

Matrix-pencil fits at the black-hole horizon, over six post-peak windows before
the earliest transition return, give

| background | `M omega_R` | `-M omega_I` |
|---|---:|---:|
| Schwarzschild | 0.4836527 | 0.0967717 |
| uniform SdS, `L/M=80` | 0.4825576 | 0.0966040 |
| far exterior SdS | 0.4836643 | 0.0967847 |

The far real-frequency discrepancy is about `1.2e-5`, roughly 95 times smaller
than the uniform-SdS discrepancy.  Its window sweep remains at a few `1e-5`.
The damping estimate is not refinement-stable and is not a supported
quantitative result.

Characteristics predict returns from the inner edge, transition center, and
outer edge near `120.35M`, `139.75M`, and `191.17M`.  The fine inner residual
begins growing at the expected time, but its amplitude and the final constraint
norm are not monotonically converged.  The return is therefore a timing
diagnostic only.  No late-time claim is made from this package.

## Resolution audit

The three coupled levels are `(N,dt/M)=(384,0.005)`, `(512,0.00375)`, and
`(768,0.0025)`.  They place 13, 17, and 26 collocation points strictly inside
the transition.  Dense reconstruction of the scalar-potential bracket gives
maximum errors `3.86e-2`, `8.63e-3`, and `1.01e-3`, respectively.  The fine
representation preserves positivity and resolves the analytic transition
minimum to about 0.6%.

The final constraint `Linf` values are `1.30e-3`, `3.19e-4`, and `8.91e-4`;
their nonmonotonicity is why the analysis is restricted to the converged
waveform and pre-return diagnostics.

## Reproduction

```text
python -m black_hole.exterior_regulator_suite exterior_sds_L80_coarse \
  --output-dir results/exterior_regulator_far_v1
python -m black_hole.exterior_regulator_suite exterior_sds_L80_medium \
  --output-dir results/exterior_regulator_far_v1
python -m black_hole.exterior_regulator_suite exterior_sds_L80_fine \
  --output-dir results/exterior_regulator_far_v1
python -m black_hole.exterior_regulator_analysis \
  --output-dir results/exterior_regulator_far_v1 --lengths 80
```

The runners refuse existing destinations.  The control package is read only.
