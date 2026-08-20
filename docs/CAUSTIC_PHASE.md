# The phase an echo acquires at the caustic

> **Status: diagnostic, not a paper claim.** The measurement below is
> reproducible and its estimator is validated, but the interpretation of the
> measured angle as a partial caustic rotation is not established. The
> completed passage has the known Hilbert transform structure, whereas the
> archived interval ends before the second passage and the measurement has not
> been tested against source width. It stays a diagnostic until a longer and
> width-resolved study settles the interpretation, and it is not to be quoted
> as a result of the paper.


## What is measured and why

A localized pulse released outside a black hole splits. Part of it reaches the
outer boundary directly. Part of it is bent around the hole and refocuses on
the axis behind it. The refocused arrival is what the visualizations show as
the antipodal caustic.

The earlier description of that arrival called it a sign reversal. That is
what the last two frames look like, but it is not the phenomenon. Geometrical
optics fails at a focus: the Jacobian of the ray congruence changes sign, the
amplitude in the transport equation becomes imaginary, and the arriving
profile is rotated in the phase of its analytic signal rather than multiplied
by a number. For Schwarzschild the resulting fourfold structure of the
retarded Green function was identified by Ori and analyzed by Casals and
collaborators, and each caustic passage acts as a Hilbert transform of the
profile (Zenginoglu and Galley, Phys. Rev. D 86, 064030 (2012), Appendix C).

The claim in a figure title should be a measured quantity, so the rotation is
now measured instead of asserted.

## Definition

Let `u_0` be the field in the emitter direction and `z_0 = u_0 + i H[u_0]` its
analytic signal. For a direction `gamma` from the emitter, fit

```text
u_gamma(U) = Re[ A exp(i phi) z_0(U - Delta) ]
           = A cos(phi) u_0(U - Delta) - A sin(phi) H[u_0](U - Delta)
```

for the amplitude `A`, the rotation `phi`, and the delay `Delta`. The three
reference values are

| relation to the direct pulse | `phi` |
| --- | ---: |
| unrotated copy | 0 |
| full Hilbert transform | 90 (either sign) |
| sign reversal | 180 |

`Delta` is a delay between two directions of one evolution. No relative time
translation between backgrounds is fitted anywhere in this measurement.

The fit is reported with the fraction of the variance it explains, so a poor
model cannot be mistaken for a measured phase. The Hilbert transform is taken
on the complete archived trace before the fit window is applied, so the
quadrature partner is not distorted by the window edges; the window is then
applied to data and model alike.

Every direction follows from the archived responses by the spherical addition
theorem, `u(U, gamma) = sum_ell g_ell (2 ell + 1) / (4 pi) R_ell(U)
P_ell(cos gamma)`. No sphere is sampled and no interpolation in angle is
involved, so the measurement is exact for the stored modes.

## Result

At the antipode, over three observers and both backgrounds:

| case | observer | `phi` (degrees) | `A` | variance explained |
| --- | ---: | ---: | ---: | ---: |
| SdS `L/M = 80` | `r = 8M` | 41.57 | 0.463 | 0.941 |
| SdS `L/M = 80` | `r = 12M` | 41.20 | 0.966 | 0.946 |
| SdS `L/M = 80` | `H_c^+`, `r_c = 78.98M` | 43.10 | 2.198 | 0.946 |
| Schwarzschild | `r = 8M` | 42.22 | 0.462 | 0.940 |
| Schwarzschild | `r = 12M` | 41.70 | 0.963 | 0.946 |
| Schwarzschild | future null infinity | 41.92 | 2.525 | 0.952 |

The mean is `41.95` degrees with a full spread of `1.90` degrees. The echo is
therefore neither a copy of the direct pulse nor its negative, and the
rotation is a property of the arrival rather than of the observer or of the
cosmological constant.

The angular scan is what identifies the rotation as a caustic effect. Away
from the axis the arriving wavefront is an almost unrotated copy of the direct
pulse; the rotation appears only in the last few degrees, together with the
amplification.

| `gamma` (degrees) | arrival `U/M` | `phi` (degrees) | `A` |
| ---: | ---: | ---: | ---: |
| 0 | 26.58 | 0.00 | 1.000 |
| 30 | 27.45 | 0.01 | 0.963 |
| 60 | 29.81 | -0.79 | 0.889 |
| 90 | 33.19 | -1.72 | 0.835 |
| 120 | 36.97 | -7.26 | 0.841 |
| 150 | 40.75 | -8.15 | 0.967 |
| 160 | 42.07 | 2.49 | 1.155 |
| 170 | 44.05 | 40.45 | 1.821 |
| 175 | 44.14 | 42.72 | 2.096 |
| 180 | 44.15 | 43.10 | 2.198 |

The largest rotation anywhere at or below `gamma = 150` degrees is `8.5`
degrees, against `43.1` degrees on the axis. The transition width is a few
degrees, which is the angular scale the source itself sets: the emitter has
`kappa = 64`, so its width is `kappa^(-1/2) = 0.125` radians, or about seven
degrees.

The variance explained dips to `0.78` near `gamma = 150` to `160` degrees.
That is expected and is reported rather than hidden: in that range the sweeping
front and the forming focus overlap, and a single rotated copy of one pulse is
not the right model. The rows retained for interpretation are the ones where
the model explains more than ninety percent.

## The arrival is where a null ray puts it

The rotation is a statement about the shape of the echo. Its timing is a
separate check, and it is answered by the ray tracer already in the repository
rather than by the fit. Tracing an inward turning null ray on the same
`L/M = 80` background from the emitter radius `r = 6M` at the source time
centre `t = 30M` out to the cosmological horizon gives

| | null ray | measured | difference |
| --- | ---: | ---: | ---: |
| direct arrival, `gamma = 0` | 26.592 | 26.577 | -0.014M |
| antipodal arrival, `gamma = pi` | 44.393 | 44.146 | -0.247M |
| delay between them | 17.801 | 17.569 | -0.233M |

and the delay returned by the phase fit as a free parameter, `17.72M`, agrees
with the ray delay to `0.08M`. The wrapping ray has impact parameter `6.418M`
and turns at `4.933M`, inside the region controlled by the photon sphere.
Arrivals in this table are envelope maxima of the raw trace, which is why the
direct value differs in its third decimal from the tapered estimator that sets
the archived peak times.

Two independent statements follow. The feature is a genuine refocusing of a
wrapped null congruence and not an artifact of the reconstruction. And the
residual differences, a few tenths of `M` on arrivals of 26 and 44, are the
expected offset between a geometrical optics arrival and the envelope maximum
of a pulse of finite temporal and radial width; the emitter here has
`sigma_t = 2M` and `sigma_r = 0.75M`.

## Sensitivities

| effect | how it is varied | spread in `phi` |
| --- | --- | ---: |
| angular truncation | `ell_max` from 30 to 50 | `6.0e-7` degrees |
| fit window | four displacements and rescalings by `1M` | 1.11 degrees |
| observer and background | three observers, two backgrounds | 1.90 degrees |

The truncation result also settles a separate question about the pictures. The
antipodal sum alternates in sign, so a truncated Legendre series could in
principle ring at the focus and the visible beam could be an artifact of the
cut. It is not: the antipodal peak amplitude changes by less than `1e-5`
relative between `ell_max = 30` and `ell_max = 50`, because the emitter is
band limited by its own angular width and the focus is smoothed by the source
rather than by the truncation. The caustic in the figures is resolved physics.

## What the number does and does not establish

A full caustic passage acts as a Hilbert transform, which is 90 degrees. The
measured 42 degrees is close to half of that, and the observer here sits on
the caustic rather than beyond it, so only part of the shift has accumulated.
That reading is consistent with the geometrical optics picture but is not
established by this measurement alone.

Two limitations are recorded with the number. The archives end at
`U/M = 57.2`, about 13M after the antipodal arrival, so the slowly decaying
quadrature tail is truncated and the fitted rotation is a conservative
estimate of its magnitude. And the second antipodal passage, which would carry
the accumulated shift of a completed crossing, would arrive near `U/M = 77`
and is outside the archived interval. Establishing the full fourfold cycle
would need a longer sourced evolution, which is not part of the frozen
production package.

## Reproduction

```text
python -m black_hole.caustic_phase --output-dir results/caustic_visualizations
```

writes

- [`caustic_phase.json`](../results/caustic_visualizations/caustic_phase.json),
  the summary record;
- [`caustic_phase_scan.csv`](../results/caustic_visualizations/tables/caustic_phase_scan.csv),
  every direction at every observer for both backgrounds;
- [`caustic_phase_truncation.csv`](../results/caustic_visualizations/tables/caustic_phase_truncation.csv);
- [`caustic_phase_windows.csv`](../results/caustic_visualizations/tables/caustic_phase_windows.csv);
- [`caustic_phase_null_rays.csv`](../results/caustic_visualizations/tables/caustic_phase_null_rays.csv).

The inputs are the frozen `v3` localized source archives at the fine level.
Nothing in this measurement modifies them, and no simulation is rerun.
