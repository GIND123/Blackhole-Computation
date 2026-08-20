# Caustic diagnostics: is the caustic resolved, and what sets its width

The rendered candidates showed a caustic without ever establishing that the
caustic was resolved. These diagnostics answer that first. They work in
physical areal radius on a symmetric linear scale, cropped to the strong field
region, and they measure the focus rather than illustrate it.

Everything here uses `Phi = u / r`. The emitter sits at `gamma = 0` in the
equatorial plane at `r = 6M`, so the antipodal axis is `gamma = 180` degrees
and the equatorial section carries the whole structure.

## 1. Choosing the snapshot

The snapshot must be chosen from the spatial field, not from an outer observer
time series. The observer series peaks when the focused wavefront arrives at
that observer, which is later than the focus itself and depends on where the
observer sits.

The first attempt maximised `|Phi|` on the antipodal axis over the whole
cropped domain. That picked `r = 2.91M`, which is not a caustic: the radial
cut there falls monotonically outward from the inner boundary, so the search
was selecting the strong field pile up around the photon sphere simply because
it is large.

The criterion now used is

* restrict to the wave zone, `r >= 4M`, outside the photon sphere with margin;
* require the axial profile to have an **interior** maximum in that band, so a
  maximum sitting on either edge means the converging front has not arrived or
  has already passed;
* among the snapshots that qualify, take the one with the largest
  amplification, defined as the axial value divided by the mean of `|Phi|`
  over `60` to `120` degrees at the same radius.

On the archived `2M` cadence sequence this selects `tau = 46.74`, with the
axial maximum at `r = 8.49M` and an amplification of `75`. The amplification
itself peaks at `82.8` slightly further out, at `r = 12.94M`. For comparison,
the outer observer waveform peaks at `U = 44.06M`, which is the arrival, not
the focus.

The behaviour across the sequence shows why the interior test matters:

| `tau/M` | axial max in `r >= 4M` | interior | amplification |
|---:|---|---|---:|
| 42.68 | at the outer edge | no | not defined |
| 44.68 | at the outer edge | no | 10.9 |
| **46.68** | `r = 8.86M` | **yes** | **82.0** |
| **46.74** | `r = 8.49M` | **yes** | **82.8** |
| 48.68 | at the inner edge | no | 77.2 |
| 50.68 | `r = 4.61M` | yes | 22.8 |
| 52.68 | at the inner edge | no | 5.6 |

## 2. What sets the width of the focus

This is the question that decides whether a sharper picture needs a longer
angular sum or a narrower emitter, and it is answered by truncating the
reconstruction rather than by comparing scales.

At the selected focus, reconstructing the same archived snapshot with
progressively shorter sums:

| `ell_max` | peak `|Phi|` | focus FWHM | relative change in peak |
|---:|---:|---:|---:|
| 20 | `1.156773e-02` | `40.170` deg | `7.8e-05` |
| 30 | `1.156863e-02` | `40.170` deg | `1.2e-07` |
| 40 | `1.156863e-02` | `40.170` deg | `4.3e-10` |
| 50 | `1.156863e-02` | `40.170` deg | 0 |

The width does not move at all and the peak changes by `4e-10` between
`ell_max = 40` and `50`. The caustic is fully converged in the angular
truncation, so **the sum is not what limits it**.

This is confirmed independently by the emitter spectrum. The angular weights
are `g_l = i_l(kappa) / i_0(kappa)`; for the production `kappa = 64` the ratio
`g_50 / g_0` is `4.8e-9`, the retained angular power at `ell_max = 50` is
`1.0000000000`, and `ell_max = 38` already suffices to omit less than `1e-10`.
The evolution carries every multipole the emitter excites.

What does set the width is the emitter itself:

| quantity | value |
|---|---:|
| measured focus FWHM | `40.2` deg |
| emitter angular width `kappa^{-1/2}` | `7.16` deg |
| angular truncation scale `pi / ell_max` | `3.60` deg |
| focus width in emitter widths | `5.6` |

The focus is `5.6` emitter widths across and `11` truncation scales across.
A longer sum would change nothing; a narrower emitter is the only lever.

## 3. The narrow emitter

On that evidence a separate visualization only case was run, following the
prescription of halving the radial and temporal widths and concentrating the
angular profile:

| parameter | production | narrow |
|---|---:|---:|
| radial half width | `0.75M` | `0.375M` |
| temporal half width | `2.0M` | `1.0M` |
| angular concentration `kappa` | 64 | 256 |
| angular width `kappa^{-1/2}` | `7.16` deg | `3.58` deg |
| angular truncation `ell_max` | 50 | 80 |
| retained angular power at that truncation | `1.0000000000` | `1.0000000000` |
| retained angular power at `ell_max = 50` | `1.0000000000` | `0.9999607857` |

The truncation is chosen from the spectrum, not by habit: at `kappa = 256` the
smallest `ell_max` that omits less than `1e-10` of the angular power is `76`,
so `80` is the first round value that clears it, and `50` would omit `3.9e-5`.

Two constraints on how this case may be described:

* it is **not a point source limit**. `kappa = 256` is a finite angular width
  of `3.58` degrees, still `1.6` times the truncation scale of its own sum, and
  the focus it produces remains a resolved feature of a smooth emitter;
* it is **not part of any production claim**. It feeds no regulator
  comparison, no production table, and no frozen package manifest. The
  archive records both facts in its metadata under `visualization_only`.

It is run at two levels, `N = 2048` with `dt = 0.0005M` and `N = 1024` with
`dt = 0.001M`, so the resulting picture can be shown to be resolved rather
than asserted to be.

## 4. What the diagnostics draw

`python -m black_hole.caustic_diagnostic_figures <archive>` builds three
figures and two tables:

* **`caustic_section_sequence`** places the equatorial sections around the
  focus on one shared symmetric linear scale, in areal radius, cropped to
  `20M`, with the event horizon filled, the photon sphere dashed, the emitter
  marked, the wavefront ridge traced from the data, and the axial focus
  circled. Because the scale is shared and fixed, brightness differences
  between panels are amplitude differences in the solution.
* **`caustic_focus_profile`** measures the focus: the transverse cut with its
  half maximum and width, the radial cut along the axis, the truncation study
  above, and the focusing history that selected the snapshot.
* **`caustic_height_colour`** draws the wavefront as a height and colour sheet
  over the equatorial plane, following the convention of the earlier
  Schwarzschild caustic study, where the same scalar sets both the elevation
  and the colour. The colour range is a fixed fraction of the strongest field
  so the strong field region saturates deliberately and the weaker caustic
  stays visible.

The ridge trace is exactly that, the per angle radial maximum of `|Phi|`. It
is not a claim to have located a mathematical cusp, and the code does not
pretend otherwise.

## Tests

`tests/test_caustic_diagnostics.py` drives the estimators with analytic fields:
a single multipole must reproduce its own Legendre shape through the addition
theorem, a planted ring focus must be found at its planted radius, a focus on
the band edge and a monotonic pile up must both be rejected, and the width
estimator must recover a known Gaussian to `2e-3` relative.

## Reproduction

```text
python -m black_hole.caustic_narrow_source fine --budget-only
python -m black_hole.caustic_narrow_source fine
python -m black_hole.caustic_narrow_source coarse
python -m black_hole.caustic_diagnostic_figures results/caustic_diagnostics/raw/narrow_source_L80_fine.npz
python -m pytest tests/test_caustic_diagnostics.py
```
