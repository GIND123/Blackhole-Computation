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
  angular contrast, defined as the axial value divided by the mean of `|Phi|`
  over `60` to `120` degrees at the same radius. This is a descriptive contrast,
  not a physical energy-amplification factor.

On the dense `0.25M` cadence sequence this selects `tau = 47.75M`, with the
axial maximum at `r = 4.72M`, peak `|Phi| = 8.5281e-3`, and angular contrast
`83.1`. For comparison, the outer-observer waveform peak identifies an arrival,
not the earlier spatial focusing event.

The dense sequence follows the axial maximum inward as the contrast approaches
a broad maximum:

| `tau/M` | axial maximum radius | peak `|Phi|` | angular contrast |
|---:|---:|---:|---:|
| 46.50 | `9.86M` | `4.2119e-3` | 70.6 |
| 47.00 | `7.27M` | `5.9126e-3` | 78.9 |
| 47.50 | `5.43M` | `7.6631e-3` | 82.6 |
| **47.75** | **`4.72M`** | **`8.5281e-3`** | **83.1** |
| 48.00 | `4.13M` | `9.3763e-3` | 83.1 |

At the next output the maximum reaches the lower edge of the search band and
is rejected by the interior-maximum criterion.

## 2. What sets the width of the focus

This is the question that decides whether a sharper picture needs a longer
angular sum or a narrower emitter, and it is answered by truncating the
reconstruction rather than by comparing scales.

At the selected focus, reconstructing the same archived snapshot with
progressively shorter sums:

| `ell_max` | peak `|Phi|` | focus FWHM | relative change in peak |
|---:|---:|---:|---:|
| 20 | `8.524409e-03` | `40.17` deg | `4.27e-04` |
| 30 | `8.528035e-03` | `40.17` deg | `2.06e-06` |
| 40 | `8.528053e-03` | `40.17` deg | `1.12e-09` |
| 50 | `8.528053e-03` | `40.17` deg | 0 |

The sampled FWHM remains `40.2` degrees throughout this truncation sequence;
the width estimator resolves full widths in increments of about `0.03` degree.
The peak changes by `4.27e-4` between `ell_max = 20` and `50`, by `2.06e-6`
between `30` and `50`, and by `1.12e-9` between `40` and `50`. Angular
truncation at `ell_max = 50` is therefore subdominant for this focus.

This is confirmed independently by the emitter spectrum. The angular weights
are `g_l = i_l(kappa) / i_0(kappa)`; for the production `kappa = 64` the ratio
`g_50 / g_0` is `4.8e-9`, the retained angular power at `ell_max = 50` is
`1.0000000000`, and `ell_max = 38` already suffices to omit less than `1e-10`.
Thus `ell_max = 50` captures the source spectrum to a much tighter tolerance
than the spatial and temporal comparisons used here.

The remaining width is much larger than either the source's nominal angular
width or the truncation scale:

| quantity | value |
|---|---:|
| measured focus FWHM | `40.2` deg |
| emitter angular width `kappa^{-1/2}` | `7.16` deg |
| angular truncation scale `pi / ell_max` | `3.60` deg |
| focus width in emitter widths | `5.6` |

The focus is `5.6` nominal emitter widths across and `11` truncation scales
across. A longer angular sum does not sharpen this archived solution. The
jointly narrower-source calculation below tests whether the source profile,
rather than angular truncation, controls the visible width.

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

### What the narrow emitter measured

Comparing at equal discretization, `N = 2048` and `dt = 0.0005M` on both
sides, so only the source prescription differs:

| quantity | production `kappa = 64` | narrow `kappa = 256` | ratio |
|---|---:|---:|---:|
| emitter angular width | `7.16` deg | `3.58` deg | `0.500` |
| focus FWHM | `40.17` deg | `18.00` deg | **`0.448`** |
| angular contrast | `83.1` | `227.1` | **`2.731`** |
| focus width in emitter widths | `5.61` | `5.03` | `0.90` |

Jointly halving the radial and temporal widths and halving the nominal angular
width reduces the focus FWHM by a factor `0.448`. This demonstrates that a
narrower smooth source produces a sharper focus, but it does not isolate which
source-width parameter controls the change.

The angular contrast rises by `2.73`. This is **not** a test of the
`1/sigma` energy amplification of the earlier Schwarzschild study: the narrow
case changes three source widths at once, and the quantity here is a field
contrast rather than an energy ratio. Establishing a one-parameter scaling
would require varying one width at a time.

### The narrow emitter uses the truncation it was given

Repeating the truncation study on the narrow case separates two things that
looked similar before. For the production source, the sampled FWHM is already
stable at `ell_max = 20` and the peak is stable by `ell_max = 30` to a few
parts in `10^6`. The narrow-source focus still changes at those cutoffs:

| `ell_max` | peak `|Phi|` | focus FWHM | relative change in peak |
|---:|---:|---:|---:|
| 20 | `2.050407e-02` | `18.270` deg | `1.5e-02` |
| 30 | `2.067965e-02` | `18.060` deg | `6.7e-03` |
| 40 | `2.079307e-02` | `18.000` deg | `1.2e-03` |
| 50 | `2.081887e-02` | `18.000` deg | `3.6e-05` |
| 60 | `2.081811e-02` | `18.000` deg | `4.1e-07` |
| 80 | `2.081812e-02` | `18.000` deg | 0 |

The choice `ell_max = 80` clears the preselected source-spectrum criterion of
omitted angular power below `1e-10`. The propagated focus is already stable at
`ell_max = 60` to `4.1e-7` in peak amplitude, with no change in the sampled
FWHM. The source-spectrum omission and the pointwise propagated-field error
are different norms, so their similar values at `ell_max = 50` are reported as
an empirical observation, not as a predicted error bound.

### Convergence of the narrow case

Comparing the two levels at their own selected focus is misleading, because
the angular contrast is nearly flat near its peak: the two levels differ by
`0.09%` in contrast yet select snapshots `0.25M` apart, at `tau = 48.00`
and `48.25`, and different focus radii. The apparent `13%` disagreement in
peak amplitude is that selection difference, not discretization.

Compared at matched bridge time and matched radius:

| `tau/M` | axial `|Phi|` relative difference | FWHM relative difference |
|---:|---:|---:|
| 46.00 | `9.5e-03` | `1.5e-03` |
| 47.00 | `1.6e-05` | `3.0e-03` |
| 47.50 | `9.5e-06` | `1.6e-03` |
| 48.00 | `6.3e-06` | `1.7e-03` |
| 48.50 | `3.2e-06` | `0` |
| 49.00 | `4.1e-07` | `0` |

At matched time and radius, the axial field agrees to a few parts in `10^6`
near the focus. The sampled FWHM differs at the `10^-3` level before becoming
identical at the estimator's angular resolution. The larger difference at
`tau = 46.00` occurs on the steep rise, where a small timing difference enters
directly.

The snapshot selection is worth stating as a property rather than hiding: near
the peak the angular contrast varies by less than `0.1%` over `0.5M`, so which
snapshot is chosen is not robust at this cadence even though the fixed-time
field near the selected focus agrees at the `10^-5` to `10^-6` level.

## 4. What the diagnostics draw

`python -m black_hole.caustic_diagnostic_figures <archive>` builds three
figures and two tables:

* **`caustic_section_sequence`** places the equatorial sections around the
  focus on one shared symmetric linear scale, in areal radius, cropped to
  `20M`, with the event horizon filled, the photon sphere dashed, the emitter
  marked, the wavefront ridge traced from the data, and the axial focus
  circled. Because the scale is shared and fixed, brightness differences below
  the common clipping threshold are amplitude differences in the solution.
* **`caustic_focus_profile`** measures the focus: the transverse cut with its
  half maximum and width, the radial cut along the axis, the truncation study
  above, and the focusing history that selected the snapshot.
* **`caustic_height_colour`** draws the wavefront as a height and colour sheet
  over the equatorial plane, following the convention of the earlier
  Schwarzschild caustic study, where the same scalar sets both the elevation
  and the colour. The display range is set by the 60th percentile of the
  disturbed region, so stronger values saturate deliberately while weaker
  structure remains visible. This clipped sheet is a development diagnostic,
  not a quantitative rendering of the focus height.

The ridge trace is exactly that, the per angle radial maximum of `|Phi|`. It
is not a claim to have located a mathematical cusp, and the code does not
pretend otherwise.

## 5. Presentation alternatives for the focus

The height and colour sheet of section 4 is a development diagnostic and is
not used in the paper. Its display range is the 60th percentile of the
disturbed region, which at the narrow-source focus is `1.6935e-3` while the
peak of the same snapshot is `2.0818e-2`. The strongest feature in the
section is therefore saturated by a factor of `12.3`, and the sheet cannot be
read as a rendering of the focus height.

`black_hole/caustic_focus_figures.py` builds five alternatives from the narrow
source near the spatial focus at `tau = 48.00M`. All five obey two rules.

* **No clipping.** The colour limits are the signed extremes of the data that
  is drawn, so the peak sample lands exactly on the end of the colour bar.
  Each builder records `colour_limit`, `drawn_minimum`, and `drawn_maximum`,
  and `caustic_focus_figures.json` carries `every_figure_is_unclipped` for the
  set.
* **A monotone transfer function.** Weak structure is recovered with a signed
  `asinh` stretch rather than by clipping. The stretch is linear for
  `|Phi| << beta`, logarithmic for `|Phi| >> beta`, invertible, and
  sign preserving. `beta` is the median `|Phi|` of the disturbed region and is
  recorded with each figure. Colour bar ticks carry physical field values, so
  the nonlinearity does not cost the reader the amplitudes.

| figure | what it shows |
|---|---|
| `focus_equatorial_inset` | four equatorial sections at `tau/M = 47.0, 47.5, 48.0, 48.5` on one shared scale, each with the antipodal axis magnified `2.7` times |
| `focus_birdseye_surface` | the unclipped height and colour surface at the focus, camera at elevation `58` degrees and azimuth `200` degrees |
| `focus_axial_cutaway` | the surface cut on the plane `y = 0` that contains the emitter and the focus, with the cross-section drawn on the cut face |
| `focus_axial_time_stack` | nine curtains of the antipodal-axis profile from `tau/M = 46` to `50`, coded by their own bridge time |
| `focus_axial_spacetime_map` | the antipodal-axis field against radius and bridge time, the same information without a projection |

The cameras place the focus nearest the viewer. The bird's-eye azimuth of
`200` degrees looks down the antipodal axis, so nothing on the retained
surface stands between the viewer and the peak; the cutaway camera sits low
and on the far side of the cut plane for the same reason.

### The marker is the selected focus, not an extremum

The strongest axial excursion between `tau/M = 46` and `50` is not the focus.
It sits at `r = 4.00M` and `tau = 48.75M`, on the inner edge of the wave zone,
which is the configuration `axial_focus` rejects because a maximum on a band
edge means the front has already passed. The alternatives therefore mark the
snapshot and radius that `axial_focus` accepts with the largest angular
contrast, which is `tau = 48.00M`, `r = 6.25M`, `Phi = -2.0818e-2`, the same
point the tables report. `tests/test_caustic_focus_figures.py` holds both the
no-clipping rule and this selection in place.

The focus is a negative excursion. The signed convention is kept rather than
plotting `|Phi|`, so the rebound to positive values after `tau = 48.5M` stays
visible in the same figures.

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
python -m black_hole.caustic_diagnostic_figures results/caustic_visualizations/raw/sds_L80_dense.npz --prefix caustic
python -m black_hole.caustic_diagnostic_figures results/caustic_diagnostics/raw/narrow_source_L80_fine.npz --prefix narrow
python -m black_hole.caustic_focus_figures
python -m pytest tests/test_caustic_diagnostics.py tests/test_caustic_focus_figures.py
```

The `--prefix` option names the figure family. The narrow-source figures were
previously produced by renaming the default outputs, which left them outside
the reproduction command; they are now written directly.
