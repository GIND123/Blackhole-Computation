# Three dimensional visualization candidates

Three figures are offered. All are generated from archived evolutions by
[black_hole/render_figures.py](../black_hole/render_figures.py), which draws
through the ray marching renderer in
[black_hole/field_render.py](../black_hole/field_render.py). Each figure writes
a high resolution PNG, a PDF, a JSON record of every display choice, and a
draft caption. They remain outside the manuscript until one is selected.

## Why the first candidates were replaced

The first set of candidates was rejected. The reasons were specific and worth
recording, because they are properties of the tool rather than of the data.

Matplotlib's 3D axes sort whole artists by a single depth key and composite
them with a painter's algorithm. There is no depth buffer, so a surface that
should be partly occluded is drawn either entirely in front or entirely
behind. That is the origin of the moire banding across the earlier cutaway and
of its collapse into a flat disc: a dense mesh viewed near edge on is exactly
the case the algorithm cannot resolve. No amount of parameter tuning fixes it.

Three further faults were independent of the tool. The field was drawn on a
diverging colour map centred on zero while being almost entirely of one sign,
so the whole domain rendered as one pale blue; the frames carried no geometric
context, so neither the black hole nor the cosmological horizon was visible
anywhere in them; and the informative quantity in the regulator comparison,
the residual, appeared only as a number in a subplot title while the five
panels it described looked identical.

## The renderer

Rays are marched through the volume in NumPy. Occlusion, the cut surfaces, and
the nesting order of the extraction spheres are therefore resolved
geometrically rather than by drawing order.

The archives store one radial response for each retained \(\ell\). Because
every background here is spherically symmetric and the source is axisymmetric
about its own direction, the spherical addition theorem rewrites the
reconstruction as a sum over \(\ell\) of \(R_\ell(r)P_\ell(\cos\gamma)\). This
is an exact rewriting of the complete angle dependent field, not an
axisymmetric approximation of it: evolving every excited \(m\) separately gives
the same field, and a regression test checks the two reconstructions against
each other. The renderer tabulates that field once on a uniform
\((r,\cos\gamma)\) grid and treats it as a genuine three dimensional volume.

Four display choices are made explicitly and recorded in each figure's JSON.

The displayed radius is \(R=\sqrt{r}\). Only the radial coordinate is remapped,
so every angle and every sphere is reproduced exactly. Without it the horizon
of the \(L/M=80\) bridge is a fortieth of the frame and the black hole is
invisible in the same view as the cosmological horizon.

Colour is a signed logarithmic transfer, linear below five percent of the
frame maximum. Both the positive and the negative ramp start from the same
neutral dark value, so a region the wave has not reached is not tinted by
whichever sign wins the comparison at zero.

Values below \(10^{-5}\) of the frame maximum are treated as exactly zero. A
logarithmic transfer otherwise assigns visible colour to round off. In these
archives the region ahead of the wave sits near \(10^{-10}\), which is
\(10^{-9}\) of the frame maximum, and its structure differs between the
Dedalus and finite difference codes. Rendering it would show numerical floor
as though it were physics.

A quarter of the volume is removed on two meridional planes. Both exposed
faces contain the source axis, so the cut always reveals a meridional section.

No spectral filter is applied, and none is needed. The antipodal sum
alternates in sign, so a truncated Legendre series could in principle ring at
the focus. Repeating the antipodal measurement at truncations from
\(\ell_{\max}=30\) to 50 moves the peak amplitude by less than \(10^{-5}\)
relative and the fitted phase by \(6\times10^{-7}\) degrees: the emitter is
band limited by its own angular width, so the focus is smoothed by the source
and not by the truncation. The caustic in these figures is resolved physics.

## Caustic echo

Each frame is one leaf of the hyperboloidal foliation, a surface of constant
bridge time, and its label is the retarded time that leaf carries at the
outer boundary, \(U=\tau-q_L\). The label is exact there and is a slice
identifier further in, which is the price of showing a spacetime calculation
as a spatial picture. Colour is the pointwise transfer function of the field
on the two exposed cut faces; in the translucent remainder it is the same
transfer function accumulated along the ray, so the colour key reads exactly
on the faces and indicates depth elsewhere.

The hero panel is the measured antipodal peak; the row below it is the same
scene at the archived snapshot times. The sequence shows the pulse leaving the
source direction, sweeping around the black hole, and converging on the
antipode. All panels share one colour scale, so the amplitude decay between
frames is the physical decay rather than a rescaling.

The snapshots are stored by a dedicated run of the same
\(\ell_{\max}=50\) Dedalus calculation, which records the two measured peak
times exactly and an evenly spaced retarded time sequence around them. Peak
times are converted to bridge time with the analytic \(q_L\) and rounded only
to an exact integration step. No time translation is fitted. The stored run
holds 22 snapshots, from the initial slice at \(U/M=-2.68\) through
\(U/M=50.25\); seventeen of them carry signal above the display floor, the
first at \(U/M=24\). The maximum constraint violation over the run is
\(1.8\times10^{-10}\).

The change of sign visible in the last two frames is a property of the
archive, not of the rendering. On the outer sphere the antipodal value sits at the
numerical floor until \(U\approx38M\), arrives at \(-8.0\times10^{-5}\) at
\(U=40M\), reaches \(-7.0\times10^{-2}\) at the measured peak \(U=44.06M\),
and is \(+2.2\times10^{-2}\) by \(U=46M\). The direct pulse in the source
direction peaks at \(-3.4\times10^{-2}\) at the measured time \(U=26.66M\).
Both are three to seven orders of magnitude above the
\(10^{-5}\) display threshold.

The change of sign is a consequence and not the phenomenon. The echo is the
direct pulse rotated in the phase of its analytic signal at the focus, by a
measured \(42.0\pm1.0\) degrees; a sign reversal would be 180 degrees. The
measurement, its angular dependence, and its sensitivities are in
[CAUSTIC_PHASE.md](CAUSTIC_PHASE.md).

The snapshots are produced and the figure drawn with

    python -m black_hole.caustic_visualizations \
        --output-dir results/caustic_visualizations \
        run-snapshots --backend dedalus --sequence 16 50 18 \
        --name-suffix _sequence 80

    python -m black_hole.render_figures echo \
        results/caustic_visualizations/raw/sds_L80_dedalus_sequence.npz

## Sphere and time

[sphere_time_echo.png](../results/caustic_visualizations/sphere_time_echo.png)
shows the extraction sphere itself, viewed along the antipodal axis so that the
source direction lies on the far side and the antipode is at the centre of each
disc. The wavefront arrives as a ring, converges, and collapses on the
antipode at the measured caustic time.

The panel below carries the same field on the axis. On the axis the Legendre
sum closes, \(P_\ell(1)=1\) and \(P_\ell(-1)=(-1)^\ell\), so both waveforms
follow from the archived responses with no sphere reconstruction and no
interpolation in time. The two dashed lines are the direct and antipodal peak
times measured by the analytic envelope estimator, \(U=26.66M\) and
\(U=44.06M\).

The two lower panels carry the measurement rather than an assertion. On the
left, the dashed white curve is the direct pulse rotated by
\(\phi=43^\circ\) in the phase of its analytic signal, delayed by \(17.7M\)
and scaled by 2.20; it accounts for 95 percent of the variance in the fit
window. On the right, the same fit is repeated along a scan of directions: the
rotation stays within 9 degrees of zero while the front sweeps around the hole
and switches on only within the last few degrees of the axis, where the
amplitude also rises. The transition width matches the angular width of the
emitter, \(\kappa^{-1/2}=0.125\) radians. See
[CAUSTIC_PHASE.md](CAUSTIC_PHASE.md).

The spheres share one colour scale taken over the displayed frames only, so the
angular structure of the echo stays visible; the waveform panel carries the
absolute amplitudes, where the antipodal caustic reaches \(-7.5\times10^{-2}\)
against \(-3.4\times10^{-2}\) for the direct pulse in the source direction.

    python -m black_hole.render_figures sphere-time

## Regulator flat limit

This is the figure for the central claim. Each shell is the future
cosmological horizon of one Schwarzschild de Sitter bridge, drawn at its own
areal radius and painted with the field it carries at the common retarded time
\(U=44M\). A quarter is removed so that all four nested horizons and the black
hole are visible at once. Increasing \(L\) moves the horizon outward; in the
limit it becomes future null infinity of the asymptotically flat problem.

The panel beside it gives the exact Parseval residual of each finite \(L\)
extraction sphere against the independently evolved Schwarzschild field at
\(\mathscr I^+\):

| \(L/M\) | \(r_c/M\) | \(\|\delta u_L\|_2/\|u_0\|_2\) |
| --- | --- | --- |
| 80 | 78.98 | 0.1895 |
| 160 | 158.99 | 0.0965 |
| 320 | 319.00 | 0.0485 |
| 640 | 639.00 | 0.0243 |

The residual halves with every doubling of \(L\). This norm is the exact sum
over the stored real harmonic modes, the same estimator used by the Dedalus
cross code validation. The earlier rejected figure quoted 0.210, 0.107, 0.054,
and 0.027 for the same quantity; those came from a discrete sum over a
Mollweide sampling of the sphere rather than from the modal norm, and are
uniformly about ten percent larger. Both show the same \(1/L\) behaviour.

    python -m black_hole.render_figures regulator

## What was checked

Every claim these figures make is a measurement of the archives, so each one
has a check that can fail.

| check | outcome |
| --- | --- |
| the addition theorem reconstruction against the independent \((\ell,m)\) sum | agree to \(10^{-11}\) at every sampled direction |
| angular truncation of the antipodal caustic, \(\ell_{\max}=30\) to 50 | peak amplitude stable to \(10^{-5}\) relative, fitted phase to \(6\times10^{-7}\) degrees |
| the phase estimator on signals whose answer is known | returns 0, 90 and 180 degrees on a copy, a Hilbert transform and a sign reversal, each to better than \(0.01\) degrees |
| the antipodal rotation across observers and backgrounds | \(41.95\) degrees, full spread \(1.90\) degrees over three observers and two backgrounds |
| fit window displacement and rescaling by \(1M\) | \(1.11\) degrees |
| Dedalus against the eighth order finite difference reconstruction at \(U=44M\) | \(3.99\times10^{-4}\) relative \(L^2\) on the extraction sphere |
| maximum constraint violation of the snapshot run | \(1.8\times10^{-10}\) |

Two faults were found and corrected while doing this.

The angular weight in the pulse time estimator was \(\cos^\ell\gamma\) where it
should have been \(P_\ell(\cos\gamma)\). The two agree exactly on the axis,
which is where the estimator is used, so the archived peak times are
unaffected and reproduce to \(10^{-12}\) after the correction. The expression
would have been wrong for any other direction, so it is fixed rather than
left in place.

The description of the antipodal arrival called it a sign reversal. The change
of sign is real but it is a consequence of a phase rotation, and the rotation
is now measured rather than described. See
[CAUSTIC_PHASE.md](CAUSTIC_PHASE.md).

## Superseded candidates

The earlier candidates are retained for provenance under
`results/caustic_visualizations/` as `sphere_time_caustic`,
`regulator_angular_comparison`, and `dedalus_candidate/caustic_cutaway`, with
their original captions. They are not proposed for the manuscript.

The cross code check behind them still stands and is independent of the
figures: at the common geometric time \(U=44M\) the Dedalus and eighth order
finite difference reconstructions agree to \(3.99\times10^{-4}\) in relative
\(L^2\) on the extraction sphere, with no fitted clock translation. Comparing
the two reconstructed meridional planes directly gives relative \(L^2\)
differences of \(5.7\times10^{-4}\) at \(U=26.66M\) and \(1.7\times10^{-4}\)
at \(U=44.06M\).
