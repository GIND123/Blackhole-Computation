# Matched-tail validation plan (recorded before the additional runs)

The existing independently refined SBP--Radau tail waveforms supply four of
the five requested signals. Complete the missing uniform minimally coupled
`L/M=640` family, and the coarse exterior controls, without changing the
physical experiment, clocks, estimator, or acceptance rule.

All new evolutions use `ell=1`, the original compact velocity bump
`u=0`, `u_tau=G(r)` centered at `6M` with half-width `3M`, the minimal
foliation, exact geometric `U=tau-q`, and outflow horizon endpoints. Evolve
through `U=1000M`; no fitted time shift or amplitude rescaling is allowed.

## Fixed ladder

- Uniform minimal `L/M=640`: local polynomial degrees 24, 32, and 40 at
  `dt=0.1M`, plus degree 40 at `dt=0.05M`, standard nine-element layout.
- Exterior minimal and conformal `L/M=640`: add degree 24 at `dt=0.1M`
  on the final sixteen-element wave-resolved layout. Reuse the completed
  degrees 32 and 40 at `dt=0.1M` and degree 40 at `dt=0.05M` on that layout.
- Schwarzschild and uniform conformal `L/M=640`: reuse the completed
  verified degree-24, degree-32, degree-40, and halved-step calculations.
- Evolution: double-double arithmetic and L-stable, two-stage Radau IIA
  (order 3), with 50-digit geometric coefficients. Preparation is serial;
  at most two new evolutions run concurrently.

The original SBP pilot and frozen production data remain read-only. New
operators, run directories, and derived data are separate. Preparation and
evolution refuse existing destinations.

## Analysis fixed before viewing the missing case

Retain the archived matched-tail criterion: 10M centered RMS envelope,
40M local logarithmic fit, amplitude above ten times the measured
spatial/time envelope difference, spatial and temporal local-index changes
at most 0.1, signed-zero exclusion within half the fit width, and a
continuous `p=3 +/- 0.3` interval of at least `40M`. Repeat the existing
3-by-3 sweep of envelope widths 5/10/20M and fit widths 30/40/60M.

The spatial floor compares degrees 32 and 40 at the same timestep;
the temporal floor compares both timesteps at degree 40. The degree-24
controls test the spatial ordering and are not silently interpreted using
a Richardson model. The degree-40 halved-step signal is displayed.

Keep the distinct stricter `L/M=3072` criterion unchanged; this revision
does not rerun that sequence. Do not replace the matched-tail criterion
with the pilot's additional Schwarzschild-index comparison. Schwarzschild
is one common physical reference curve, not two independent simulations.

Create a genuine two-panel figure with all five RMS envelopes and their
masked local power indices. Use the replaced figure's 3.4-by-3.5-inch
footprint, show the accepted uniform-conformal interval, and retain invalid
segments as gaps. Regenerate interval/sensitivity/refinement tables with
the new curves rather than combining them with old numerical values.
