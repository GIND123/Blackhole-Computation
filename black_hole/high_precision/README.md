# Banded double-double Hermite backend

This experimental backend advances a supplied stationary semidiscrete system

\[
\partial_tu=v,\qquad\partial_tv=Gv-Ju
\]

with fourth-order symmetric Hermite integration, equivalently the diagonal
Padé `[2/2]` approximation to its matrix exponential, or optionally third-order
L-stable Radau IIA integration (Padé `[1/2]`). It does **not** construct
the spatial discretization: the caller must supply a consistent, stable
operator, including any interface or boundary terms.

All decimal input parsing, matrix assembly, banded LU factorization, time
integration and output conversion use QD double-double arithmetic (roughly
31 decimal digits). `double` is an explicitly selected comparison build.

## Build

```
sh black_hole/high_precision/build.sh dd
sh black_hole/high_precision/build.sh double
```

The DD build requires an existing checkout of BL-highprecision/QD at commit
`b1c8ddfd2d4a0f0901a88491524df728a26cbe8e`. Set `QD_SOURCE` to its location
or use the script's local default. The build never downloads anything or
modifies that checkout. QD supplies its own license in that checkout. Local
platform configuration headers are kept in this directory. Compiler flags
disable fast-math and implicit FMA contraction; explicit QD error-free
arithmetic uses FMA where appropriate. Binaries go in ignored `_build/`.

## Contract

```
hermite_banded_dd input.txt output_directory dt endtime cadence [runtime_seconds] [probe_indices...] [--integrator hermite|radau3]
```

`output_directory` must not exist; its parent must exist. The optional runtime
limit defaults to 3600 seconds. Probe indices are zero-based and follow that
limit. Input format is plain whitespace-separated decimal text:

1. `n bandwidth` (one half-bandwidth for both matrices).
2. `n` rows of `rho u_initial v_initial`.
3. For each row `i=0,...,n-1`, then each column
   `j=max(0,i-bandwidth),...,min(n-1,i+bandwidth)`, one pair `J_ij G_ij`.
   No explicit indices are written. Entries outside that band are zero.

Output files:

- `history.csv`: `tau,u_left,v_left,u_right,v_right`, followed by requested
  `u_index,v_index` probes, initially and every `cadence` steps. The final
  time is always included, with a fresh factorization for any partial step.
- `final_state.csv`: full final `rho,u,v` state.
- `backend.json`: completion marker, precision, timestep, number of steps,
  elapsed times, selected integrator, maximum discarded imaginary residue,
  and decimal underflow/subnormal counts. Its existence with
  `completed=true` distinguishes completed output from a partial failed run.

Files use scientific notation with 32 digits after the decimal point.
Nonfinite states fail the run. An interrupted or runtime-limited calculation
retains its partial history but does not write the completion marker.

Double-double arithmetic retains the exponent range of binary64. A decimal
that rounds below that range, for example `1e-428`, is set to zero and counted
in `decimal_underflow_count`; the raw operator input is unchanged. Native
subnormal inputs are correctly rounded to the available binary64 spacing and
counted separately. Normal inputs are parsed at full precision using bounded
power-of-ten scaling, avoiding intermediate exponent overflow in QD's reader.

## Algorithm

Let `alpha=dt/(3+i sqrt(3))`. Each Cayley factor adds the increment
`2 alpha (I-alpha A)^(-1) A y` to the current state; the second factor uses
the conjugate pole. Eliminating `delta_u` reduces each solve to

```
F = I - alpha G + alpha^2 J
r_u = 2 alpha v
r_v = 2 alpha (G v - J u)
delta_v = F^-1 (r_v - alpha J r_u)
delta_u = r_u + alpha delta_v
```

One complex band factorization, with partial row pivoting and twice the upper
bandwidth for fill, serves both conjugate stages and every full time step.
Its conjugate solves are obtained by conjugating the right-hand side and
answer. Tiny imaginary roundoff is measured and discarded after each pair.
This preserves the real invariant subspace; it is not a physical filter.

Increment form reduces cancellation when small increments are added to the
field. It does not remove spatial truncation error, guarantee stability of
an arbitrary supplied operator, or damp unresolved stiff modes: Padé `[2/2]`
is A-stable but not L-stable.

### Optional Radau IIA

`--integrator radau3` selects the third-order stability function

```
R(z) = (1+z/3)/(1-2z/3+z^2/6).
```

It applies the numerator followed by two conjugate resolvents, with
`alpha=dt*(1/3+i/(3 sqrt(2)))`. Each resolvent uses the same Schur matrix form
`F=I-alpha G+alpha^2 J`: solve `v_new=F^-1(v-alpha J u)`, then
`u_new=u+alpha v_new`. The imaginary component is retained between the two
stages and measured/discarded only after the conjugate pair. Radau damps stiff
unresolved modes as `|z|` grows; Hermite instead approaches unit amplification.
This matters for high-degree spatial operators whose very fast, weakly excited
modes can contaminate tiny tails if integrated with an undamped rational method.
Radau's damping is a time-discretization effect, so physical waveform and tail
measurements still need timestep convergence.

## Independent backend checks

```
python -m unittest black_hole.high_precision.test_backend -v
```

With `mpmath` installed these cover fourth-order harmonic-oscillator
convergence, full-decimal input retention, small amplitudes, the partial final
step, overwrite refusal, and comparison against a 60-digit dense Padé solve
for nonsymmetric banded matrices. A separate forced-pivot example has zero
Schur diagonals in exact arithmetic. These are backend algebra tests, not a
validation of the physical spatial discretization.

Radau checks additionally establish third-order convergence, stiff-mode damping,
dense-reference agreement, and partial-step correctness. An extreme-decimal
test covers a `1e-428` underflow and a long-mantissa `1e-300` normal input.
