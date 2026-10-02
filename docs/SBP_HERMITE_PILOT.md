# Independent high-precision scalar-tail solver

This is an independent numerical pilot, not a replacement of the frozen
production archives or manuscript figures. It evolves one spherical-harmonic
sector of the stationary scalar equation. It is a radial verification solver,
not a new nonlinear or three-dimensional evolution implementation.

## Discretization and physical matching

`black_hole/sbp_geometry.py` evaluates the same Schwarzschild, uniform SdS,
and fixed-width exterior-supported metrics, with either minimal or conformal
scalar coupling. Geometry and operator construction use 50-digit arithmetic.
The original velocity bump, minimal foliation, field `u=r Phi`, cosmological
length, and fixed retarded clock `U=tau-q` are retained. No fitted time shift
or amplitude rescaling is used. The exterior curvature includes both
derivatives of the transition function.

The independent equation is

```
u_tau = v
v_tau/A = (p u_rho)_rho + 2 B v_rho + B_rho v - P u,
p = A(1-B^2).
```

It follows directly from the production first-order equations. There is no
independent spatial-derivative variable and hence no reduction constraint to
damp. This is a change of formulation and discretization, not of physics.

`black_hole/sbp_hermite.py` uses continuous Legendre–Gauss–Lobatto spectral
elements. Each element satisfies the diagonal-norm SBP identity. Assembly
identifies the field and velocity at interfaces and couples the flux in the
weak form; this is not an SBP–SAT or discontinuous-Galerkin implementation.
The standard mesh has nine elements and `9p+1` nodes, where `p` is the local
polynomial degree. The exterior transition endpoints are element boundaries.
The `split-layer` layout quarters that element without changing the physical
transition, adding three elements. A shifted-block Schwarzschild control
tests sensitivity to artificial interface locations.

The final exterior `wave-resolved` layout additionally places interfaces at
areal radii `80M`, `120M`, `160M`, and `200M`: 16 elements and `16p+1` nodes.
This matters independently of resolving the transition coefficients. The
original pretransition element spans approximately `60.5M < r < 228.1M`,
or `340.5M` in the incoming optical coordinate. It underresolves the wave
scattered inward by the transition even when the outward signal is smooth
in the compact coordinate. Subdividing only the transition did not cure this
error. On the conformal case at `U~400M`, the highest eight velocity modes
carried about `1.25e-4` of the coefficient norm in that broad element, compared
with at most `2.6e-9` in the four transition elements. The extra pretransition
blocks target this measured spatial error without changing the metric.

The assembled system is `M u_tautau = C u_tau - K u`, with

```
M = sum H/A
K = sum (D^T H p D + H P)
C = sum (H B D - D^T H B) + diag(-1,0,...,0,-1).
```

It obeys `d(v^T M v + u^T K u)/dt = -2(v_left^2+v_right^2)`.
This is a useful stability identity, not an accuracy estimate. Positivity
also requires positive semidefinite `K`; its low generalized eigenvalues are
checked diagnostically. Variable-coefficient quadrature still needs spatial
refinement.

## Time integration and arithmetic

`black_hole/high_precision/` implements fixed-operator banded solves in QD
double-double arithmetic (about 31 decimal digits), with a binary64 build for
an arithmetic control. Decimal coefficients are not first rounded to binary64.
The local QD dependency and build instructions are recorded in that folder.

Both the original fourth-order Hermite/Padé `[2/2]` method and two-stage
Radau IIA/Padé `[1/2]` are available. **Radau is used for the tail comparison.**
The initial Hermite runs exposed a genuine stiff-mode problem: on the
Schwarzschild degree-32 mesh, an eigenvalue approximately
`-1.4763 + 39758.923 i` should decay within a few `M`. At timestep `0.1M`,
Hermite maps it to an effective decay timescale near `9e5 M`, contaminating
the tiny tail. This is not cured by carrying more digits. Radau is L-stable
and suppresses those unresolved modes. No spatial filter or tuned constraint
damping was added. See
[Hairer and Wanner, Stiff differential equations solved by Radau methods](https://www.unige.ch/~hairer/preprints/coimbra.pdf)
for the integrator background.

Hermite remains available for appropriate nonstiff/resolved applications, but
its order alone is not a reason to prefer it for these clustered grids.
Double-double arithmetic is likewise not a claim of 31-digit PDE accuracy.

## Validation and safeguards

- Tests check the SBP identity, exact polynomial derivatives, the assembled
  energy identity, and a manufactured polynomial PDE across an interface.
- The actual C++ backend reproduces fourth-order Hermite and third-order
  Radau convergence, agrees with a 60-digit dense rational solve, and handles
  band pivoting, partial final steps, and small amplitudes.
- Geometry tests compare the existing coefficients and check regular endpoint
  limits, curvature coupling, clocks, and exact compact-support endpoints.
- Comparison tests preserve sub-binary64 differences through decimal
  subtraction, reject mixed-factor controls, enforce quarantine reasons, and
  check relative norms and complete-window coverage. The focused suite has
  69 passing tests.
- Prepared operators and raw run directories refuse overwriting. Each run
  records the decimal input hash, binary hash, completion status, and timing.
- A range-checked decimal reader avoids a QD input-reader overflow. An early
  uniform degree-24 input contained an astronomically negative exponent at a
  compact-support endpoint; the old reader interpreted it as a finite initial
  velocity. The corrected reader underflows it safely, and the geometry now
  assigns exact zero at the support endpoints. The two preliminary Radau runs
  made before this fix are explicitly quarantined in `excluded_runs.json`;
  their histories are retained and corrected runs replace them in the audit.

## Comparison protocol

The pilot evolves the dipole velocity-bump problem to `U=1000M` for
Schwarzschild, uniform minimal SdS at `L/M=3072`, uniform conformal SdS at
`L/M=640`, and exterior-supported SdS at `L/M=640` with both couplings.
It uses a degree ladder, timestep halving, transition-block subdivision, a
Schwarzschild block-shift control, and an arithmetic control. Missing controls
must not be inferred from the nominal arithmetic precision.

The analysis reports raw signed `-Uv/u` separately from the unchanged
production 10M RMS/40M logarithmic-fit estimator. A smoother curve is not
itself evidence of greater accuracy. Refinement uses decimal subtraction at
shared nominal times; comparison with the independently sampled frozen
archives uses cubic interpolation and a separately reported full/half-cadence
sensitivity. Differences from a frozen archive are cross-code discrepancies,
not errors measured against an exact solution. Relative differences near
waveform zero crossings require particular care.

The machine-readable audit and diagnostic plots are under
`results/sbp_hermite_pilot_v1/analysis/`. Its manifest lists the authoritative
current figures and excluded runs. The original manuscript plots and all
production archives remain unchanged.

`black_hole/sbp_flagship_summary.py` makes the compact five-case overview,
window-error table, and reference-matched, refinement-screened Price-interval
table under `results/sbp_hermite_pilot_v1/flagship/`. Those intervals are an
independent consistency test, not a silent replacement of the preregistered
production selection and its quoted interval.

## Uniform-tail outcome

The independently refined uniform calculations reproduce the main physical
conclusions without needing to smooth the instantaneous derivative:

| Calculation | Independent continuous Price-compatible interval | Duration |
|---|---|---:|
| Uniform minimal SdS, L/M=3072, 5% criterion | U/M=172.30–381.10 | 208.80M |
| Uniform conformal SdS, L/M=640, 10% criterion | U/M=153.25–294.70 | 141.45M |

These closely corroborate the archived approximately 209M and 141.5M
intervals. They do not redefine the production study's selected 150M interval.
The Schwarzschild instantaneous index is approximately 3.01346 at 300M,
3.00231 at 500M, and 2.99754 at 900M. The departures from exactly 3 are
finite-time physics, not an estimate of numerical error.

For these three Schwarzschild/uniform families, the observed p=32 to p=40
waveform changes are below 2e-10 in relative L2 norm on the tested tail
windows. Halving the timestep from 0.1M to 0.05M changes the early-tail
waveform by at most 1.9e-7, and the later windows by much less. Independent
differences from the frozen N=3072 results are about 0.007–0.026 percent.
The frozen waveform is not an exact reference, so these discrepancies are
reported separately from the new solver's refinement changes.

The binary64-versus-double-double Schwarzschild control differs by about
4.2e-9 in relative L2 norm over 500–950M. Thus extra digits become useful at
the new method's small truncation scale, but do not explain the much larger
improvement over the old calculation. Spatial formulation, appropriate
resolution of incoming waves, and suppression of stiff grid modes matter
more for these particular experiments.

## Final exterior outcome and plot recommendation

With the wave-resolved mesh, **all five families pass the pilot's numerical
refinement gate throughout U=150–950M**. The final comparison uses 361 radial
nodes for Schwarzschild/uniform SdS and 641 for exterior SdS, versus 3072 in
the finest frozen archives. The following entries are the largest observed
spatial or timestep relative L2 waveform differences among the windows
150–300M, 300–500M, and 500–950M. They are percentages, not absolute error
bounds or pointwise guarantees.

| Calculation | Largest refinement difference | Difference from finest frozen waveform, across those windows |
|---|---:|---:|
| Schwarzschild | 0.0000158% | 0.0144–0.0253% |
| Uniform minimal, L/M=3072 | 0.0000168% | 0.0069–0.0257% |
| Uniform conformal, L/M=640 | 0.0000183% | 0.0071–0.0255% |
| Exterior minimal, L/M=640 | 0.0000038% | 1.23–2.34% |
| Exterior conformal, L/M=640 | 0.000106% | 1.08–2.94% |

An additional degree-48 wave-resolved check was run in binary64 to limit
cost. Against degree-40 double-double at the same timestep, its late-window
relative L2 difference is approximately 1.5e-9 (minimal) and 6.0e-8
(conformal). This changes both degree and arithmetic, so it is supplementary
consistency evidence, not labelled a pure spatial control. The primary
degree-32/40 and timestep comparisons above use double-double throughout.

The exterior transition-generated signal survives the numerical improvement.
Under the reference-matched 10%/40M pilot criterion, the longest acceptable
interval is 12.90M for minimal coupling and 8.50M for conformal coupling.
Neither passes. These agree qualitatively with the existing negative tail
result, but the raw exterior waveforms differ from the frozen calculations
at the percent level. The new curves should therefore not be substituted
into the manuscript while retaining old numerical table entries unchanged.

The improvement is sufficient to justify upgrading the Price-tail figure
and, if desired, the exterior-tail numerical evidence. The new instantaneous
indices can be plotted directly, without relying on a time-averaged estimate
to remove numerical oscillations. A manuscript update should introduce this
independent radial solver, retain the defined acceptance estimator when
quoting intervals, and regenerate the associated tail tables consistently.
There is no finding here that warrants repeating the localized-source or
production ell=2 ringdown study: those experiments were not rerun in this
pilot. The main physical claims remain unchanged.

The final degree-40 runs to U=1000M took approximately 2.4–4.1 minutes each
on this laptop at timestep 0.05M (some concurrent); coarser runs were faster.
All final benchmark evolutions completed. Manuscript assets remain untouched.

See the [overview figure](../results/sbp_hermite_pilot_v1/flagship/flagship_tail_overview.pdf),
[window-error table](../results/sbp_hermite_pilot_v1/flagship/flagship_window_errors.csv),
and [interval table](../results/sbp_hermite_pilot_v1/flagship/flagship_price_intervals.csv).

## Reproduction

From the repository root, with `mpmath`, NumPy, SciPy, and Matplotlib installed:

```sh
sh black_hole/high_precision/build.sh dd
sh black_hole/high_precision/build.sh double
python -m pytest -q tests/test_sbp_geometry.py tests/test_sbp_hermite.py tests/test_sbp_pilot_analysis.py black_hole/high_precision/test_backend.py

python -m black_hole.sbp_hermite prepare --background schwarzschild --degree 40 --destination /tmp/sbp-example-operator
python -m black_hole.sbp_hermite run --prepared /tmp/sbp-example-operator --output /tmp/sbp-example-run --backend black_hole/high_precision/_build/hermite_banded_dd --integrator radau3 --dt 0.05 --end-u 1000 --output-every 2

python -m black_hole.sbp_pilot_analysis --root results/sbp_hermite_pilot_v1
python -m black_hole.sbp_flagship_summary
```

The example destinations must not already exist. For exterior-supported SdS,
use `--background exterior --length 640 --coupling 1/6 --layout wave-resolved`
at preparation. Omit `--coupling 1/6` for minimal coupling. The extra spatial
blocks do not alter the metric. All measured runtime comparisons here are
local pilot timings, not matched hardware/compiler benchmarks against the
original Ubuntu production runs.
