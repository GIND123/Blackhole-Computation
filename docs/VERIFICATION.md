# Standalone verification and reproducibility

Three checks are kept separate from the physics results so that each can be
quoted on its own: the cross code comparison of the sourced evolution, the
standalone Schwarzschild convergence and truncation checks, and the foliation
diagnostic. None of them is used to calibrate a physical result.

## Sourced Dedalus and finite difference comparison

Two independent implementations evolve the same sourced system. They share no
radial discretization, no time integrator, and no derivative operator.

| | finite difference | Dedalus |
| --- | --- | --- |
| radial coordinate | uniform \(\rho\in[0,1]\) | ChebyshevT \(\rho\in[0,1]\) |
| radial derivative | 8th order centred, matched one sided ends | spectral, dealias 1.5 |
| radial resolution | 2048 | 512 |
| time integrator | explicit classical RK4 | RK443 |
| timestep \(\Delta\tau/M\) | 0.0005 | 0.002 |
| source evaluation | staged | `GeneralFunction` at every RK stage |
| angular cutoff | \(\ell_{\max}=50\) | \(\ell_{\max}=50\) |
| evolved responses | 51 | 51 |
| reconstructed real modes | 676 | 676 |

Both evolve one radial response per retained \(\ell\) and reconstruct the
angle dependent field through the identity
\(u_{\ell m}=g_\ell Y_{\ell m}(\theta_s,\phi_s)\,u_\ell\), which holds for zero
initial data on a spherically symmetric background. The stored angular
expansion retains a fraction \(1-10^{-16}\) of the source weight, with a
maximum relative reconstruction error of \(3.4\times10^{-9}\) against an
independent 400 point quadrature.

The comparison is evaluated at one common geometric retarded time with no
fitted clock translation:

| quantity | value |
| --- | --- |
| retarded time | \(U=44M\) |
| sphere relative \(L^2\) | \(3.99\times10^{-4}\) |
| max modal difference over reference maximum | \(2.69\times10^{-4}\) |
| norm | exact Parseval sum over stored real harmonic modes |

Comparing the two reconstructed meridional planes rather than the extraction
sphere gives relative \(L^2\) differences of \(5.7\times10^{-4}\) at
\(U=26.66M\) and \(1.7\times10^{-4}\) at \(U=44.06M\).

This is a validation of the sourced waveform and of the reconstruction. It is
not combined with the \(D_1\) timing results, which use the matched template
lag as their primary estimator, while the historical cross code timing
comparison used the analytic envelope. The two are not measurements of the
same observable.

    python -m black_hole.caustic_visualizations \
        --output-dir results/caustic_visualizations/dedalus_candidate \
        validate-dedalus \
        results/caustic_visualizations/dedalus_candidate/raw/sds_L80_dedalus.npz \
        results/regulator_production_v3/raw/source/fine/sds_L80.npz --time 44

## Standalone Schwarzschild checks

These use the \(\Lambda=0\) sourced archives only, so they are independent of
the regulator sequence. The refinement ladder varies radial resolution,
timestep, and angular cutoff together; the truncation rows then isolate the
angular cutoff alone by discarding retained responses from the single fine
evolution, which changes nothing else.

| check | level | \(N_r\) | \(\Delta\tau/M\) | \(\ell_{\max}\) | sphere-time rel. \(L^2\) to fine | max constraint \(L^\infty\) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| combined refinement | coarse | 1024 | 0.001 | 42 | \(3.44\times10^{-7}\) | \(4.65\times10^{-12}\) |
| combined refinement | medium | 1536 | 0.000667 | 46 | \(7.66\times10^{-8}\) | \(1.01\times10^{-11}\) |
| combined refinement | fine | 2048 | 0.0005 | 50 | 0 | \(1.80\times10^{-11}\) |
| isolated angular truncation | fine responses | 2048 | 0.0005 | 42 | \(2.29\times10^{-9}\) | \(1.80\times10^{-11}\) |
| isolated angular truncation | fine responses | 2048 | 0.0005 | 46 | \(1.26\times10^{-10}\) | \(1.80\times10^{-11}\) |

The constraint stays below \(1.8\times10^{-11}\) on every level, and the
angular cutoff contributes four orders of magnitude less than the combined
refinement difference, so the ladder is limited by the radial discretization
and the timestep rather than by \(\ell_{\max}\). No time translation is fitted
in any row.

    python -m black_hole.schwarzschild_verification

Outputs land in [results/schwarzschild_verification](../results/schwarzschild_verification):
a CSV, and the publication-size figure as PNG and PDF with embedded TrueType
fonts.

## Foliation diagnostic

The foliation table that selects the minimal gauge is evaluated from the
closed-form bridge coefficients in `black_hole/sds_model.py`. Nothing is read
from a simulation archive. The three quantities are the maximum characteristic
speed, the minimum propagation coefficient, and the retarded time offset

\[
q_B=\int_{r_0}^{r_c}\frac{1+B}{f}\,\mathrm{d}r .
\]

Numerator and denominator both vanish at the cosmological horizon, so the
quadrature evaluates the analytic ratio inside the domain and uses its one
sided limit only in the endpoint neighbourhood where direct subtraction would
lose precision.

    python -m black_hole.foliation_diagnostics write
    python -m black_hole.foliation_diagnostics show 80 160

The first command regenerates `foliation_conditioning.csv`,
`foliation_min_A.csv`, and `foliation_retarded_offsets.csv` under
`paper/figs/data`. The regenerated files are byte for byte identical to the
archived ones, and `tests/test_foliation_diagnostics.py` asserts both that
identity and the underlying values to nine decimal places.

## Later manuscript calculations: implementation details

The manuscript uses the high-precision spectral-element solver for the
exterior ringdown and matched `L/M=640` tail results, including the latter's
Schwarzschild and uniform-SdS controls. The uniform fixed-mode waveform
sequence and the `L/M=3072` Price-law benchmark use Dedalus. The
[SBP solver note](SBP_HERMITE_PILOT.md) records the earlier development of the
spectral-element method; its opening description refers to the pilot stage.
The final calculations are specified in
[the ringdown package](../results/revision_sbp_ringdown_v1/) and
[the matched-tail plan](../results/sbp_matched_tail_revision_v1/PLAN.md).

### Spectral-element formulation and mesh

The continuous Legendre–Gauss–Lobatto discretization evolves `u` and
`v = partial_tau u` through `M partial_tau v = C v - K u`. Each element has
a quadrature matrix `H` and derivative matrix `D` satisfying
`H D + D^T H = E`, where `E` is the signed boundary matrix. The mass,
stiffness, and transport matrices are assembled from

```text
M_e = H/A
K_e = D^T H p D + H P
C_e = H B D - D^T H B + E B.
```

Coefficient functions denote diagonal matrices of their nodal values.
Continuity of `u` and `v` cancels internal boundary terms. The physical
boundaries are outflow and require no prescribed data. The semidiscrete
energy `E_wave = (v^T M v + u^T K u)/2` obeys
`partial_tau E_wave = -v(0)^2 - v(1)^2`; positivity also requires positive
semidefinite `K`. This identity alone does not establish waveform accuracy.
Geometry and operators use 50 decimal digits; evolution uses double-double
arithmetic (about 31 digits) and third-order, L-stable Radau IIA.

Element boundaries include the initial-data support, `rho=0.55,0.70,0.90,0.97`,
and the transition edges. Schwarzschild and uniform-SdS controls use fixed
template positions for the latter. Exterior tails and all ringdown runs
divide the transition or template interval into four elements and add
boundaries at `r/M=80,120,160,200` where these radii lie between `rho=0.97`
and the inner transition or template boundary. The matched-tail controls
retain the standard nine-element mesh; the exterior tail cases use sixteen
elements. The operator configurations record the exact boundaries.

### Independent shared-flux check

The first-order exterior check uses the characteristic fields and fluxes

```text
h = pi + psi,    j = pi - psi
F_plus  = A(1+B)h
F_minus = A(1-B)j
partial_tau u = (F_plus + F_minus)/2
partial_tau h =  partial_rho F_plus  - P u
partial_tau j = -partial_rho F_minus - P u.
```

Reusing the same discrete fluxes preserves
`C_ch = (h-j)/2 - partial_rho u = C` at the semidiscrete level:
`partial_tau C_ch = 0`. No damping is added. The boundary factors are
retained as `A(1+B)=(1-rho)alpha_plus` and `A(1-B)=rho alpha_minus`, with
finite `alpha_plus` and `alpha_minus`.

At `L/M=320,640`, this check uses `N=512,768,1024` and `Delta tau=0.01M`,
with an `N=768` check at `0.005M`, through `U=80M`. RK222 treats transport
implicitly and the potential explicitly, with `3/2` dealiasing. On
`15 <= U/M <= 45`, the largest successive spatial waveform changes decrease
from `1.08e-4` to `6.70e-6`, and timestep halving gives at most `5.12e-6`.
The finest shared-flux and spectral-element exterior waveforms agree to
`1.12e-5`, normalized by the fine Schwarzschild waveform's `L2` norm.
The exterior-supported error remains smaller than the uniform-SdS error
at every tested resolution and timestep. Sampling the absolute reduction
constraint every `0.1M` gives a maximum `5.17e-10`; this sampled diagnostic
is distinct from the waveform comparison.

Earlier single-domain exterior runs are excluded from the manuscript's
ringdown table because coefficient and matrix truncation could force the
discrete reduction constraint. The shared-flux formulation removes that
semidiscrete forcing, and the spectral-element formulation evolves no
independent derivative field. The
[constraint and waveform checks](../results/ringdown_constraint_revision_v1/)
retain the implementation history and numerical evidence.

### Sourced-waveform comparison quoted in the manuscript

The manuscript's `5.58e-4` sphere-integrated relative difference comes from
the sourced calculation described in [GREEN_FUNCTION.md](GREEN_FUNCTION.md).
It compares Schwarzschild and uniform SdS at `L/M=80`, using `ell_max=42`,
`Delta tau=0.002M`, and evolution through `72M`. The finite-difference run
uses `N_r=768` and RK4; Dedalus uses 512 Chebyshev modes, `3/2` dealiasing,
and RK443. Both evaluate the source at each integrator stage. This is a
different comparison from the fixed-time visualization check at the start
of this document, and it is independent of the matched-template timing
measurements.
