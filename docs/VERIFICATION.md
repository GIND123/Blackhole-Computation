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
