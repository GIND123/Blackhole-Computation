# Corrected Schwarzschild flat-limit validation

This calculation implements the corrected one-dimensional experiment requested
by Professor Anil Zenginoglu on July 17, 2026. It tests whether the
Schwarzschild-de Sitter (SdS) scalar signal at the future cosmological horizon
approaches the Schwarzschild signal at future null infinity as
\(\Lambda=3/L^2\) tends to zero.

## Geometry and regular coefficients

The finite-\(L\) backgrounds use

\[
f_L(r)=1-\frac{2M}{r}-\frac{r^2}{L^2},\qquad
\rho_L(r)=\frac{1-r_b/r}{1-r_b/r_c}.
\]

Thus \(\rho=0\) and \(1\) are the black-hole and cosmological horizons. In the
flat limit, \(r_b\to2M\), \(r_c\to\infty\), and
\(\rho_L\to1-2M/r\), which is used exactly for the independent Schwarzschild
reference.

All runs use the minimal gauge. The lapse-height poles are cancelled
analytically before evaluating the boost \(B=f h'\). The propagation
coefficient

\[
A=\frac{f\,d\rho/dr}{1-B^2}
\]

is assigned at each horizon by its analytic l'Hopital limit

\[
A_h=\frac{f'(r_h)\rho'(r_h)}{-2s_hB'(r_h)},\qquad
s_b=+1,\quad s_c=-1.
\]

No displaced endpoint or fitted horizon value is used. Unit tests compare the
closed forms with interior limits.

## Identical physical initial data

The earlier Gaussian in \(\rho\) has been removed from this workflow because
\(\rho_L(r)\) depends on \(L\). The corrected data are the same smooth,
compactly supported profile for \(u=r\Phi\) in areal radius:

\[
x=\frac{r-4M}{1.5M},\qquad
u(0,r)=
\begin{cases}
\exp\!\left(1-\dfrac{1}{1-x^2}\right),& |x|<1,\\
0,& |x|\ge1.
\end{cases}
\]

This standard bump is \(C^\infty\), has unit amplitude at \(r=4M\), and has
support \(2.5M<r<5.5M\). Its continuum derivative is

\[
\frac{du}{dr}=-\frac{2x}{1.5M(1-x^2)^2}u,
\qquad
\psi=\frac{du}{dr}\frac{dr}{d\rho},
\qquad
\pi=-B\psi.
\]

In the finite Chebyshev representation, \(\psi\) is initialized as
\(D_\rho u\), the derivative of the represented common profile. This is the
constraint-consistent spectral realization of the same chain-rule identity;
the continuum implementation is tested directly against the analytic formula.

## Geometric retarded time

The additive constants are fixed independently by

\[
h(4M)=0,\qquad r_*(4M)=0,\qquad \frac{dr_*}{dr}=\frac1f.
\]

For SdS, with the negative third root \(r_u\), the normalized tortoise
coordinate is

\[
r_{*,L}(r)=
\frac{1}{2\kappa_b}\ln\frac{r-r_b}{4M-r_b}
-\frac{1}{2\kappa_c}\ln\frac{r_c-r}{r_c-4M}
+\frac{1}{2\kappa_u}\ln\frac{r-r_u}{4M-r_u}.
\]

The cosmological-horizon logarithms in \(h_L+r_{*,L}\) cancel analytically.
For a general normalization radius \(r_0\), the implemented endpoint limit is

\[
q_L=
\frac{1}{2\kappa_b}
\ln\!\frac{(r_c-r_b)^2r_0}{r_c(r_0-r_b)^2}
+\left(\frac{1}{2\kappa_u}-\frac{1}{2\kappa_c}\right)
\ln\!\frac{r_c}{r_0}.
\]

The Schwarzschild limit is also closed form:

\[
q_0=4M\ln\frac{r_0}{r_0-2M}=4M\ln2
\quad\text{for }r_0=4M.
\]

Signals are compared using \(U=\tau-q_L\), on the common sampled interval
\(-2.4226M\le U\le197.2274M\). No cross-correlation, fitted translation, or
endpoint extrapolation is used.

## Production calculation

The production settings are \(M=1\), \(\ell=2\), 256 Chebyshev modes, RK222
with \(\Delta\tau=0.01M\), and final time \(200M\). No boundary conditions are
imposed at the null boundaries.

| \(L/M\) | \(q_L/M\) | \(|q_L-q_0|/M\) | Relative waveform L2 error | Linf error | Maximum constraint |
|---:|---:|---:|---:|---:|---:|
| 20 | 2.461850 | 0.310739 | 0.637800 | 0.215998 | \(6.41\times10^{-10}\) |
| 40 | 2.597389 | 0.175199 | 0.265234 | 0.092405 | \(3.36\times10^{-9}\) |
| 80 | 2.679137 | 0.093452 | 0.127435 | 0.041692 | \(4.01\times10^{-9}\) |
| 160 | 2.724272 | 0.048316 | 0.056625 | 0.020420 | \(1.62\times10^{-9}\) |

The Schwarzschild constraint maximum is \(6.96\times10^{-10}\). Both waveform
norms decrease monotonically. The empirical powers under \(L\)-doubling are
1.27, 1.06, and 1.17. These are finite-range numerical observations, not a
claim of a proved asymptotic law.

The late-time finite-\(L\) signals need not be uniformly ordered: the SdS decay
scale itself grows with \(L\). The present conclusion concerns convergence of
the complete finite-time signals. Larger \(L\) and longer evolutions are a
separate tail study.

## Numerical convergence

Independent checks were run at \(L/M=20\) and \(160\) through \(100M\).
The spatial sequence uses \(N=192,256,384,512\) and
\(\Delta\tau=0.0025M\). Successive relative waveform differences decrease as

- \(L=20\): \(5.54\times10^{-2}\), \(7.38\times10^{-3}\),
  \(7.45\times10^{-4}\);
- \(L=160\): \(2.71\times10^{-2}\), \(1.01\times10^{-2}\),
  \(2.96\times10^{-3}\).

The production-resolution difference \(N=256\) versus \(384\) is therefore
about 0.74% at \(L=20\) and 1.01% at \(L=160\), below the corresponding
physical flat-limit differences of 63.8% and 5.66%.

The timestep sequence uses 512 modes and
\(\Delta\tau/M=0.04,0.02,0.01,0.005\). Every refinement decreases the
waveform difference. On the triplet bracketing the production step
(0.02, 0.01, 0.005), the measured orders are 1.71 for \(L=20\) and 2.01 for
\(L=160\). The latter recovers the nominal second order of RK222; the former
is convergent but not fully in the asymptotic time-step regime. At the
production step, the relative change to \(\Delta\tau=0.005M\) is
\(1.53\times10^{-4}\) in both cases.

## Reproduction and evidence

From the repository root under the supplied `dedalus3` environment:

```bash
python -m black_hole --verbose sds-flat-limit \
  --resolution 256 --timestep 0.01 --end-time 200 \
  --signal-dt 0.05 --snapshot-dt 0.5 \
  --convergence-end-time 100 \
  --output-dir results/sds_scalar/flat_limit
```

Principal outputs:

- [Initial profiles](../results/sds_scalar/flat_limit/initial_profiles_areal_radius.png)
- [Retarded-time offsets](../results/sds_scalar/flat_limit/retarded_time_offsets.png)
- [Aligned waveforms](../results/sds_scalar/flat_limit/waveform_comparison.png)
- [Differences versus time](../results/sds_scalar/flat_limit/waveform_differences.png)
- [Error norms](../results/sds_scalar/flat_limit/flat_limit_norms.png)
- [Constraint preservation](../results/sds_scalar/flat_limit/constraints.png)
- [Convergence summary](../results/sds_scalar/flat_limit/convergence/convergence_summary.png)
- [Numerical summary](../results/sds_scalar/flat_limit/flat_limit_summary.csv)
- [Aligned waveform data](../results/sds_scalar/flat_limit/waveform_differences.csv)
- [Offset data](../results/sds_scalar/flat_limit/retarded_time_offsets.csv)
- [Initial-profile data](../results/sds_scalar/flat_limit/initial_profiles.csv)
