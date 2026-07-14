# Schwarzschild flat-limit validation

This calculation implements the controlled one-dimensional experiment
requested by Professor Anil Zenginoglu. Its purpose is to test whether the
Schwarzschild-de Sitter (SdS) scalar signal at the future cosmological horizon
approaches the asymptotically flat Schwarzschild signal at future null
infinity as the artificial cosmological constant is removed.

## Controlled setup

The finite-\(L\) backgrounds use

\[
f_L(r)=1-\frac{2M}{r}-\frac{r^2}{L^2},
\qquad \Lambda=\frac{3}{L^2},
\]

and the common compact coordinate

\[
\rho=\frac{1-r_b/r}{1-r_b/r_c}.
\]

Thus \(\rho=0\) is the black-hole horizon, \(\rho=1\) is the cosmological
horizon, and

\[
\rho\longrightarrow1-\frac{2M}{r}
\]

as \(L\to\infty\). The Schwarzschild reference uses the limiting coordinate
exactly, with \(\rho=1\) representing future null infinity.

For the Schwarzschild reference, the regular minimal-gauge coefficients are
implemented independently in closed form:

\[
B_0=-1+2(1-\rho)^2,\qquad
A_0=\frac{1}{8M(2-\rho)},\qquad
P_0=\frac{\ell(\ell+1)+(1-\rho)}{2M}.
\]

All calculations use the minimal gauge, \(M=1\), \(\ell=2\), and the same
Gaussian as a function of \(\rho\):

\[
u(0,\rho)=\exp\left[-\frac{(\rho-0.45)^2}{2(0.06)^2}\right].
\]

Both \(u\) and \(\psi=\partial_\rho u\) are therefore identical across the
sequence. The physical time derivative is zero initially, implemented as
\(\pi=-B\psi\). The additive constants in all height functions are fixed by

\[
h_L(4M)=h_{\mathrm{Schw}}(4M)=0.
\]

For reproducibility, the unnormalized SdS minimal height is

\[
\begin{aligned}
\widehat h_L(r)={}&
\frac{1}{2\kappa_b}\ln\frac{r-r_b}{r}
+\frac{1}{2\kappa_c}\ln\frac{r_c-r}{r}\\
&+\frac{1}{2\kappa_u}\ln\frac{r}{r-r_u},
\end{aligned}
\]

where \(r_u<0\) is the third SdS root. The Schwarzschild primitive is

\[
\widehat h_0(r)=-r-4M\ln\frac{r}{M}
+2M\ln\frac{r-2M}{M}.
\]

The code uses \(h(r)=\widehat h(r)-\widehat h(4M)\) in every case.

The production settings are 256 Chebyshev modes, RK222 with
\(\Delta\tau=0.01M\), and final time \(200M\). No boundary conditions are
imposed at either null boundary.

## Analytic horizon treatment

No numerical horizon offset is used. The minimum and minimal boost functions
are evaluated after analytically cancelling their height-function poles
against the factorized metric function. At either horizon,

\[
A=\frac{f\rho'}{1-B^2}
\]

is assigned using its l'Hopital limit

\[
A_h=\frac{f'(r_h)\rho'(r_h)}{-2s_h B'(r_h)},
\qquad s_b=+1,\quad s_c=-1.
\]

Unit tests compare these endpoint expressions with the interior limits.

## Production result

The comparison uses

\[
\Delta u_L(\tau)=u_L(\tau,\mathcal H_c^+)
-u_0(\tau,\mathscr I^+).
\]

| \(L/M\) | \(\Lambda M^2\) | Relative waveform L2 difference | Linf difference | Maximum constraint |
|---:|---:|---:|---:|---:|
| 20 | 0.0075 | 0.363607 | 0.102710 | \(3.15\times10^{-11}\) |
| 40 | 0.001875 | 0.150190 | 0.033806 | \(2.51\times10^{-11}\) |
| 80 | 0.00046875 | 0.072086 | 0.014036 | \(2.70\times10^{-11}\) |
| 160 | 0.0001171875 | 0.036141 | 0.007146 | \(1.91\times10^{-11}\) |

The difference decreases monotonically. The empirical powers obtained when
doubling \(L\) are 1.28, 1.06, and 1.00. This is numerical evidence for
waveform convergence over this parameter range; it is not asserted as an
analytic asymptotic law.

The flat limit is pointwise on compact subsets of \(\rho<1\), not uniform in
every rescaled coefficient exactly at the receding cosmological horizon
\(r_c\sim L\). The reported conclusion is therefore based on the evolved
horizon signals and their verified numerical convergence, not on an
unsupported assumption of uniform coefficient convergence.

## Numerical convergence

Independent convergence studies were performed at \(L=20\) and \(L=160\).
The spatial sequence \(N=64,96,128,192\), with
\(\Delta\tau=0.0025M\), shows rapid spectral convergence until the difference
reaches approximately the numerical floor. The timestep sequence
\(0.02,0.01,0.005,0.0025\) at 256 modes recovers order 2.000 on the refined
triplet for both values of \(L\), as expected for RK222.

## Reproduction

From the repository root:

    /home/govind/miniforge3/bin/mamba run -n dedalus3 \
      python -m black_hole --verbose sds-flat-limit \
      --resolution 256 --timestep 0.01 --end-time 200 \
      --signal-dt 0.05 --snapshot-dt 0.5 \
      --convergence-end-time 100 \
      --output-dir results/sds_scalar/flat_limit

Principal evidence:

- [Waveform comparison](../results/sds_scalar/flat_limit/waveform_comparison.png)
- [Difference versus time](../results/sds_scalar/flat_limit/waveform_differences.png)
- [Flat-limit norms](../results/sds_scalar/flat_limit/flat_limit_norms.png)
- [Coordinate convergence](../results/sds_scalar/flat_limit/coordinate_flat_limit.png)
- [Height alignment](../results/sds_scalar/flat_limit/height_alignment.png)
- [Constraint preservation](../results/sds_scalar/flat_limit/constraints.png)
- [Convergence summary](../results/sds_scalar/flat_limit/convergence/convergence_summary.png)
- [Numerical summary](../results/sds_scalar/flat_limit/flat_limit_summary.csv)
- [Waveform data](../results/sds_scalar/flat_limit/waveform_differences.csv)
