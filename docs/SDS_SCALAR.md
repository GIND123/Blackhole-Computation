# Schwarzschild-de Sitter scalar bridge study

This study implements the 1D reduced scalar wave equation on
Schwarzschild-de Sitter bridge foliations described in section 5.2 of
Anil Zenginoglu's article, *Bridging time across null horizons*.

## Equation

For a scalar spherical-harmonic mode with field variable \(u=r\Phi_\ell\),

\[
(-\partial_t^2+\partial_{r_*}^2-V_\ell)u=0,
\qquad
\frac{dr_*}{dr}=\frac{1}{f},
\]

\[
f(r)=1-\frac{2M}{r}-\frac{r^2}{L^2},
\qquad
V_\ell=f\left[\frac{\ell(\ell+1)}{r^2}+\frac{f'}{r}\right].
\]

The black-hole and cosmological horizons are the positive roots
\(r_b\) and \(r_c\) of \(f\). The numerical coordinate is

\[
\rho=\frac{r-r_b}{r_c-r_b}, \qquad \tau=t+h(r),
\]

and the future-directed bridge boost is

\[
B=f\frac{dh}{dr}.
\]

With \(p=d\rho/dr_*=f/(r_c-r_b)\), \(A=p/(1-B^2)\), and
\(P=V_\ell/p\), the evolved first-order system is

\[
\partial_\tau u=A(B\psi+\pi),
\]

\[
\partial_\tau\psi=\partial_\rho[A(B\psi+\pi)],
\]

\[
\partial_\tau\pi=\partial_\rho[A(\psi+B\pi)]-Pu.
\]

The constraint is

\[
C=\psi-\partial_\rho u.
\]

No boundary conditions are imposed. The bridge boosts make the black-hole
and cosmological horizons outflow boundaries for the two physical radial
characteristics.

## Bridge choices

The implementation compares six future-directed bridge choices:

- minimum height function
- minimal gauge
- linear boost
- flat-limit modification of the linear boost
- Mavrogiannis boost
- slow-roll boost

For Mavrogiannis, the code uses the sign for which
\(B(r_b)=+1\) and \(B(r_c)=-1\), matching the future-directed horizon
behavior used by the evolution.

## Numerical run

The generated result set used:

| Quantity | Value |
|---|---:|
| \(M\) | 1 |
| \(L\) | 10 |
| \(\ell\) | 2 |
| \(r_b\) | 2.0914884844 |
| \(r_c\) | 8.7888506625 |
| Chebyshev modes | 192 |
| Timestep | 0.02 |
| Final time | 200 |

The Gaussian data were centered at 45 percent of the interval between the
horizons with width 0.35 in areal-radius units. The physical time
derivative was set to zero on the initial bridge slice.

Because each bridge has a different \(\tau=0\) hypersurface, the six
waveforms are not identical-Cauchy-data comparisons. They are a numerical
test of regular evolution quality and coordinate behavior for the same
localized profile specified on each bridge slice.

## Bridge summary

| Bridge | Turning radius | Max constraint |
|---|---:|---:|
| minimum height | 3.867294 | \(2.37\times10^{-9}\) |
| minimal gauge | 2.863694 | \(1.79\times10^{-11}\) |
| linear boost | 5.440170 | \(1.47\times10^{-11}\) |
| flat-limit linear | 2.797390 | \(6.27\times10^{-9}\) |
| Mavrogiannis | 2.999986 | \(2.48\times10^{-11}\) |
| slow-roll | 3.557876 | \(5.03\times10^{-9}\) |

The Mavrogiannis bridge turns at \(r\simeq 3M\), as expected from the
photon-sphere construction. The linear bridge turns at the midpoint of the
domain.

## Convergence

The convergence study used the minimal bridge. The spatial sequence was
run with \(\Delta\tau=0.0025\) through \(\tau=80M\):

| Modes | Difference from next resolution |
|---:|---:|
| 32 | \(1.05\) |
| 48 | \(5.04\times10^{-4}\) |
| 64 | \(5.88\times10^{-7}\) |

The timestep sequence used 256 modes:

| Timestep | Difference from next timestep | Observed order |
|---:|---:|---:|
| 0.02 | \(2.31\times10^{-4}\) | 2.78 |
| 0.01 | \(3.36\times10^{-5}\) | 2.00 |
| 0.005 | \(8.40\times10^{-6}\) | |

The refined temporal sequence recovers the expected second-order RK222
behavior.

## Generated evidence

- [Bridge boosts and characteristic speeds](../results/sds_scalar/bridge_boost_characteristics.png)
- [Spacetime bridge gallery](../results/sds_scalar/spacetime_bridge_gallery.png)
- [Cosmological-horizon waveforms](../results/sds_scalar/cosmological_horizon_waveforms.png)
- [Black-hole and cosmological horizon comparison](../results/sds_scalar/horizon_signal_comparison.png)
- [Constraint comparison](../results/sds_scalar/constraint_bridge_comparison.png)
- [Convergence plot](../results/sds_scalar/convergence/convergence.png)
- [Bridge summary table](../results/sds_scalar/bridge_summary.csv)
- [Convergence table](../results/sds_scalar/convergence/convergence.csv)
