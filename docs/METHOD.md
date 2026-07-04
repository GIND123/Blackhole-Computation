# Numerical method

## Physical equation

For an axial gravitational perturbation of a Schwarzschild black hole,

\[
(\partial_t^2-\partial_{r_*}^2+V_\ell)u=0,
\qquad
V_\ell=\frac{f}{r^2}\left[\ell(\ell+1)-\frac{6M}{r}\right],
\qquad
f=1-\frac{2M}{r}.
\]

The compactification

\[
\rho=f=1-\frac{2M}{r}
\]

maps the event horizon to \(\rho=0\) and future null infinity to
\(\rho=1\). The minimal-gauge boost is

\[
H=-1+2(1-\rho)^2.
\]

With

\[
\psi=\partial_\rho u,\qquad
A=\frac{1}{8M(2-\rho)},\qquad
P=\frac{\ell(\ell+1)-3(1-\rho)}{2M},
\]

the evolved system is

\[
\partial_\tau u=A(H\psi+\pi),
\]

\[
\partial_\tau\psi=\partial_\rho[A(H\psi+\pi)],
\]

\[
\partial_\tau\pi=\partial_\rho[A(\psi+H\pi)]-Pu.
\]

The minus sign multiplying \(P u\) follows from the original
Regge--Wheeler equation and is also the sign used by the reference Colab
implementation. The final displayed system in the blog post appears to
contain a sign typo.

## Initial and boundary data

The reference Gaussian is used:

\[
u(0,\rho)=\exp[-(\rho-0.5)^2/(2(0.04)^2)],\qquad
\psi(0,\rho)=\partial_\rho u,\qquad
\pi(0,\rho)=0.
\]

No boundary values are imposed. At the horizon and future null infinity,
the characteristic speeds are directed out of the computational domain.
Adding boundary values would overdetermine the causal problem.

## Discretization

- Dedalus 3 with a Chebyshev-T basis on \(0\leq\rho\leq1\)
- First-order hyperbolic system evolved with the RK222 IMEX timestepper
- Variable-coefficient linear terms are treated by Dedalus as part of the
  linear operator
- Default production parameters: \(N=384\), \(\Delta\tau=0.02M\), and
  \(\tau_{\max}=1000M\)

## Validation

The workflow performs four independent checks:

1. Monitors the reduction constraint
   \(C=\psi-\partial_\rho u\).
2. Compares waveforms across Chebyshev resolutions.
3. Compares waveforms across timesteps.
4. Fits the ringdown and the late-time power law. For \(M=1,\ell=2\),
   the fundamental Schwarzschild mode is approximately
   \(M\omega=0.37367168-0.08896232i\), while the expected tail exponents
   are \(-4\) at infinity and \(-7\) at fixed finite radius.

The reference frequency is taken from the published Schwarzschild
quasinormal-mode data collected by
[Emanuele Berti](https://pages.jh.edu/eberti2/ringdown/).

The fundamental-mode fit uses \(40M\leq\tau\leq100M\), after the prompt
response and early overtone contamination have decayed. The tail fit uses
\(400M\leq\tau\leq900M\).
