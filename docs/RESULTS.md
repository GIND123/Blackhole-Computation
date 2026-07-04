# Numerical results

The production calculation used \(M=1\), \(\ell=2\), 384 Chebyshev modes,
\(\Delta\tau=0.02M\), and evolved through \(\tau=1000M\). The initial
Gaussian was centered at \(\rho=0.5\) with width \(0.04\).

## Ringdown

The waveform at future null infinity was fitted over
\(40M\leq\tau\leq100M\) to an exponentially damped sinusoid. The later
start avoids fitting the prompt response and the strongest overtone
contamination.

| Quantity | Measured | Reference | Relative error |
|---|---:|---:|---:|
| Oscillation frequency \(M\omega_R\) | 0.37335717 | 0.37367168 | 0.0842% |
| Damping rate \(M\alpha\) | 0.08893283 | 0.08896232 | 0.0332% |

The normalized RMS residual of the fit is \(4.67\times10^{-4}\).
The comparison value comes from the
[published Schwarzschild QNM tables](https://pages.jh.edu/eberti2/ringdown/).

## Late-time tails

A log-log least-squares fit over \(400M\leq\tau\leq900M\) gives:

| Observer | Measured exponent | Expected exponent | \(R^2\) |
|---|---:|---:|---:|
| Future null infinity, \(\rho=1\) | -4.0364 | -4 | 0.999998 |
| Finite radius, \(\rho=0.9\) (\(r=20M\)) | -6.9021 | -7 | 0.999978 |

These reproduce the distinct asymptotic decay rates described in the
reference article.

## Constraint preservation

For the first-order reduction constraint

\[
C=\psi-\partial_\rho u,
\]

the maximum norm over the full evolution was

\[
\max_\tau\|C\|_\infty=3.37\times10^{-11}.
\]

The final value was \(2.39\times10^{-11}\), and the maximum RMS value was
\(1.94\times10^{-12}\).

## Convergence

The spatial study compares the infinity waveform through \(200M\):

| Modes | Difference from next resolution |
|---:|---:|
| 48 | \(8.91\times10^{-2}\) |
| 64 | \(4.41\times10^{-4}\) |
| 96 | \(3.04\times10^{-10}\) |

The rapid error reduction is the expected spectral convergence for smooth
Gaussian data. The 128-mode result is the spatial reference.

The timestep self-convergence study gives an observed order of 2.00 between
\(\Delta\tau=0.02M\) and \(0.01M\), as expected for RK222. The
\(\Delta\tau=0.005M\) result is the temporal reference.

## Generated evidence

- [Spacetime evolution](../results/black_hole/spacetime.png)
- [Waveform at infinity](../results/black_hole/waveform_infinity.png)
- [Tail decay](../results/black_hole/tail_decay.png)
- [Constraint preservation](../results/black_hole/constraint.png)
- [Convergence plot](../results/black_hole/convergence/convergence.png)
- [Machine-readable diagnostics](../results/black_hole/diagnostics.json)
- [Convergence table](../results/black_hole/convergence/convergence.csv)
