# Large cosmological length dipole tail study

This calculation asks whether one finite Schwarzschild de Sitter waveform
contains both a resolved Schwarzschild Price tail and the later cosmological
exponential decay. The earlier crossover runs answered a different question.
They measured departure before the Schwarzschild tail became established.

The physical data are the compact pure dipole velocity data already used in
the crossover study. The reduced field and its radial derivative vanish at
the initial slice. The Killing time derivative is the same smooth function
of areal radius on every background, supported on \(3M<r<9M\). The evolved
momentum is initialized as \(\pi=G/A\). This keeps the physical initial
velocity fixed as the geometry changes.

The primary pair is the field at \(\mathcal H_c^+\) and an independently
evolved Schwarzschild field at \(\mathscr I^+\). Observers at \(r=8M\) and
\(r=16M\) are retained as secondary diagnostics. All signals use
\(U=\tau-q_L\), with \(q_L\) evaluated analytically. No relative time shift is
fitted.

For a centered RMS amplitude \(A(U)\), the analysis evaluates

\[
p_{\mathrm{eff}}=-\frac{d\ln A}{d\ln U},\qquad
\frac{\gamma_{\mathrm{eff}}}{\kappa_c}
=-\frac{1}{\kappa_c}\frac{d\ln A}{dU}.
\]

The derivatives come from local linear regressions. The Price estimator uses
a \(40M\) log time window. The primary cosmological estimator uses
\(\Delta(\kappa_c U)=0.25\); the final sensitivity calculation also uses
0.15 and 0.4. Samples below a scale dependent floating point floor are
discarded.

## Screening decision

Screening begins at \(L/M=640\) and \(1280\), with \(N=1536,2048\),
\(\Delta\tau=0.0025M\), RK222, and \(U/M\geq500\). A length passes only if the
SdS and Schwarzschild asymptotic values of \(p_{\mathrm{eff}}\) agree with each
other and with 3 to within five percent for one continuous interval of at
least \(150M\). The secondary fixed radius curves are checked against the
dipole target \(p=5\). Throughout each accepted interval the two resolution
envelopes must differ by less than one percent of the local fine envelope.
The same test is applied to the independent Schwarzschild refinement pair.
The primary accepted interval must contain \(U=300M\), which implements the
requested check that the resolved tail persists through that time.

If a length fails, the next value is tested. The sequence is \(640,1280,2560,
5120\). The first passing value is \(L_*\). This decision is written to JSON
and CSV before any final run is started.

The completed \(L/M=640\) screen is rejected. At \(U=300M\), the fine outer
SdS value is \(p_{\mathrm{eff}}=3.856\), while the matched Schwarzschild value
is 3.015. The longest joint interval inside the five percent bands is only
\(2.55M\), from \(U=32.94M\) to \(35.49M\). This is far shorter than the
required \(150M\) and does not contain \(U=300M\). The rejection is physical,
not a refinement failure. The maximum envelope refinement differences on
that short interval are \(3.3\times10^{-9}\) for SdS and
\(3.2\times10^{-9}\) for Schwarzschild.

The \(L/M=1280\) and 2560 screens are also rejected. Their longest accepted
outer intervals last \(7.25M\) and \(98.80M\), respectively. The latter runs
from \(U=173.88M\) to \(272.68M\), with maximum SdS and Schwarzschild
refinement differences below \(1.1\times10^{-4}\). At \(U=300M\), the fine
\(L/M=2560\) index is 3.135 and the Schwarzschild index is 3.015. A small
excursion above the five percent SdS band near \(U=275M\) breaks continuity,
so the result does not meet the preregistered \(150M\) requirement. The
screening sequence therefore continues to \(L/M=5120\).

The \(L/M=5120\) screen passes. Its accepted outer interval runs from
\(U=154.13M\) to \(474.98M\), a continuous duration of \(320.85M\) that
contains \(U=300M\). The maximum envelope differences between \(N=1536\)
and \(2048\) are 0.147 percent for Schwarzschild de Sitter and 0.095 percent
for Schwarzschild. Thus the calculation, rather than a prior choice, fixes
\(L_*/M=5120\).

## Final calculation

The final case reaches \(\kappa_cU\geq4\). It uses \(N=1536,2048,3072\) at
\(\Delta\tau=0.0025M\), plus an \(N=2048\) check at
\(\Delta\tau=0.00125M\). The matched Schwarzschild references reach the same
retarded time. The main figure has aligned panels for \(A\),
\(p_{\mathrm{eff}}\), and \(\gamma_{\mathrm{eff}}/\kappa_c\). Price departure,
cosmological entry, and the interval between them are reported separately.
Cosmological entry requires the normalized rate to stay within ten percent
of unity through the last resolved sample for at least
\(\Delta(\kappa_cU)=0.4\). It is not assigned when the numerical floor ends
the signal first.
The numerical sensitivity table repeats the transition measurement at all
three spatial resolutions and at the halved timestep. A second table gives
the maximum envelope change across the Price and cosmological intervals for
both backgrounds. The signal floor sweep records the corresponding fraction
of the peak waveform explicitly.

The runner never overwrites a completed archive. It first creates a running
reservation, writes an incomplete archive, and publishes the final NPZ only
after a successful evolution. Long evolutions also write an atomic checkpoint
every \(500M\). Each checkpoint contains the three evolved fields, all output
histories, the Dedalus iteration and simulation time, and the accumulated wall
time. A restarted run checks the full physical and numerical configuration
before loading that state.

From the pinned Dedalus environment, one screening stage is run with

    OMP_NUM_THREADS=1 python -m black_hole.large_l_tail screen 640

The next length is started only if the JSON decision rejects the completed
one. Named cases and the final ladder can be inspected with

    python -m black_hole.large_l_tail list
    python -m black_hole.large_l_tail final 1280

The complete campaign can be resumed without repeating completed archives:

    python -m black_hole.large_l_tail campaign

After an operating system interruption, first confirm that the earlier Python
process is absent, then load the reserved case with

    python -m black_hole.large_l_tail campaign --resume-interrupted

The implementation is
[black_hole/large_l_tail.py](../black_hole/large_l_tail.py). Its synthetic
regression tests recover exact \(U^{-3}\) and \(e^{-\kappa_cU}\) rates before
the production archives are analyzed.
