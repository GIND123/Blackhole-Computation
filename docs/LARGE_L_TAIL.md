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
outer intervals last \(7.20M\) and \(129.75M\), respectively. The latter runs
from \(U=173.98M\) to \(303.73M\), with maximum SdS and Schwarzschild
refinement differences below \(1.9\times10^{-4}\). That interval does contain
the \(U=300M\) anchor, so \(L/M=2560\) fails on duration alone: \(129.75M\) is
short of the preregistered \(150M\). The screening sequence therefore
continues to \(L/M=5120\).

The \(L/M=5120\) screen passes. Its accepted outer interval runs from
\(U=154.13M\) to \(474.98M\), a continuous duration of \(320.85M\) that
contains \(U=300M\). The maximum envelope differences between \(N=1536\)
and \(2048\) are 0.150 percent for Schwarzschild de Sitter and 0.095 percent
for Schwarzschild. Thus the calculation, rather than a prior choice, fixes
\(L_*/M=5120\).

The \(L/M=2560\) interval is the one quantity in this table that is sensitive
to the integration details, because its endpoints are threshold crossings of
\(p_{\mathrm{eff}}\) against the five percent band rather than a converged
number. Under the earlier implicit-potential split (below) the same screen
returned \(98.80M\) from \(U=173.88M\) to \(272.68M\). Both values reject the
length, and the \(L/M=5120\) interval is identical under either split.

## Integration of the potential term

The three evolved equations are split for the IMEX timestepper.  The transport
terms carry the stiffness of the Chebyshev discretization and are integrated
implicitly.  The zeroth-order term \(P\,u\) is bounded, with
\(\max|P|=1.5\) on every background used here and
\(\Delta\tau\sqrt{\max|P|}=3.1\times10^{-3}\), so it is not stiff and is
carried on the explicit side.

Keeping it implicit is what the earlier campaign did, and it is very expensive
on a large-\(L\) bridge.  The Chebyshev spectrum of \(P\) is broad when the
compactification packs a large areal range against \(\rho=1\): at
\(L/M=5120\) and \(N=1536\) its retained bandwidth is 301 coefficients against
2 on Schwarzschild.  A wide non-constant coefficient turns the banded
subproblem matrices nearly dense, which cost an order of magnitude in the
per-step solve and over two orders in matrix construction, without buying any
stability.  Measured at \(N=1536\), \(L/M=5120\):

| quantity | implicit \(P u\) | explicit \(P u\) |
| --- | --- | --- |
| matrix construction | 163.2 s | 0.1 s |
| time per step | 8.91 ms | 0.75 ms |
| resident memory | 2.84 GB | 0.21 GB |

The explicit split runs at the same speed as the Schwarzschild reference,
which is the expected behavior once the artificial density is removed.

Both splits discretize the same continuum system, and the change is validated
against the frozen implicit-split archive rather than assumed.  Re-running the
\(L/M=5120\) screening case reproduces the archived waveform to a maximum
relative difference of \(1.0\times10^{-7}\) at \(\mathcal H_c^+\),
\(3.8\times10^{-8}\) at \(r=8M\), and \(6.1\times10^{-8}\) at \(r=16M\), with a
smaller maximum constraint violation (\(4.6\times10^{-10}\) against
\(2.1\times10^{-9}\)).  The screening decision is unchanged.

One provenance consequence is recorded rather than hidden. The four screening
lengths and their two Schwarzschild references were rerun with the explicit
split. The implicit-split set was moved to
`results/large_l_tail_implicit_split/` rather than discarded, so both are
available, but the new archives were written before the solver change was
committed and each of them records `git_worktree_dirty` as true. The v3 manifest builder refuses archives
with that flag, and these are therefore screening data rather than manifest
grade production data. The waveform difference between the two splits is
\(10^{-7}\) relative, five orders of magnitude below the one percent
refinement criterion, and the screening decisions are identical under either
split, so nothing in the reported screen depends on which set is used. The
final ladder is run from the committed solver.

The split is recorded in every archive under `metadata["imex_split"]`.  The
default remains implicit so that the frozen production archives reproduce
byte-for-byte; only this campaign opts in, through
`large_l_tail.EXPLICIT_POTENTIAL`, and it does so for the SdS waveform and its
Schwarzschild reference alike so that the two are integrated identically.

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

Every case is an independent evolution writing its own archive, so the ladder
parallelizes exactly.  [run_tail_campaign.sh](../run_tail_campaign.sh) runs the
screening set five at a time and the final ladder four at a time, slowest case
first, and then writes the reports.

After an operating system interruption, first confirm that the earlier Python
process is absent, then load the reserved case with

    python -m black_hole.large_l_tail campaign --resume-interrupted

The implementation is
[black_hole/large_l_tail.py](../black_hole/large_l_tail.py). Its synthetic
regression tests recover exact \(U^{-3}\) and \(e^{-\kappa_cU}\) rates before
the production archives are analyzed.
