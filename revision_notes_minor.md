# Final focused revision — private notes

No commit, push, public release, or change to historical raw data is part of
this revision. Existing unrelated worktree changes are preserved.

## Baseline and scope

The canonical source is `paper/SdS.tex`. The specifically named
`SdS(20260926-224923).pdf` was not found in the workspace or attachments.
The unmodified source was force-built with `latexmk -g -pdf`; it gives the
stated **21 pages**. The snapshot is `/tmp/sds-minor-revision.H6N9tk/`.

Using the same `texcount -inc -sum=1,1,1,0,0,0,0 -sub=section SdS.tex`
method as before, baseline main prose (front matter through Conclusions,
including headings/captions) is **5,462 words** and complete source prose
including appendices is **8,395 words**. Whole-PDF whitespace tokens,
including bibliography and extracted mathematics, are **15,206**.
TeXcount's existing multiline-documentclass warning does not prevent the
section counts. The baseline build has ordinary underfull boxes and the
existing APS bibliography-control warning.

The exterior-ringdown and matched-tail simulations are retained unchanged.
The only new evolutions authorized here concern the operator-cutoff check
of retained single-domain calculations. No new physical family, length,
coupling, frequency fit, or large-L sequence is introduced.

## Verified analytical and descriptive changes

- The illustrative outgoing expansion now uses `Uhat=t-r_*0`, not the
  finite-radius hyperboloidal time `U=tau-q0`. No evolution or analysis
  clock changes. From `B0=-1+8M^2/r^2`,
  `(h0+r_*0)'=8M^2/[r(r-2M)]`, so
  `U-Uhat=-4M log[r/(r-2M)]=-8M^2/r+O(r^-2)`.
  The outgoing radial equation is
  `f u_rr+f' u_r-2u_rUhat-[ell(ell+1)/r^2+2M/r^3]u=0`.
  Substitution gives `2G'=ell(ell+1)F` at fixed Uhat. At fixed U,
  `G_U=G+8M^2F'` and
  `2G_U'=ell(ell+1)F+16M^2F''`. These signs were checked analytically;
  no unavailable computer-algebra test is claimed.
- Appendix B.4 now gives the actual local transport operator
  `C_e=HBD-D^T HB+EB`. The code assembles its skew part locally and adds
  only the surviving physical endpoint terms; no solver change or second
  addition of EB is needed. With `H=J Hhat`, `D=Dhat/J`, all mapping factors
  agree with the implemented mass/stiffness/transport assembly.
- Continuous interfaces cancel their EB terms. M is positive diagonal,
  K is symmetric, and the energy identity is semidiscrete only; positivity
  still requires K positive semidefinite. Degree-8, 50-digit static
  assemblies for Schwarzschild, uniform640 and exterior640 give maximum
  SBP, energy-identity, and K-symmetry defects of 1.34e-51, 5.35e-51,
  and 2.14e-50 respectively. No evolution was needed for this check.
- The mesh description is extracted from `standard_boundaries()` and the
  actual configurations. It documents the existing base positions,
  quartered layer interval, and conditional r/M=80,120,160,200 insertions;
  it does not change the mesh. The unexplained Hermite comparison is gone.
- Price-law language consistently describes a compatible intermediate
  rate, not complete tail-waveform recovery or a finite-L asymptotic law.
  The 3072 and 640 acceptance rules remain distinct and unchanged.
- Figure6 now marks the accepted uniform-conformal interval with light
  shading and a readable duration label, dynamically obtained from the same
  interval rows as Table IV. All eight scientific CSV/NPZ files are
  byte-identical; five signals, axes, masks, fits and dimensions are unchanged.

## Completed numerical cutoff audit

The configuration audit is in `minor_operator_scope.md`. Frozen fixed-mode
and crossover calculations put transport and potential on the implicit side;
the large-L tail calculation treats the potential explicitly. Their resolved
Dedalus cutoffs are 1e-6 (NCC) and 1e-12 (assembled entries). Localized-source
production is finite difference, while the refined exterior comparisons use
the independent SBP formulation; neither shares this Dedalus assembly path.

Both new L3072/Schwarzschild trajectories completed normally from tau=0
through U=450, with N2048, dt=.0025, RK222, dealias1.5, signal cadence .05M,
snapshot cadence25M, and cutoffs1e-10/0. All physical data, clocks, output
times and configurations match the archived controls. The local Dedalus3.0.4
and archived3.0.5 radial operator mathematics were checked against the
official version diff: no relevant change was found. Platform/dependency
differences are nevertheless not isolated, so these are additional repeat
sensitivities, not exclusively attributable cutoff errors.

On the unchanged published231.88–381.88M interval:

| Tail-signal-normalized diagnostic | SdS | Schwarzschild |
|---|---:|---:|
| Maximum relative RMS-envelope change | 2.50431e-4 | 2.41444e-4 |
| Relative waveform L2 difference | 2.31487e-4 | 2.17035e-4 |
| Maximum absolute local-index change | 2.65947e-4 | 3.48965e-4 |
| Maximum sampled absolute C | 5.74001e-9 | 2.34991e-9 |

These envelope changes exceed the original finest spatial changes by factors
5.25 and10.98; they must not be described as subdominant to those changes.
The original table remains a spatial/temporal refinement table, not an
absolute-accuracy bound. Evolved-state implicit constraint forcing is at
most2.1531e-15 and8.5476e-16 respectively; explicit forcing is zero. Constraint
normalization uses the fixed nonzero M||G_v||, never displacement S0 or the
tiny tail signal, and is not a waveform-error estimate.

Both runs pass every unchanged condition: both indices within5% of3,
mutual agreement within.15, an anchored150M interval containing300M, and
the original two-finest-envelope1% masks. The continuous qualifying interval
changes from209.05M to208.30M; matched N2048 departure changes from381.33001M
to380.58001M (-.75M), selecting230.58001–380.58001M. This is a fixed-N
sensitivity check, not a new spatial ladder. The manuscript's original
finest-grid231.88–381.88M interval is not replaced by a medium-grid result.
The68.12M distance from the published endpoint to the new record end exceeds
the full25M padding required by the composed10M RMS/40M local fit.

An independent implementation using explicit201-sample RMS windows and
direct801-sample least-squares fits reproduces the metrics and classification.
No fitted shifts, changed windows, new tolerance, or prompt-peak normalization
was used. Two concise sentences in Appendix B.2 report the additional
sensitivity and its scope, rather than asserting numerical clearance.

Production/tightened N3072 initial operator tests give implicit constraint
defects1.83e-14/1.64e-14 and zero explicit defect; these short tests do not
validate its long trajectory. Separate zero-step L640 quadrupole audits use
N128 at production,1e-10/0,1e-14/0 and0/0, and N768 at production and1e-10/0.
The fine-grid actual-datum constraint forcing stays at2.15e-13/7.11e-13,
unlike the earlier exterior forcing. However, complete RHS changes reach
8.16e-6 on the actual datum and2.44e-5 on a constrained test field. A small
constraint defect does not establish waveform cutoff insensitivity.

### Remaining scope and judgment

- The restricted intermediate Price-rate claim survives this test. The
  index change is about0.23% of the existing.15 allowance, and the departure
  shift is below the original2M spatial spread. No additional trajectory
  was launched for that rate-only claim.
- A second stricter trajectory, and possibly a same-platform production
  baseline, would be needed to establish an amplitude cutoff plateau or
  isolate pure cutoff sensitivity at the old finest-grid precision. That
  stronger claim is **not** made. This is a deliberate scope restriction,
  not a passing cutoff-independence result.
- The U1984 long record has not been newly cutoff-validated. No extrapolation
  of the U450 test to that record is made.
- Configuration inspection covers the retained uniform paths, but static
  checks remain representative, not all-member waveform tests. Other flat
  members, finite-radius crossover trajectories, cumulativeU160 norms, and
  the full weighted extrapolants have not received direct cutoff regressions.
  No unchecked waveform changes are set to zero or propagated as if known.
  The entire manuscript is therefore **not** declared cutoff-cleared.

## Files and checks

Changed manuscript/figure files: `paper/SdS.tex`, compiled PDF;
`black_hole/sbp_matched_tail_figure.py`, `paper/make_submission_figures.py`,
`tests/test_sbp_matched_tail_figure.py`, the new-package figure PDF/PNG and
manifest, `paper/figs/matched_tail_comparison.pdf`, and regenerated evidence
index documents. The annotation's eight focused tests pass and all 72
new-package hashes verify. The evidence index reports 11 entries with no
consistency problems. No historical manifest or raw archive was altered.

Additional new files are `black_hole/uniform_cutoff_audit.py`,
`black_hole/uniform_cutoff_analysis.py`, `black_hole/uniform_operator_scope.py`,
their focused tests, the two isolated result directories, and the private
scope audit. The analysis verifies all11 archived/new input hashes and three
source hashes. New outputs are in
`results/uniform_cutoff_revision_v1/analysis/`; no new public package is made.

### Final verification

The final `latexmk -pdf` build succeeds. It has no undefined references or
citations, overfull boxes, or stuck floats; ordinary underfull-box messages
remain. Visual inspection of pages 13, 15, and 16 confirms that the revised
tail annotation, sensitivity paragraph, table caption, and formulation fit
legibly without collisions or layout compression.

| Consistent measure | Before | After |
|---|---:|---:|
| PDF pages | 21 | 21 |
| Main-text prose count | 5462 | 5462 |
| Complete-source prose count | 8395 | 8327 |
| Whole-PDF whitespace-token count | 15206 | 15195 |

The prose counts use `texcount -inc -sum=1,1,1,0,0,0,0 -sub=section SdS.tex`
before and after, including headings and captions but excluding equations
and bibliography; the main count ends with Conclusions. The supplemental
PDF count also includes extracted mathematics and references.

The focused final suite passes 102 tests, covering SBP geometry and time
integration, pilot analysis, figure annotation, the evidence index, cutoff
analysis, and the high-precision backend. Two passive-instrumentation tests
also passed earlier. `git diff --check` passes.

No further evolution was launched after the completed cutoff runs. No
commit or push was made.
