# Focused manuscript revision — private working notes

These notes are not part of the submission manuscript. No commit, push, release,
or change to old raw archives is authorized or performed in this revision.

## Baseline and length accounting

- Source: `paper/SdS.tex`; existing unrelated worktree changes were preserved.
- The specifically named `SdS(20260926-195807).pdf` was not found in the scoped
  workspace/attachment search. The actual source was force-compiled before editing:
  **22 pages**, matching the stated reference length. The hard limit remains 22.
- Baseline snapshot: `/tmp/sds-focused-revision.RnxLI0/` (source, PDF, bibliography).
- Reproducible prose count: run `texcount -inc -sum=1,1,1,0,0,0,0 -sub=section SdS.tex`
  from `paper/`. Count text, headings, and captions; exclude equations/math and
  bibliography. Main text means front matter through Conclusions: **5,722 words**.
  Complete source prose, including appendices: **8,912 words**.
- Whole compiled PDF, including references and mathematical text, gives **15,701
  whitespace-delimited tokens** with `pdftotext SdS.pdf - | wc -w`. This is a
  supplementary, consistent whole-document check, not a linguistic word count.
- TeXcount reports that it cannot identify the multiline document-class declaration;
  its section counts remain available. `-incbib` additionally misparses REVTeX
  bibliography macros, so that result is not used as an exact word count.
- Baseline build succeeded but reported a stuck float, underfull boxes, and the
  existing APS bibliography-control warning. Final checks and counts are below.

## Verified analytical corrections

- Distinguish the regular coordinate/formulation limit from convergence of the
  physically matched waveform. At the cosmological horizon, `U=t-r_*` follows
  exactly from `tau=t+h` and `q=lim(h+r_*)`. A divergent bridge offset is not a
  proof that an exactly retimed observable fails to converge.
- Principal-coefficient conditions are adopted formulation criteria. The
  Mavrogiannis clock is excluded only by this particular comparison protocol.
- Common areal profiles/Killing derivatives are data on different, converging
  slices, not identical Cauchy data on one hypersurface.
- The residues of `1/f`, whose large-r behavior is O(r^-2), sum to zero:
  `1/(2 kb)-1/(2 kc)+1/(2 ku)=0`. Thus the shortened clock formula is exact.
  Original formula, simplified formula, and defining quadrature were compared at
  70 digits for L/M=20,80,320,640,3072. Largest simplified/integral discrepancy:
  1.43e-67 M. The existing double-precision clock differs by at most 1.63e-12 M.
  No production clock code or waveform alignment was changed.
- At L/M=640 the actual profile has R1/M=440.433010739587 and
  (R1/L)^2=0.473587004270354. This is a lapse-function deformation, not a
  gauge-invariant norm. The transition mass increment is 104.291675 M.
- Monotonicity gives `(f_chi/r^2)'=-2/r^3+6M/r^4-chi'/L^2<0` outside R0>6M:
  no additional exterior circular null orbit, without claims about timelike
  orbits, finite-ell potential extrema, or global QNM spectra.
- Conformal coupling leaves `r lambda'/9+r^2 lambda''/18` in the transition.
  These terms are not isolated as the sole cause of secondary lobes.
- In outgoing Schwarzschild coordinates the equation is
  `-2u_Ur+f u_rr+f' u_r-[ell(ell+1)/r^2+f'/r]u=0`.
  Substituting `u=F(U)+G(U)/r+O(r^-2)` gives
  `2G'=ell(ell+1)F` at order r^-2. This illustrates compatible 1/r and L^-2
  scales; it does not derive the measured SdS bias or its coefficient.

## Numerical work and choice of improved solver

The independent continuous LGL/SBP solver is useful for the matched tail and
quadrupole cross-checks. It is not substituted for the requested direct
shared-flux constraint audit. Radau IIA, not the available Hermite integrator,
is used because its L stability damps unresolved stiff grid modes. The physical
profiles, geometric clocks, data, norms, and tail acceptance criteria are fixed.

- `results/ringdown_constraint_revision_v1/`: passive inspection of actual
  Dedalus assembled implicit/explicit RHS, initial bump and polynomial test data,
  fully discrete stages, and sampled constraint histories. The fixed spatial
  completed-target ladder is N=512,768,1024 at dt=.01 M plus N=768 at .005 M
  for L=320,640. Cost amendments were recorded before inspecting any shared-flux
  waveform ordering: first the timestep, then the spatial ladder after the
  N2048 setup/throughput trials proved impractical. Their interrupted logs
  remain excluded; no completed N2048 campaign is claimed. The final choice
  is not the original frozen timestep or spatial ladder; adequacy is tested
  through actual refinement and the independent solver.
- `results/revision_sbp_ringdown_v1/`: independent ell=2, xi=0 displacement
  runs for Schwarzschild and uniform/exterior L=320,640; degrees24/32/40 at
  dt=.05 M plus degree32 at .025 M, through U=80 M. No QNM fits.
- `results/sbp_matched_tail_revision_v1/`: fill the missing uniform-minimal640
  and coarse exterior controls, then reanalyze all five matched signals using
  the original 10 M envelope, 40 M fit, numerical-floor and zero-crossing
  masks, 10%/40 M Price criterion, and nine-width sensitivity sweep.
  Existing completed pilot runs are reused read-only. No new large-L sequence.

## Initial diagnostic finding

At N=128, the old assembled transport operator gives constraint-RHS residuals
2.65e-7 for the actual bump and 2.29e-9 for polynomial data, versus 2.13e-13
and approximately 2e-17 for the shared-flux form. The explicit potential
contribution is zero in both. Tightening only old nonconstant-coefficient and
matrix truncation cutoffs reduces these defects to roundoff. This identifies
a discrete transport consistency issue, not physical constraint growth or an
initial-data error. High-resolution waveform consequences must be assessed
from the actual campaign, not inferred from this small-grid diagnostic.

## Final results, changed files, and checks

### Completed independent calculations

- All 20 planned quadrupole SBP/Radau runs completed. On U/M=15–45:

  | L/M | uniform error (%) | exterior error (%) | relative reduction (%) |
  |---:|---:|---:|---:|
  | 320 | 3.181313841 | 2.772981739 | 12.835329 |
  | 640 | 1.553498873 | 1.419706905 | 8.612299 |

  The paper rounds these values conservatively. The ringdown-window
  ordering survives all three spatial degrees and the timestep check.
  Across the three ringdown windows, largest successive waveform changes
  are 7.60e-7 and 4.93e-8; timestep halving changes at most 2.16e-6.
  These are fractions normalized by the common Schwarzschild norm, not
  bounds. The 0–80M cumulative window has a larger 8.96e-5 medium/fine
  change, so its accuracy is not transferred to the disjoint late signal.
- Both finest shared-flux checks completed: their signed
  waveform differences from SBP are 8.6720e-6 (L320) and 1.1192e-5 (L640)
  on the nominal window, far smaller than the 0.0040833 and 0.0013379
  improvement gaps. The complete shared-flux ladder is summarized below.
- Read-only inspection found substantial stored constraint drift in the
  older L80/160 exterior runs too. Their quantitative Table II rows were
  removed, rather than expanding the requested campaign or claiming their
  constraint behavior had been repaired. Private legacy comparisons and
  every original archive are preserved.
- All six additional tail runs completed. Combined with the existing
  independent runs, all five matched L640 families have degrees24/32/40
  and finest-degree timestep halving. The unchanged criterion passes only
  Schwarzschild and uniform conformal SdS. Primary intervals (M) are
  140.40–975.00, 134.50–144.10 (uniform minimal), 153.25–294.70 (uniform
  conformal), 299.00–312.30 (exterior minimal), and 617.60–626.10
  (exterior conformal). All nine estimator choices preserve the classification.
- The comparison now uses the original matched-640 rule, not the pilot's
  extra Schwarzschild-index-matching condition. No tolerances or physical
  configurations were tuned. The older large-L3072 results are untouched.
- Old/new exterior waveform discrepancies reach 5.46% in the separated
  750–950M window, larger than the earlier merged-window pilot summary.
  The paper now states 0.8–5.5% and uses the independently refined signals
  for Fig.6 and the matched-tail interval table. This changes detailed
  waveform values, not the pass/fail conclusion.

### Files changed or added in this revision

- `paper/SdS.tex`, `paper/SdS_refs.bib`, generated bibliography/PDF.
- `paper/figs/matched_tail_comparison.pdf` replaces the ringdown-residual
  figure in the manuscript; the displaced figure remains on disk unused.
- `paper/make_submission_figures.py` regenerates the new figure from its
  actual five-signal inputs and numerical masks.
- `black_hole/sbp_geometry.py`, `black_hole/sbp_hermite.py`: explicit
  displacement initial-data option and support-aware block boundaries;
  existing velocity-tail defaults unchanged.
- `black_hole/sbp_ringdown_validation.py`: predeclared quadrupole campaign,
  old/new controls, residual-vector and joint-reference comparisons.
- `black_hole/ringdown_constraint_diagnostic.py`,
  `black_hole/run_ringdown_constraint_revision.py`,
  `black_hole/ringdown_constraint_analysis.py`: passive split/stage/history
  instrumentation and the separate shared-flux campaign/analysis.
- `black_hole/sbp_matched_tail_figure.py`,
  `black_hole/sbp_matched_tail_audit.py`: exact existing tail criteria,
  plotting, interval/sweep/refinement/formulation audits.
- New/extended tests for the geometry/initial data, passive constraint
  hooks, and matched-tail masks/estimator equivalence.
- `black_hole/evidence_index.py`, `docs/EVIDENCE_INDEX.md`,
  `docs/evidence_index.json`, and the stale-path test expectation: replace
  only the displaced figure and two revised table mappings, so the index
  no longer points to superseded numerical evidence. New-package output
  hashes are generated by the relevant analysis scripts; historical frozen
  manifests are unchanged.
- The three isolated result directories listed above and this private note.
  No changes were made to old raw production data or the existing geometry,
  scalar equation, or Dedalus solver defaults.

### Completed shared-flux validation

- All eight final-plan runs completed normally through U=80M. At each length,
  N=512,768,1024 use dt=.01M and N=768 also uses dt=.005M. The separate
  interrupted high-N throughput trials are not numerical evidence.
- Across the three ringdown windows, the largest successive spatial waveform
  changes decrease from 1.48464e-4 to 9.59516e-6. Timestep halving changes at
  most 5.11911e-6; shared-flux/SBP differences reach 1.25053e-5 and old/shared
  differences 3.42256e-5. Every number is a waveform norm, not a difference
  of error magnitudes, and is normalized by the fine Schwarzschild norm.
- The sampled absolute constraint maximum is 5.16212e-10 and the fixed-scale
  maximum is 1.34984e-11. Each history has 829 samples at nominal .1M cadence,
  including initial/final times. The initial six-stage checks reach at most
  1.13719e-10. These are sampled diagnostics, not continuous-time bounds.
- The improvement gap is positive for all 24 length/resolution/timestep/window
  combinations on the three ringdown windows. An independent root-agent
  check of the saved waveform vectors confirms a smallest gap of 1.03752e-3.
  It also remains positive on the secondary 0–80M cumulative window.
- Joint reference substitution changes the gap by at most 5.054e-8;
  half-output cadence changes the fine waveform norm by at most 2.371e-11.
  No Richardson extrapolation or rigorous remainder bound is assigned.
- Private signed residual plots show increasing old/new differences after
  50M. The paper therefore retains the disjoint late-window limitation.
  The primary Table II values are the independently refined SBP values,
  not a mixture of old production norms and new error estimates.

### Verification

- Final combined SBP/geometry/backend/analysis/evidence-index run: **83 tests
  passed**. Separately, the two MPI-dependent passive-instrumentation
  regression tests passed. These are focused checks, not a claim that every
  repository test was rerun.
- Independent reviewer verified the weak form, Radau rational map, analytic
  corrections, and the matched-tail table against CSV/JSON. Every original
  citation key is retained; only Gassner's LGL/SBP-operator reference and
  Hairer–Wanner's Radau reference were added, with metadata checked against
  publisher/author primary sources.
- 72 matched-tail input/source/output hashes and 91 quadrupole hashes verified, plus 15
  read-only legacy-audit hashes and all 92 shared-flux input/output hashes.
  Scientific tail CSV/NPZ values are unchanged
  by the panel-label relocation; the paper plotting wrapper reproduces the
  identical PNG and PDF apart from the creation timestamp.
- The updated evidence index checks all 11 current entries with zero
  consistency problems. The final matched-tail PDF differs from the prior
  rendering only in its creation timestamp and is recopied into `paper/figs`;
  scientific CSV/NPZ hashes and rendered PNG bytes are unchanged.
- Fig.6, the new compact Table II, the matched-tail appendix table, and
  neighboring manuscript pages were visually inspected. Table I fits one
  column without a font change; this removes the baseline stuck-float warning
  and places the large-L tail figure closer to its discussion. Fonts, margins,
  line spacing, and all figure footprints remain unchanged.
- The final source compiles with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error SdS.tex`.
  There are no undefined references/citations, overfull boxes, or stuck floats.
  Ordinary underfull-box messages remain; the earlier fresh BibTeX run also
  reported the existing APS bibliography-control warning.
- Final analytical/consistency review found no remaining substantive
  contradiction. The abstract explicitly distinguishes finite-L errors from
  extrapolated recovery. No publication/availability statement was added.

### Final length checks and remaining limitations

| Consistent measure | Before | After |
|---|---:|---:|
| Compiled pages | 22 | 21 |
| Main-text prose including front matter, headings, captions | 5,722 | 5,462 |
| Complete source prose including appendices | 8,912 | 8,395 |
| Whole-PDF tokens including bibliography/math | 15,701 | 15,206 |

The complete prose count falls by 5.8%, and the main-text count by 4.5%.
There are still six numbered figures. The final new-figure page and adjacent
text, compact tables, and Appendix C audit pages were inspected visually.
No font, margin, or line-spacing reduction was used.

No calculation needed for the revised quantitative claims remains pending.
The old L80/160 exterior constraint behavior has not been revalidated and is
not used for the improved-ringdown claim. Neither the later cosmological
regime nor complete tail-waveform recovery is claimed. The exact named
reference PDF was unavailable; the initial compiled source supplied the
recorded baseline. No old raw archives were overwritten or deleted.
