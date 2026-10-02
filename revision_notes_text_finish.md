# Final text corrections — private summary

## Scope and source

Edited the canonical `paper/SdS.tex` and rebuilt `paper/SdS.pdf`. The named
`SdS(20260927-005557).pdf` was not found in the checked workspace; the canonical
source builds to the specified 21-page baseline. A before-edit source/PDF
snapshot is in `/tmp/sds-text-finish.DeHuX6/`. Existing unrelated worktree
changes were preserved.

No new simulations, operator audits, validation campaigns, scientific figure
generation, or changes to methods, data, clocks, windows, or acceptance rules
were performed. No commit or push was made.

## Corrections

- IV.C now summarizes the spatial/timestep endpoint spread and the separate
  N=2048 cutoff/platform comparison. The fine-grid endpoint remains 381.88M;
  the -0.75M shift is explicitly assigned to the N=2048 comparison. The
  detailed envelope values remain in Table III, not repeated in the main text.
- IV.C and B.2 describe the longer record as spatial/timestep-refined. The
  U/M=1983.58 endpoint, kappa_c U=0.645, and unresolved crossover are unchanged.
  The cutoff/platform comparison is not extended beyond U=450M.
- B.2 retains the regression settings and measured maxima, identifies its
  N=2048 departure endpoint, and states the platform contribution and the
  limitation on cutoff-insensitivity claims. No cutoff row or combined error
  estimate was added to Table III.
- A.1 now states that the quoted uniform fixed-mode scales describe
  spatial/timestep refinement at production operator settings, without an
  independent cutoff contribution. The localized-source discussion is kept
  separate. IV.A no longer equates refinement with a complete accuracy budget.
- B.4 replaces the ambiguous upper-bound rho_0 with the inner layer/template
  boundary. Fixed template positions in the controls remain explicit.
- B.5 identifies uniform conformal SdS as the only **finite-L** passing case,
  preserving Schwarzschild's passing classification and the following
  endpoint-limit interpretation.
- Redundant wording in the same affected paragraphs was shortened to offset
  the added qualifications. No typography, margins, or figure sizes changed.

## Existing evidence inspected

`minor_operator_scope.md` and `revision_notes_minor.md` document the existing
audit limits. `results/uniform_cutoff_revision_v1/analysis/summary.json`
confirms the N=2048 dipole comparison and its -0.75M endpoint shift.
`results/revision_sbp_ringdown_v1/analysis/compact_summary.csv` contains
independent fixed-mode comparisons only through U/M=80 for L/M=320,640.
These are not cutoff-only perturbations and do not cover cumulative U/M=160
or every member of the nested extrapolants. The recorded static operator
checks likewise do not supply a finite-time waveform comparison. Therefore
the requested scope sentence, rather than a blanket assurance, was used.

The mesh wording was checked against `standard_boundaries()` in
`black_hole/sbp_hermite.py` and archived exterior-tail and ringdown
`configuration.json` files. The wave-resolved layout inserts the specified
areal radii only between 0.97 and `layer[0]`, then quarters the layer.
Exterior runs use the physical inner layer boundary; controls use the fixed
template boundary. No configuration discrepancy was found.

No material contradiction was found in the inspected records. The broader
fixed-mode cutoff contribution remains unquantified by these records, as
stated in the manuscript; no new audit was undertaken.

## Counts and checks

| Measure | Before | After |
|---|---:|---:|
| PDF pages | 21 | 21 |
| Main prose, front matter through Conclusions | 5462 | 5437 |
| Complete source prose, including appendices | 8327 | 8323 |
| Whole-PDF whitespace tokens | 15195 | 15179 |

Both prose counts use
`texcount -inc -sum=1,1,1,0,0,0,0 -sub=section SdS.tex`, including headings
and captions but excluding equations and bibliography. The existing
multiline-documentclass identification warning is unchanged. The supplemental
whole-PDF count uses `pdftotext SdS.pdf - | wc -w` and includes extracted
mathematics and references.

`latexmk -pdf -interaction=nonstopmode -halt-on-error SdS.tex` succeeds.
Cross-references resolve; there are no undefined citations, overfull boxes,
or stuck floats. Ordinary underfull-box messages remain. The affected pages
were visually inspected for collisions and overflow.

Before/after source comparisons confirm that the abstract, conclusions,
all 103 displayed equation/align environments, all six figure environments,
all four tables, and both acceptance-protocol paragraphs are unchanged.
The corrected outgoing-coordinate expansion is untouched. SHA-256 comparisons
confirm every file under `paper/figs`, including figure data, and both
bibliography files are unchanged. The source diff contains only the listed
text corrections and adjacent wording reductions; reported results are
unchanged. `git diff --check` passes.
