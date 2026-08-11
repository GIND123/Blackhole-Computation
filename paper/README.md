# PRD manuscript work plan

`SdS.tex` is the shared manuscript scaffold.  It contains the motivation,
global regulator argument, target claims, and section structure.  The
`GOVIND TODO` comments in the source identify the material that should now
be completed from the frozen simulations and final analysis package.

## Scientific hierarchy

The primary question is whether Misner's positive-cosmological-constant
regulator can retain useful asymptotically flat black-hole observables as
`L/M` grows.  The scalar calculation tests the geometry, foliation,
compactification, physical data identification, waveform clock, and
`L -> infinity` extrapolation that a later Einstein calculation would need.

The secondary question concerns physical Schwarzschild--de Sitter wave
propagation, especially caustic echoes and the eventual late-time departure
from Schwarzschild behavior.  These results support and stress-test the
regulator construction; they should not displace it as the main story.

The paper is a scalar black-hole proof of principle.  It is not a numerical
evolution of the full Einstein equations and must not be described as one.

## Supported central claims

- For the prompt-dominated pure-`ell=2` sequence, direct agreement including
  the case-specific numerical margin first reaches 5% at `L/M=320` and 2%
  at `L/M=640`; direct 1% agreement is not reached through 640.  These
  thresholds do not apply to the localized-source runs.
- The nested extrapolants based at `L/M=80` and 160 agree with each other and
  with the independent Schwarzschild result at a conservative 1% level on
  the declared cumulative windows.  The 0.0314% central residual is not a
  resolved accuracy: observed refinement changes reach 0.409%, while the
  propagated Richardson estimate reaches 0.119%.
- For the localized multimode response on the common archived interval,
  the direct sphere-integrated errors are 5.26710% and 2.62569% at
  `L/M=320` and 640.  The nested extrapolants differ by 0.051706%.
- The disjoint late-time waveform windows are diagnostic because refinement
  is not subdominant.  Do not claim uniform 1% late-time accuracy.
- The `D1` curves are consistency evidence for local `O((M/L)^2)` and outer
  `O(M/L)` behavior.  They are not precision coefficient measurements; the
  requested combined deterministic timing-sensitivity targets are not met.

## Govind's next drafting tasks

1. Complete the numerical-method section from the frozen implementation and
   manifest.  Include both resolution ladders, time integration, output
   cadence, angular truncation, the independent Schwarzschild reference,
   and the finite-difference/Dedalus check.
2. Give the full fixed-data and normalized localized-source prescriptions.
   State clearly what is evolved mode by mode and how the angular response is
   reconstructed.  Define every comparison interval and confirm that no
   relative time translation is fitted.
3. Explain the direct norms, the Parseval sphere-integrated norm, and the
   nested extrapolation.  Keep observed refinement changes, Richardson
   estimates, extraction sensitivities, and physical source-width dependence
   conceptually separate.
4. Write the main regulator-results subsections and integrate the principal
   figures and a compact headline table.  The source material is:
   - `../results/regulator_production_v3/flat_waveform_sequence.pdf`
   - `../results/regulator_production_v3/nested_extrapolants.pdf`
   - `../results/regulator_production_v3/localized_source_regulator.pdf`
   - `../results/regulator_production_v3/flat_window_errors.pdf`
   - `../results/regulator_production_v3/tables/paper_localized_source.tex`
5. Add one concise caustic-echo subsection using `D1_scaling.pdf`.  Put the
   timing-sensitivity and phase-exclusion audits in an appendix.  Retain only
   the late-time result that materially supports the regulator narrative.
6. Complete the verification and reproducibility appendices from
   `../docs/REGULATOR.md` and `../results/regulator_production_v3/manifest.json`.
   A permanent DOI should replace the mutable repository citation before
   submission.
7. Prepare self-contained PRD captions, check every number against the final
   CSV tables, and remove each `GOVIND TODO` only when its task is complete.

The professor will retain responsibility for the central motivation,
geometric interpretation, hierarchy of claims, and final editorial pass.

## Small package corrections to make during drafting

- Remove the phrase "the professor's approximate checks" from
  `../docs/REGULATOR.md`.
- Fix the remaining overlapping logarithmic tick labels in
  `flat_window_errors` and `D1_error_separation`.
- In `D1_estimator_window_sensitivity.csv`, make the estimator rows identify
  primary and alternate `D1` values unambiguously; the current difference is
  correct, but the `D1_over_M` field repeats the primary value.
- Add `flat_waveform_sequence.pdf` to the report's list of paper artifacts.

These are presentation/data-label corrections only.  Do not rerun or alter
the frozen production simulations.

## Building

From this directory:

```sh
latexmk -pdf SdS.tex
```

Missing legacy theory figures are shown as labeled placeholders until their
final versions are added under `paper/figs/`.
