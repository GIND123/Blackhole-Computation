# Physical Review D manuscript

This directory contains the standalone manuscript source. The paper presents
a fixed-background scalar-wave benchmark of artificial cosmology and does not
claim to regularize the nonlinear conformal Einstein equations.

## Build

From this directory, run:

```sh
latexmk -pdf SdS.tex
```

All graphics required by `SdS.tex` are stored under `figs/`. The generated
`SdS.bbl` is tracked so that the REVTeX source package does not require BibTeX
at submission time; `SdS_refs.bib` remains the editable bibliography.

The standalone APS source bundle consists of `SdS.tex`, `SdS.bbl`,
`SdS_refs.bib`, and the PDF files in `figs/`; the manuscript does not read any
file outside this directory.

## Figure regeneration

The foliation figure is evaluated directly from the analytic bridge formulas:

```sh
python make_foliation_figure.py
```

The numerical result figures can be regenerated from the frozen public
archives by running this command from the repository root:

```sh
python paper/make_submission_figures.py
```

The submission versions use embedded TrueType fonts and do not modify the
frozen simulation archives or their published analysis products.

## Submission statements for author confirmation

These drafts are not included in `SdS.tex`. Confirm their factual details
before submission; do not treat the AI-disclosure templates as final text.

### Data and software availability

Draft for the APS submission form, subject to confirming that the public
release contains all data and software supporting the submitted results:

> The data and numerical software supporting this article are publicly
> available in the project repository [1].

Proposed reference [1]: A. Zenginoğlu and G. Arun Kumar,
*Blackhole-Computation* (2026), GitHub,
<https://github.com/GIND123/Blackhole-Computation>.
Use a permanent archive identifier instead if one is assigned. Include the
final data/software citation in the reference list and submission statement;
no commit hashes or directory inventory are needed.
[APS data-availability guidance](https://journals.aps.org/authors/data-availability-statements).

### AI assistance

OpenAI Codex was used in this project; Govind's use of Anthropic Claude has
also been confirmed. The complete tool/model versions and the checks
personally performed by each author still need confirmation.

Methods draft:

> OpenAI Codex [model/version] and Anthropic Claude [model/version] assisted
> with code development, debugging, and numerical analysis. The authors
> directed this work and verified the resulting calculations through
> [specify the analytic, convergence, and cross-implementation checks
> personally reviewed by the authors].

Acknowledgments draft:

> OpenAI Codex [model/version] and Anthropic Claude [model/version] also
> assisted with manuscript development. [Describe the authors' review and
> verification of the scientific text and references.]

Confirm which tools assisted with each task rather than assigning every task
to both. Identify any retained figures whose generation involved substantive
AI assistance and add an accurate disclosure in their captions. Excluded
illustrations are not part of the submission.
[APS AI-use guidance](https://journals.aps.org/authors/appropriate-use-ai-tools).
