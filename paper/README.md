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
