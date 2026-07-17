# Black-hole perturbations and bridge coordinates with Dedalus

This repository evolves axial gravitational perturbations of a
non-rotating black hole using the hyperboloidally compactified
Regge--Wheeler equation described by Professor Anıl Zenginoğlu.

It also includes a Schwarzschild-de Sitter scalar-wave bridge-coordinate
study based on the coordinate choices in *Bridging time across null
horizons*.

The computational interval includes both physically important boundaries:

- `rho = 0`: future event horizon
- `rho = 1`: future null infinity, where the gravitational waveform is read

The complete equations, sign convention, initial data, and numerical method
are documented in [docs/METHOD.md](docs/METHOD.md).

## Repository layout

```text
black_hole/             Reusable solver, analysis, and convergence package
docs/                   Equations, method, and generated result summary
legacy/flat_wave/       Original periodic-wave Dedalus exercise
results/black_hole/     Production data, plots, and diagnostics
results/sds_scalar/     Schwarzschild-de Sitter scalar bridge study
results/flat_wave/      Outputs from the introductory flat-wave exercise
tests/                  Analytic coefficient and initial-data tests
environment.yml         Reproducible conda environment
```

## Environment

The existing WSL environment can be verified with:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 \
  python -c "import dedalus; print(dedalus.__version__)"
```

To recreate it:

```bash
/home/govind/miniforge3/bin/mamba env create -f environment.yml
/home/govind/miniforge3/bin/conda env config vars set \
  -n dedalus3 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
```

Run all commands from the repository root.

## Full study

The single command below runs the \(1000M\) production evolution, generates
all figures and tables, and performs independent spatial and temporal
convergence studies:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 \
  python -m black_hole --verbose study --output-dir results/black_hole
```

For a quick installation check:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 \
  python -m black_hole --verbose simulate \
  --resolution 64 --timestep 0.02 --end-time 2 \
  --signal-dt 0.02 --snapshot-dt 0.1 \
  --output results/smoke_test.npz
```

Analyze an existing run:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 \
  python -m black_hole analyze \
  --input results/black_hole/primary_evolution.npz \
  --output-dir results/black_hole
```

## Schwarzschild-de Sitter scalar bridge study

The command below runs the 1D scalar wave equation on six
Schwarzschild-de Sitter bridge foliations, generates comparison plots, and
performs a focused convergence study:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 \
  python -m black_hole --verbose sds-suite \
  --resolution 192 --timestep 0.02 --end-time 200 \
  --signal-dt 0.05 --snapshot-dt 0.5 \
  --output-dir results/sds_scalar \
  --convergence-end-time 80
```

The controlled flat-limit workflow requested by Professor Zenginoglu runs
\(L=20,40,80,160\), an independent \(\Lambda=0\) Schwarzschild reference,
and convergence studies at \(L=20\) and \(L=160\). It uses an identical
\(C^\infty\) compact pulse in areal radius centered at \(4M\), and aligns the
horizon signals with the analytic geometric time \(U=\tau-\lim(h+r_*)\):

    /home/govind/miniforge3/bin/mamba run -n dedalus3 \
      python -m black_hole --verbose sds-flat-limit \
      --resolution 256 --timestep 0.01 --end-time 200 \
      --signal-dt 0.05 --snapshot-dt 0.5 \
      --convergence-end-time 100 \
      --output-dir results/sds_scalar/flat_limit

The equations, coordinate choices, diagnostics, and generated figures are
summarized in [docs/SDS_SCALAR.md](docs/SDS_SCALAR.md). The controlled
Schwarzschild comparison and numerical findings are documented in
[docs/FLAT_LIMIT.md](docs/FLAT_LIMIT.md).

Run unit tests:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 \
  python -m unittest discover -s tests -v
```

## Principal outputs

The full workflow creates:

- `primary_evolution.npz`: complete reproducible numerical dataset
- `waveforms.csv`: waveforms at infinity and finite-radius observers
- `spacetime.png`: propagation across the compactified exterior
- `waveform_infinity.png`: transient, ringdown, and tail
- `tail_decay.png`: late-time decay and local power index
- `constraint.png`: first-order reduction constraint
- `diagnostics.json`: fitted ringdown, tail, and constraint numbers
- `convergence/`: raw comparison runs, table, and plot

After running the production study, the measured values are summarized in
`docs/RESULTS.md`.

The SdS scalar workflow creates:

- `bridge_boost_characteristics.png`: boost functions and light speeds
- `spacetime_bridge_gallery.png`: side-by-side pulse propagation
- `cosmological_horizon_waveforms.png`: extracted horizon signals
- `horizon_signal_comparison.png`: black-hole and cosmological signals
- `constraint_bridge_comparison.png`: constraint preservation by bridge
- `bridge_summary.csv`: turning radii and constraint errors
- `convergence/`: scalar-wave convergence data and plot

The flat-limit workflow adds the following under
results/sds_scalar/flat_limit:

- waveform_comparison.png: SdS cosmological-horizon signals and the
  Schwarzschild future-null-infinity reference
- waveform_differences.png: requested difference versus aligned time
- flat_limit_norms.png: difference norms as \(L\) increases
- height_alignment.png: common \(h(4M)=0\) normalization
- coordinate_flat_limit.png: minimal-gauge coefficient approach
- constraints.png: constraint preservation for the complete sequence
- convergence/: independent \(L=20\) and \(L=160\) studies
- raw/: reproducible production datasets

## References

- [Bridging time across null horizons](https://arxiv.org/abs/2502.08581)
- [Misner hyperboloidal coordinates](https://anilzen.github.io/post/2023/misner-hyperboloidal/)

- [Banging a black hole — Anıl Zenginoğlu](https://anilzen.github.io/post/2026/black-hole-gravitational-waves/)
- [Reference Colab notebook](https://colab.research.google.com/drive/1ii9zyHE9MlaOW4e9K1IBGPSPQzxltO7c)
- [Dedalus v3 documentation](https://dedalus-project.readthedocs.io/en/latest/)
- [Published Schwarzschild quasinormal-mode data](https://pages.jh.edu/eberti2/ringdown/)
