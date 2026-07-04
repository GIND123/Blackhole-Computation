# Schwarzschild black-hole perturbations with Dedalus

This repository evolves axial gravitational perturbations of a
non-rotating black hole using the hyperboloidally compactified
Regge--Wheeler equation described by Professor Anıl Zenginoğlu.

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

## References

- [Banging a black hole — Anıl Zenginoğlu](https://anilzen.github.io/post/2026/black-hole-gravitational-waves/)
- [Reference Colab notebook](https://colab.research.google.com/drive/1ii9zyHE9MlaOW4e9K1IBGPSPQzxltO7c)
- [Dedalus v3 documentation](https://dedalus-project.readthedocs.io/en/latest/)
- [Published Schwarzschild quasinormal-mode data](https://pages.jh.edu/eberti2/ringdown/)
