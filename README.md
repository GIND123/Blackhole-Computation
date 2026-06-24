# Blackhole-Computation

This workspace contains the first Dedalus check requested by Professor Zenginoglu:
install Dedalus and solve a simple wave equation.

## Dedalus Setup

Dedalus was installed in WSL Ubuntu 22.04 using Miniforge:

```bash
/home/govind/miniforge3/bin/mamba create -y -n dedalus3 -c conda-forge dedalus matplotlib
/home/govind/miniforge3/bin/conda env config vars set -n dedalus3 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
```

Verify the install:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 python -c 'import dedalus; print(dedalus.__version__)'
```

Run the wave-equation example:

```bash
/home/govind/miniforge3/bin/mamba run -n dedalus3 python examples/wave_equation_1d.py
```

The script solves the periodic 1D wave equation

```text
u_tt = c^2 u_xx
```

by rewriting it as a first-order-in-time initial value problem for Dedalus:

```text
dt(u) = v
dt(v) = c^2 dx(dx(u))
```

It writes:

- `outputs/wave_equation_1d.png`
- `outputs/wave_equation_1d_final.csv`
