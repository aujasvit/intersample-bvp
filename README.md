# intersample-bvp

Reproduction and stress tests of Haba & Budai (2025), *Determining the intersample response of state feedback control systems using boundary value problems*, Boundary Value Problems 2025:71.

A sampled-data controller only sees the plant every τ seconds, but the plant keeps moving in between. The paper recovers the exact continuous output between two samples by treating the samples as the boundary conditions of a boundary value problem:

```
y(kτ + s) = C Φ(s) Q⁻¹ [y_k; y_{k+1}],   Φ(s) = A_d(s) − B_d(s) K,   Q = [C; C Ã_d]
```

This repository reimplements the method, reproduces the paper's examples and motor experiments, and adds stress tests (noise, model error, friction, pathological sampling, constraint-violation detection) and extensions (computation delay, observer-based control, an input-aware BVP, a Kalman-smoother baseline).

## Contents

All code is in `code/`. See `code/README.md` for what each file does.

| File | What it does |
|---|---|
| `code/core.py` | ZOH matrices, the paper's BVP interpolator, baselines, simulators, metrics |
| `code/core_ext.py` | Extensions: any-controller simulator with delay, augmented BVP, input-aware BVP, Kalman smoother |
| `code/exp1_*.py` ... `code/exp15_*.py` | The 15 experiments |
| `code/run_all.py` | Runs every experiment and writes a summary |

## Running

```bash
cd code
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python run_all.py      # all experiments, about 10 s
```

Figures are written to `figs/`, tables to `results/`, and a summary to `results/summary.md` (created on first run).
