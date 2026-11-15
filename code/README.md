# Code: Haba & Budai (2025) reproduction + stress tests

```bash
cd code
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib   # once
.venv/bin/python run_all.py          # all experiments (~10 s)
.venv/bin/python exp5_noise.py       # or any single experiment
```

Figures go to `../figs/`, tables (CSV) to `../results/`, and everything is summarised in `../results/summary.md`.

| File | What it does |
|---|---|
| `core.py` | ZOH matrices, `Phi(s)`, `BVPInterpolator` (paper's method + n>2 / least-squares extension), baselines, simulators (exact linear; motor with friction + encoder quantization), metrics |
| `exp1_examples.py` | Paper Examples 1-3 (Figs. 3-5): Gamma(s) check, BVP vs truth vs fast sampling |
| `exp2_motor.py` | Paper motor experiments (Table 1, Figs. 7-9), ideal and realistic plant |
| `exp3_comparison.py` | Linear, spline, fast sampling (m=3,10), model simulation, BVP on all datasets |
| `exp4_sampling_sweep.py` | Error vs samples per oscillation period (noise-free and noisy) |
| `exp5_noise.py` | Measurement-noise sweep + conditioning of Q vs k2 (singular at k2 = 2/tau) |
| `exp6_mismatch.py` | Wrong model parameters a, b (+-30 %) |
| `exp7_friction.py` | Unmodelled Coulomb friction (paper Exp. 3 artifact) |
| `exp8_third_order.py` | Extension to a 3rd-order plant (3 samples) and least squares |
| `core_ext.py` | General sampled-data simulator (any controller, computation delay), generalized BVP (`IntervalBVP`), input-aware BVP, Kalman/RTS smoother |
| `exp9_gain_map.py` | Map over (k1, k2): how much the samples under-report the true peak |
| `exp10_pathological.py` | Pathological sampling: cond(Q) blows up exactly at ω_d τ = kπ (sampled observability) |
| `exp11_kalman.py` | Kalman/RTS smoother baseline (exact vs wrong model, tuning sweep) |
| `exp12_detection.py` | Monte Carlo: catching constraint violations between samples |
| `exp13_delay.py` | Extension: computation delay (augmented 3-sample BVP and input-aware BVP) |
| `exp14_observer.py` | Extension: observer-based output feedback (augmented 4-sample BVP and input-aware BVP) |
| `exp15_sysid.py` | Identify the motor from step tests (with friction), then reconstruct |
| `hardware_reconstruct.py` | Run the method on real logged CSV data (`--demo` validates it on simulated logs) |


Assumptions not given in the paper: initial angles and durations of the motor runs are read approximately from Figs. 7-9. The "realistic" motor adds Coulomb friction Fc = 20 rad/s² (stiction 1.2 Fc) and a 2048-count encoder.
