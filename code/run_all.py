"""Run every experiment; figures -> ../figs, tables -> ../results, summary -> ../results/summary.md"""
import time

import exp1_examples
import exp2_motor
import exp3_comparison
import exp4_sampling_sweep
import exp5_noise
import exp6_mismatch
import exp7_friction
import exp8_third_order
import exp9_gain_map
import exp10_pathological
import exp11_kalman
import exp12_detection
import exp13_delay
import exp14_observer
import exp15_sysid
from core import RES

EXPERIMENTS = [exp1_examples, exp2_motor, exp3_comparison, exp4_sampling_sweep,
               exp5_noise, exp6_mismatch, exp7_friction, exp8_third_order, exp9_gain_map,
               exp10_pathological, exp11_kalman, exp12_detection, exp13_delay, exp14_observer, exp15_sysid]

if __name__ == "__main__":
    parts = ["# Results summary - Haba & Budai (2025) reproduction and stress tests\n"]
    for mod in EXPERIMENTS:
        t0 = time.time()
        parts.append(mod.run())
        print(f"{mod.__name__:<22s} done in {time.time() - t0:5.1f} s")
    (RES / "summary.md").write_text("\n\n".join(parts))
    print("summary ->", RES / "summary.md")
