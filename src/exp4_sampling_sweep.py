"""Experiment 4 - how sparse can the sampling be?

Double integrator, tau = 1. For each target "samples per oscillation period" Nsp, place
the discrete closed-loop poles at r e^{+-j 2 pi / Nsp} with r^Nsp = 0.5 (amplitude halves
every period), simulate 4 periods from x0 = [1, 0], and compare reconstruction error.
Left: noise-free samples. Right: samples with Gaussian noise sigma = 0.01 (Monte Carlo).
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, RES, double_integrator, linear_interp, nrmse,
                  place_gain, savefig, simulate_linear, spline_interp, style, write_table, plt)

SIGMA = 0.01
MC = 100


def run(seed=0):
    style()
    rng = np.random.default_rng(seed)
    A, B, C = double_integrator()
    nsp_grid = np.linspace(2.5, 20, 36)
    res = {m: [] for m in ("linear", "spline", "bvp")}
    res_n = {m: [] for m in ("linear", "spline", "bvp")}
    for nsp in nsp_grid:
        th = 2 * np.pi / nsp
        r = 0.5 ** (1 / nsp)
        K = place_gain(A, B, 1.0, [r * np.exp(1j * th), r * np.exp(-1j * th)])
        N = int(np.ceil(4 * nsp)) + 2
        t = np.linspace(0, N - 1, 40 * N)
        tk = np.arange(N)
        sim = simulate_linear(A, B, C, K, 1.0, (1.0, 0.0), N, t)
        R = BVPInterpolator(A, B, C, K, 1.0).operator(t, N)
        res["linear"].append(nrmse(sim["y"], linear_interp(tk, sim["yk"], t)))
        res["spline"].append(nrmse(sim["y"], spline_interp(tk, sim["yk"], t)))
        res["bvp"].append(max(nrmse(sim["y"], R @ sim["yk"]), 1e-16))
        acc = {m: [] for m in res_n}
        for _ in range(MC):
            yn = sim["yk"] + SIGMA * rng.standard_normal(N)
            acc["linear"].append(nrmse(sim["y"], linear_interp(tk, yn, t)) ** 2)
            acc["spline"].append(nrmse(sim["y"], spline_interp(tk, yn, t)) ** 2)
            acc["bvp"].append(nrmse(sim["y"], R @ yn) ** 2)
        for m in res_n:
            res_n[m].append(np.sqrt(np.mean(acc[m])))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, rr, title in ((axes[0], res, "Noise-free samples"),
                          (axes[1], res_n, f"Noisy samples (σ = {SIGMA}, {MC} runs)")):
        ax.plot(nsp_grid, rr["linear"], color=COL["linear"], ls="--", marker="s", ms=4, label=LABEL["linear"])
        ax.plot(nsp_grid, rr["spline"], color=COL["spline"], ls=":", marker="D", ms=4, label=LABEL["spline"])
        exact_y = 2e-5   # numerically exact values (< 1e-12) are drawn in an "exact" lane
        ax.plot(nsp_grid, [v if v > 1e-12 else exact_y for v in rr["bvp"]], color=COL["bvp"],
                marker="o", ms=4, label=LABEL["bvp"])
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.axhspan(1e-5, 4e-5, color=COL["grid"], zorder=0)
        ax.set_yticks([exact_y, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        ax.set_yticklabels(["exact\n(<1e-12)", "1e-4", "1e-3", "0.01", "0.1", "1"])
        ax.set_xlabel("samples per oscillation period of the closed loop")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("normalized RMSE  (RMSE / RMS of true output)")
    axes[0].legend(loc="center right")
    savefig(fig, "exp4_sampling_sweep.png")

    pick = [0, 3, 7, 15, 25, 35]
    rows = [[float(nsp_grid[i]), res["linear"][i], res["spline"][i], res["bvp"][i],
             res_n["linear"][i], res_n["spline"][i], res_n["bvp"][i]] for i in pick]
    header = ["samples/period", "nRMSE linear", "nRMSE spline", "nRMSE BVP",
              "noisy nRMSE linear", "noisy nRMSE spline", "noisy nRMSE BVP"]
    md = write_table(RES / "exp4_sampling_sweep.csv", header, rows, fmt="{:.3g}")
    return "## Experiment 4 - sampling-density sweep\n\n" + md


if __name__ == "__main__":
    print(run())
