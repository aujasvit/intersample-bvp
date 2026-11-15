"""Experiment 5 - measurement noise and conditioning.

(a,b) Paper Examples 1 and 2 with Gaussian noise added to the recorded samples y_k
      (the controller itself uses the true state). Methods: linear, spline, BVP (2 samples,
      the paper) and BVP least-squares over 4 samples (extension).
(c)   Noise amplification of the BVP formula vs the velocity gain k2 (k1 = 2, tau = 1):
      cond(Q) and max_s ||C Gamma(s)||_2. Q is singular at k2 = 2/tau = 2.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, PAPER_EXAMPLES, RES, double_integrator, linear_interp,
                  savefig, simulate_linear, spline_interp, style, write_table, plt)

N = 20
MC = 300
SIGMAS = np.logspace(-4, -0.5, 15)


def run(seed=1):
    style()
    rng = np.random.default_rng(seed)
    A, B, C = double_integrator()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = []
    for ax, ex in zip(axes[:2], PAPER_EXAMPLES[:2]):
        bvp2 = BVPInterpolator(A, B, C, ex["K"], 1.0, m=2)
        bvp4 = BVPInterpolator(A, B, C, ex["K"], 1.0, m=4)
        t = np.linspace(0, bvp4.valid_until(N), 2000)
        tk = np.arange(N)
        sim = simulate_linear(A, B, C, ex["K"], 1.0, ex["x0"], N, t)
        R2, R4 = bvp2.operator(t, N), bvp4.operator(t, N)
        curves = {m: [] for m in ("linear", "spline", "bvp", "bvp_ls")}
        for sig in SIGMAS:
            acc = {m: 0.0 for m in curves}
            for _ in range(MC):
                yn = sim["yk"] + sig * rng.standard_normal(N)
                acc["linear"] += np.mean((linear_interp(tk, yn, t) - sim["y"]) ** 2)
                acc["spline"] += np.mean((spline_interp(tk, yn, t) - sim["y"]) ** 2)
                acc["bvp"] += np.mean((R2 @ yn - sim["y"]) ** 2)
                acc["bvp_ls"] += np.mean((R4 @ yn - sim["y"]) ** 2)
            for m in curves:
                curves[m].append(np.sqrt(acc[m] / MC))
        for m, c, ls, mk in (("linear", COL["linear"], "--", "s"), ("spline", COL["spline"], ":", "D"),
                             ("bvp", COL["bvp"], "-", "o"), ("bvp_ls", COL["model"], "-.", "P")):
            ax.loglog(SIGMAS, curves[m], color=c, ls=ls, marker=mk, ms=4.5, label=LABEL[m])
        ax.loglog(SIGMAS, SIGMAS, color=COL["ink2"], lw=0.8, ls=(0, (1, 2)))
        ax.text(3e-4, 2e-4, "RMSE = σ", fontsize=8, color=COL["ink2"], ha="left", va="top", rotation=0)
        ax.set_title(f"{ex['name']}", loc="left")
        ax.set_xlabel("noise std σ on samples")
        ax.set_ylabel("RMSE vs true output")
        # crossover where BVP gets worse than the spline
        cross = next((float(s) for s, b, sp in zip(SIGMAS, curves["bvp"], curves["spline"]) if b > sp), None)
        rows.append([ex["name"], bvp2.cond, curves["linear"][0], curves["spline"][0], curves["bvp"][0],
                     curves["bvp_ls"][0], curves["bvp"][-1], curves["bvp_ls"][-1], curves["spline"][-1],
                     "none" if cross is None else f"{cross:.3g}"])
    axes[0].legend(loc="lower right")

    # (c) conditioning vs k2
    ax = axes[2]
    k2s = np.linspace(0.5, 3.5, 601)
    conds, gains = [], []
    s_grid = np.linspace(0, 1, 101)
    for k2 in k2s:
        if abs(k2 - 2.0) < 1e-9:
            conds.append(np.nan)
            gains.append(np.nan)
            continue
        I = BVPInterpolator(A, B, C, (2.0, k2), 1.0)
        conds.append(I.cond)
        gains.append(max(np.linalg.norm(C @ I.Gamma(s)) for s in s_grid))
    ax.axvspan(1.0, 2.0, color=COL["grid"], zorder=0)
    ax.text(1.5, 2e2, "closed loop\nstable", ha="center", fontsize=8.5, color=COL["ink2"])
    ax.semilogy(k2s, conds, color=COL["bvp"], label="cond(Q)")
    ax.semilogy(k2s, gains, color=COL["linear"], ls="--", label="max$_s$ ‖C Γ(s)‖  (noise gain)")
    ax.axvline(2.0, color=COL["ink2"], lw=0.8, ls=":")
    ax.text(2.05, 1.2e3, "Q singular\nk2 = 2/τ", fontsize=8.5, color=COL["ink2"], va="top")
    for ex in PAPER_EXAMPLES[:1]:
        ax.plot(ex["K"][1], BVPInterpolator(A, B, C, ex["K"], 1.0).cond, "o", color=COL["truth"], ms=6)
        ax.annotate("Example 1", (ex["K"][1], BVPInterpolator(A, B, C, ex["K"], 1.0).cond),
                    xytext=(-60, 10), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("velocity gain k2  (k1 = 2, τ = 1)")
    ax.set_title("Noise amplification of the BVP formula", loc="left")
    ax.set_ylim(0.8, 1e4)
    ax.legend(loc="upper right")
    savefig(fig, "exp5_noise.png")

    header = ["example", "cond(Q)", "RMSE linear @σ=1e-4", "RMSE spline @σ=1e-4", "RMSE BVP @σ=1e-4",
              "RMSE BVP-LS4 @σ=1e-4", "RMSE BVP @σ=0.32", "RMSE BVP-LS4 @σ=0.32", "RMSE spline @σ=0.32",
              "σ where BVP becomes worse than spline"]
    md = write_table(RES / "exp5_noise.csv", header, rows, fmt="{:.3g}")
    return "## Experiment 5 - measurement noise\n\n" + md


if __name__ == "__main__":
    print(run())
