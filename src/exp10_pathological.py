"""Experiment 10 - pathological sampling: the BVP condition = sampled observability.

Lightly damped mode (1 Hz, zeta = 0.02), y = position. Classical result (Kalman-Ho-Narendra):
sampled observability is lost when omega_d * tau = k * pi. The BVP matrix Q = [C; C Phi(tau)]
is exactly that observability matrix, so cond(Q) must blow up at tau = k T/2.
(a) cond(Q) vs tau/T for K = 0 and a light damping feedback.
(b) tau = 0.9 T : samples alias to a slow wave; BVP still recovers the 1 Hz oscillation.
(c) tau = T     : samples look like a slow decay; Q singular, BVP cannot recover it.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, RES, linear_interp, savefig, simulate_linear, style,
                  write_table, plt)

W0, ZETA = 2 * np.pi, 0.02
A = np.array([[0.0, 1.0], [-W0 ** 2, -2 * ZETA * W0]])
B = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
WD = W0 * np.sqrt(1 - ZETA ** 2)
T = 2 * np.pi / WD


def run():
    style()
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :])
    taus = np.linspace(0.05, 1.6, 1200) * T
    rows = []
    for K, col, ls, lab in (((0.0, 0.0), COL["bvp"], "-", "no feedback (K = 0)"),
                            ((0.0, 0.8), COL["linear"], "--", "light damping feedback K = [0, 0.8]")):
        conds = [BVPInterpolator(A, B, C, K, tau).cond for tau in taus]
        ax.semilogy(taus / T, np.minimum(conds, 1e8), color=col, ls=ls, label=lab)
    for k in (1, 2, 3):
        ax.axvline(k / 2, color=COL["ink2"], lw=0.8, ls=":")
        ax.text(k / 2 + 0.01, 3e4, f"τ = {k}·T/2", fontsize=8.5, color=COL["ink2"], va="top")
    ax.set_xlabel("sampling period τ / oscillation period T")
    ax.set_ylabel("cond(Q)")
    ax.set_title("(a) BVP matrix Q loses rank exactly where theory says sampled observability is lost "
                 "(ω_d τ = kπ)", loc="left")
    ax.set_ylim(1, 1e8)

    for j, frac in enumerate((0.9, 1.0)):
        tau = frac * T
        N = 14
        t = np.linspace(0, (N - 1) * tau, 4000)
        tk = np.arange(N) * tau
        sim = simulate_linear(A, B, C, (0.0, 0.0), tau, (0.5, 5.0), N, t)
        I = BVPInterpolator(A, B, C, (0.0, 0.0), tau)
        yb = I.reconstruct(sim["yk"], t)
        yl = linear_interp(tk, sim["yk"], t)
        err = float(np.sqrt(np.mean((yb - sim["y"]) ** 2)))
        rows.append([f"tau = {frac} T", I.cond, err, float(np.sqrt(np.mean((yl - sim["y"]) ** 2)))])
        ax2 = fig.add_subplot(gs[1, j])
        ax2.plot(t, sim["y"], color=COL["truth"], lw=0.9, label=LABEL["truth"])
        ax2.plot(t, yl, color=COL["linear"], ls="--", lw=1.6, label=LABEL["linear"])
        ax2.plot(t, yb, color=COL["bvp"], lw=2.0, alpha=0.85, label=LABEL["bvp"])
        ax2.plot(tk, sim["yk"], "o", color=COL["truth"], ms=5, zorder=5, label="Samples")
        ttl = ("(b) τ = 0.9 T: samples alias to a slow wave, BVP recovers the true 1 Hz motion"
               if frac < 1 else "(c) τ = T (pathological): Q singular, nothing can recover the oscillation")
        ax2.set_title(ttl + f"\n     cond(Q) = {I.cond:.3g},  BVP RMSE = {err:.3g}", loc="left", fontsize=9.5)
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("y")
        ax2.set_xlim(0, t[-1])
        if j == 0:
            leg_top = ax.legend(loc="upper left")
            ax.add_artist(leg_top)
            h, l = ax2.get_legend_handles_labels()
            ax.legend(h, l, loc="upper right", title="bottom panels", fontsize=8.5, title_fontsize=8.5)
    savefig(fig, "exp10_pathological_sampling.png")
    md = write_table(RES / "exp10_pathological.csv", ["case", "cond(Q)", "RMSE BVP", "RMSE linear"], rows,
                     fmt="{:.3g}")
    return "## Experiment 10 - pathological sampling and observability\n\n" + md


if __name__ == "__main__":
    print(run())
