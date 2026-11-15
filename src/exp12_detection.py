"""Experiment 12 - catching constraint violations between samples (Monte Carlo).

Random initial states; a limit |y(t)| <= c must hold for t >= tau (c = median true peak,
so half the runs violate). Detect a violation from: the samples, a cubic spline, the BVP
reconstruction (noise-free), and the BVP reconstruction from noisy samples.
Report miss rate (violations not detected) and false-alarm rate.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, PAPER_EXAMPLES, RES, double_integrator, motor, savefig,
                  spline_interp, style, write_table, plt, Phi)
from core_ext import motor_stable_K

MC = 2000


def cases():
    A, B, C = double_integrator()
    out = [dict(name=ex["name"].split(" (")[0], A=A, B=B, C=C, K=np.atleast_2d(ex["K"]), tau=1.0, N=12,
                q0=1.0, v0=2.0, sigma=0.01) for ex in PAPER_EXAMPLES]
    A, B, C = motor()
    out.append(dict(name="Motor (stable design)", A=A, B=B, C=C, K=motor_stable_K(), tau=0.089118, N=25,
                    q0=0.6, v0=6.0, sigma=0.003))
    return out


def run(seed=5):
    style()
    rng = np.random.default_rng(seed)
    rows, plotdata = [], []
    for c in cases():
        A, B, C, K, tau, N = c["A"], c["B"], c["C"], c["K"], c["tau"], c["N"]
        sgrid = np.linspace(0, tau, 41)
        Cphi = np.array([C[0] @ Phi(A, B, K, s) for s in sgrid])          # (S, n)
        Acl = Phi(A, B, K, tau)
        t = (np.arange(1, N - 1)[:, None] * tau + sgrid[None, :]).ravel()  # t in [tau, (N-1) tau]
        R = BVPInterpolator(A, B, C, K, tau).operator(t, N)
        tk = np.arange(N) * tau
        P = {m: np.empty(MC) for m in ("true", "samples", "spline", "bvp", "bvp_noisy")}
        for i in range(MC):
            x = np.array([rng.uniform(-c["q0"], c["q0"]), rng.uniform(-c["v0"], c["v0"])])
            X = [x]
            for _ in range(N - 1):
                X.append(Acl @ X[-1])
            X = np.array(X)
            ys = X @ C[0]
            ytrue = (Cphi @ X[1:N - 1].T).T.ravel()
            P["true"][i] = np.abs(ytrue).max()
            P["samples"][i] = np.abs(ys[1:]).max()
            P["spline"][i] = np.abs(spline_interp(tk, ys, t)).max()
            P["bvp"][i] = np.abs(R @ ys).max()
            P["bvp_noisy"][i] = np.abs(R @ (ys + c["sigma"] * rng.standard_normal(N))).max()
        lim = np.median(P["true"])
        viol = P["true"] > lim
        res = {}
        for m in ("samples", "spline", "bvp", "bvp_noisy"):
            det = P[m] > lim
            res[m] = (float(np.mean(~det[viol])), float(np.mean(det[~viol])))
            rows.append([c["name"], m, res[m][0], res[m][1]])
        plotdata.append((c["name"], res, c["sigma"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    meths = [("samples", "Samples only", COL["linear"]), ("spline", LABEL["spline"], COL["spline"]),
             ("bvp", "BVP (noise-free)", COL["bvp"]), ("bvp_noisy", "BVP (noisy samples)", COL["model"])]
    w = 0.2
    names = [p[0] for p in plotdata]
    for ax, idx, ttl in ((axes[0], 0, "Missed violations (lower is better)"),
                         (axes[1], 1, "False alarms (lower is better)")):
        for j, (m, lab, col) in enumerate(meths):
            vals = [100 * p[1][m][idx] for p in plotdata]
            xs = np.arange(len(names)) + (j - 1.5) * w
            ax.bar(xs, vals, width=w * 0.9, color=col, label=lab)
            for xx, vv in zip(xs, vals):
                ax.text(xx, vv + 1, f"{vv:.0f}", ha="center", fontsize=7.5, color=COL["ink2"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(ttl, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of runs")
    axes[0].legend(loc="upper right", fontsize=8)
    savefig(fig, "exp12_violation_detection.png")
    md = write_table(RES / "exp12_detection.csv", ["case", "detector", "miss rate", "false-alarm rate"], rows,
                     fmt="{:.3f}")
    return ("## Experiment 12 - detecting intersample constraint violations\n\nLimit = median true peak; "
            f"{MC} random initial states per case; noisy case sigma = 0.01 (DI) / 0.003 rad (motor).\n\n" + md)


if __name__ == "__main__":
    print(run())
