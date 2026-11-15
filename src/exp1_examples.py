"""Experiment 1 - reproduce the paper's three double-integrator examples (Figs. 3-5).

Checks: (i) our Gamma(s) equals the paper's eqs. (4.4), (4.6), (4.8);
(ii) the BVP reconstruction from samples equals the exact continuous response;
(iii) fast sampling (m = 3, as in the paper's Fig. 3b) lands on the BVP curve.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, PAPER_EXAMPLES, RES, double_integrator,
                  fast_sampling, linear_interp, savefig, simulate_linear, style, write_table, plt)

TAU = 1.0


def run():
    style()
    A, B, C = double_integrator()
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    rows = []
    for ex, ax in zip(PAPER_EXAMPLES, axes):
        N = 10 if "deadbeat" not in ex["name"] else 5
        t = np.linspace(0, (N - 1) * TAU, 3001)
        tk = np.arange(N) * TAU
        sim = simulate_linear(A, B, C, ex["K"], TAU, ex["x0"], N, t)
        interp = BVPInterpolator(A, B, C, ex["K"], TAU)
        y_bvp = interp.reconstruct(sim["yk"], t)
        y_lin = linear_interp(tk, sim["yk"], t)
        ft, fy = fast_sampling(A, B, C, TAU, sim["xk"], sim["uk"], 3)

        gam_err = max(np.abs(interp.Gamma(s) - ex["Gamma"](s)).max() for s in np.linspace(0, 1, 101))
        bvp_err = np.abs(y_bvp - sim["y"]).max()
        lin_err = np.abs(y_lin - sim["y"]).max()
        fast_err = np.abs(np.interp(ft, t, y_bvp) - fy)[ft <= t[-1]].max()
        i_min = np.argmin(sim["y"])
        rows.append([ex["name"], gam_err, bvp_err, fast_err, lin_err,
                     float(sim["y"][i_min]), float(t[i_min]), float(sim["yk"].min()), float(y_bvp.min())])

        ax.plot(t, sim["y"], color=COL["truth"], lw=1.0, ls=(0, (4, 3)), label=LABEL["truth"], zorder=4)
        ax.plot(t, y_bvp, color=COL["bvp"], lw=2.4, label=LABEL["bvp"], zorder=3)
        ax.plot(t, y_lin, color=COL["linear"], lw=1.6, ls="--", label=LABEL["linear"], zorder=2)
        m = ft <= t[-1]
        ax.plot(ft[m], fy[m], "o", mfc="none", mec=COL["fast"], mew=1.6, ms=8,
                label="Fast sampling m=3 (lifting)", zorder=5)
        ax.plot(tk, sim["yk"], "o", color=COL["truth"], ms=6, label="Samples $y_k$", zorder=6)
        ax.plot(t[i_min], sim["y"][i_min], "x", color=COL["truth"], ms=10, mew=2, zorder=7)
        ax.annotate(f"true min {sim['y'][i_min]:.3f}  (lowest sample: {sim['yk'].min():.3f})",
                    (t[i_min], sim["y"][i_min]),
                    xytext=(14, 0) if t[i_min] < 0.5 * t[-1] else (0, 40), textcoords="offset points",
                    fontsize=8.5, color=COL["ink2"], va="center",
                    ha="left" if t[i_min] < 0.5 * t[-1] else "center",
                    arrowprops=None if t[i_min] < 0.5 * t[-1] else dict(arrowstyle="-", color=COL["ink2"], lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.9))
        ax.margins(y=0.12)
        ax.set_title(f"{ex['name']}:  K = [{ex['K'][0]}, {ex['K'][1]}],  x0 = {list(ex['x0'])}", loc="left")
        ax.set_ylabel("output y")
        ax.set_xlim(0, t[-1])
    axes[-1].set_xlabel("time t  (sampling period τ = 1)")
    axes[0].legend(loc="upper right", ncol=2)
    savefig(fig, "exp1_paper_examples.png")

    header = ["example", "max |Gamma - paper|", "max |BVP - truth|", "max |fast3 - BVP|",
              "max |linear - truth|", "true min", "t of true min", "min of samples", "min of BVP curve"]
    md = write_table(RES / "exp1_examples.csv", header, rows)
    return "## Experiment 1 - paper examples (Figs. 3-5)\n\n" + md


if __name__ == "__main__":
    print(run())
