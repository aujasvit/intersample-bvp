"""Experiment 9 - map of hidden overshoot over the gain plane.

Double integrator, tau = 1, x0 = [1, 0] (as in the paper). For every stabilizing (k1, k2):
ratio = (true peak |y(t)| for t >= tau) / (peak of the samples |y_k| for k >= 1).
The true intersample curve inside an interval is the parabola
y(k + s) = q_k + v_k s - (k1 q_k + k2 v_k) s^2 / 2   (constant held force).
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from core import COL, PAPER_EXAMPLES, RES, savefig, style, write_table, plt

BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
N = 25


def peak_ratio(k1, k2, x0=(1.0, 0.0)):
    """Vectorized over arrays k1, k2 (same shape). Returns ratio, stable mask."""
    a11, a12, a21, a22 = 1 - k1 / 2, 1 - k2 / 2, -k1, 1 - k2
    tr, det = a11 + a22, a11 * a22 - a12 * a21
    disc = np.sqrt((tr ** 2 - 4 * det).astype(complex))
    rho = np.maximum(np.abs((tr + disc) / 2), np.abs((tr - disc) / 2))
    stable = rho < 1 - 1e-9
    q = np.full(k1.shape, x0[0])
    v = np.full(k1.shape, x0[1])
    s = np.linspace(0, 1, 41)[:, None]
    true_pk = np.zeros(k1.shape)
    samp_pk = np.zeros(k1.shape)
    for k in range(N):
        if k >= 1:
            u = -(k1 * q + k2 * v)
            ys = q.ravel()[None, :] + v.ravel()[None, :] * s + 0.5 * u.ravel()[None, :] * s ** 2
            true_pk = np.maximum(true_pk, np.abs(ys).max(axis=0).reshape(k1.shape))
            samp_pk = np.maximum(samp_pk, np.abs(q))
        q, v = a11 * q + a12 * v, a21 * q + a22 * v
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(samp_pk > 1e-9, true_pk / samp_pk, np.nan)
    return ratio, stable


def run():
    style()
    k1v = np.linspace(0.02, 4.0, 321)
    k2v = np.linspace(0.02, 3.5, 301)
    K1, K2 = np.meshgrid(k1v, k2v)
    ratio, stable = peak_ratio(K1, K2)
    R = np.where(stable, ratio, np.nan)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad("#ecebe7")
    im = ax.pcolormesh(K1, K2, np.clip(R, 1, 4), cmap=cmap, norm=LogNorm(1, 4), shading="auto")
    cs = ax.contour(K1, K2, np.nan_to_num(R, nan=0), levels=[1.1, 1.5, 2.0], colors="#fcfcfb", linewidths=1.2)
    ax.clabel(cs, fmt={1.1: "+10 %", 1.5: "+50 %", 2.0: "2×"}, fontsize=8.5,
              manual=[(1.35, 0.85), (2.0, 1.6), (2.6, 1.72)])
    ax.axhline(2.0, color=COL["truth"], ls="--", lw=1.0)
    ax.text(3.95, 2.05, "k2 = 2/τ : Q singular (BVP undefined)", ha="right", va="bottom", fontsize=8.5)
    for ex, mk in zip(PAPER_EXAMPLES, ["o", "s", "^"]):
        r, _ = peak_ratio(np.array([ex["K"][0]]), np.array([ex["K"][1]]))
        ax.plot(*ex["K"], mk, color="#fcfcfb", mec=COL["truth"], ms=9, mew=1.5)
        ax.annotate(f"{ex['name'].split(' (')[0]}  ({r[0]:.2f}×)", ex["K"], xytext=(9, -3),
                    textcoords="offset points", fontsize=8.5, color=COL["truth"],
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))
    ax.text(0.25, 3.3, "unstable", fontsize=9, color=COL["ink2"])
    cb = fig.colorbar(im, ax=ax, ticks=[1, 1.1, 1.25, 1.5, 2, 3, 4])
    cb.ax.set_yticklabels(["1×", "1.1×", "1.25×", "1.5×", "2×", "3×", "≥4×"])
    cb.set_label("true peak / peak seen in the samples")
    ax.set_xlabel("position gain k1")
    ax.set_ylabel("velocity gain k2")
    ax.set_title("How much the samples under-report the peak (double integrator, τ = 1, x0 = [1, 0])", loc="left",
                 fontsize=10)
    ax.grid(False)
    savefig(fig, "exp9_gain_map.png")

    vals = R[stable & np.isfinite(R)]
    rows = [["stable gains with ratio > 1.1", float(np.mean(vals > 1.1))],
            ["stable gains with ratio > 1.5", float(np.mean(vals > 1.5))],
            ["stable gains with ratio > 2", float(np.mean(vals > 2.0))],
            ["median ratio over stable region", float(np.median(vals))],
            ["max ratio over stable region", float(np.max(vals))]]
    for ex in PAPER_EXAMPLES:
        r, _ = peak_ratio(np.array([ex["K"][0]]), np.array([ex["K"][1]]))
        rows.append([f"ratio at {ex['name']}", float(r[0])])
    md = write_table(RES / "exp9_gain_map.csv", ["quantity", "value"], rows, fmt="{:.3g}")
    return "## Experiment 9 - hidden-overshoot map over the gain plane\n\n" + md


if __name__ == "__main__":
    print(run())
