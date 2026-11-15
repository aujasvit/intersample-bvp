"""Experiment 8 (extension) - third-order plant.

The paper only treats n = 2. Generalization: with n = 3 use three consecutive samples,
x_k = O^{-1} [y_k, y_{k+1}, y_{k+2}],  O = [C; C Acl; C Acl^2],  y(k tau + s) = C Phi(s) x_k.
Plant 1/(s(s+1)(s+3)) (motor with an extra lag), tau = 0.5, poles {0.6 e^{+-j60deg}, 0.3}.
Also: least squares over 6 samples under noise.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, RES, linear_interp, place_gain, savefig,
                  simulate_linear, spline_interp, style, write_table, plt)

TAU = 0.5
N = 16
SIGMA = 0.005
MC = 300


def run(seed=2):
    style()
    rng = np.random.default_rng(seed)
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -3.0, -4.0]])
    B = np.array([[0.0], [0.0], [1.0]])
    C = np.array([[1.0, 0.0, 0.0]])
    th = np.deg2rad(60)
    K = place_gain(A, B, TAU, [0.6 * np.exp(1j * th), 0.6 * np.exp(-1j * th), 0.3])
    bvp3 = BVPInterpolator(A, B, C, K, TAU, m=3)
    bvp6 = BVPInterpolator(A, B, C, K, TAU, m=6)
    t = np.linspace(0, bvp6.valid_until(N), 3000)
    tk = np.arange(N) * TAU
    sim = simulate_linear(A, B, C, K, TAU, (1.0, 0.0, 0.0), N, t)
    y3 = bvp3.reconstruct(sim["yk"], t)
    yl = linear_interp(tk, sim["yk"], t)
    ysp = spline_interp(tk, sim["yk"], t)

    R3, R6 = bvp3.operator(t, N), bvp6.operator(t, N)
    acc = {"linear": 0.0, "spline": 0.0, "bvp3": 0.0, "bvp6": 0.0}
    for _ in range(MC):
        yn = sim["yk"] + SIGMA * rng.standard_normal(N)
        acc["linear"] += np.mean((linear_interp(tk, yn, t) - sim["y"]) ** 2)
        acc["spline"] += np.mean((spline_interp(tk, yn, t) - sim["y"]) ** 2)
        acc["bvp3"] += np.mean((R3 @ yn - sim["y"]) ** 2)
        acc["bvp6"] += np.mean((R6 @ yn - sim["y"]) ** 2)
    noisy = {k: float(np.sqrt(v / MC)) for k, v in acc.items()}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, sim["y"], color=COL["truth"], lw=1.0, ls=(0, (4, 3)), label=LABEL["truth"], zorder=5)
    ax.plot(t, y3, color=COL["bvp"], lw=2.4, label="BVP, 3 samples (n = 3 extension)")
    ax.plot(t, yl, color=COL["linear"], ls="--", lw=1.6, label=LABEL["linear"])
    ax.plot(t, ysp, color=COL["spline"], ls=":", lw=2.0, label=LABEL["spline"])
    sel = tk <= t[-1]
    ax.plot(tk[sel], sim["yk"][sel], "o", color=COL["truth"], ms=5.5, label="Samples", zorder=6)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output y")
    ax.set_title("Third-order plant 1/(s(s+1)(s+3)), τ = 0.5 s", loc="left")
    ax.legend(loc="upper right")
    savefig(fig, "exp8_third_order.png")

    def rm(y):
        return float(np.sqrt(np.mean((y - sim["y"]) ** 2)))

    rows = [["noise-free", rm(yl), rm(ysp), rm(y3), float("nan")],
            [f"noisy σ={SIGMA}", noisy["linear"], noisy["spline"], noisy["bvp3"], noisy["bvp6"]]]
    header = ["case", "RMSE linear", "RMSE spline", "RMSE BVP (3 samples)", "RMSE BVP-LS (6 samples)"]
    md = write_table(RES / "exp8_third_order.csv", header, rows, fmt="{:.3g}")
    return (f"## Experiment 8 - third-order extension\n\ncond(O_3) = {bvp3.cond:.3g}, "
            f"cond(O_6) = {bvp6.cond:.3g}\n\n" + md)


if __name__ == "__main__":
    print(run())
