"""Experiment 6 - model mismatch.

True plant: motor with a_true = a (1 + da) or b_true = b (1 + db). The interpolators use the
NOMINAL model (paper eq. 5.1). Controller: same tau as paper Exp 1, but a stable design
(discrete poles 0.85 e^{+-j75deg}, full state feedback) so instability does not mask the effect.
Compared: linear, spline, BVP (nominal model), forward model simulation from the true x0.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, MOTOR_A, MOTOR_B, RES, linear_interp, motor, nrmse,
                  place_gain, savefig, simulate_linear, spline_interp, style, write_table, plt)

TAU = 0.089118
X0 = (0.6, 0.0)
T_END = 3.0
DELTAS = np.linspace(-0.3, 0.3, 13)


def case(A_true, B_true, A_nom, B_nom, C, K, N, t):
    tk = np.arange(N) * TAU
    sim = simulate_linear(A_true, B_true, C, K, TAU, X0, N, t)
    est = {"linear": linear_interp(tk, sim["yk"], t), "spline": spline_interp(tk, sim["yk"], t),
           "bvp": BVPInterpolator(A_nom, B_nom, C, K, TAU).reconstruct(sim["yk"], t),
           "model": simulate_linear(A_nom, B_nom, C, K, TAU, X0, N, t)["y"]}
    return sim, est


def run():
    style()
    A_nom, B_nom, C = motor()
    th = np.deg2rad(75)
    K = place_gain(A_nom, B_nom, TAU, [0.85 * np.exp(1j * th), 0.85 * np.exp(-1j * th)])
    N = int(T_END / TAU) + 1
    t = np.linspace(0, (N - 1) * TAU, 3000)
    out = {"a": {m: [] for m in ("linear", "spline", "bvp", "model")},
           "b": {m: [] for m in ("linear", "spline", "bvp", "model")}}
    for which in ("a", "b"):
        for d in DELTAS:
            a = MOTOR_A * (1 + d) if which == "a" else MOTOR_A
            b = MOTOR_B * (1 + d) if which == "b" else MOTOR_B
            A_t, B_t, _ = motor(a, b)
            sim, est = case(A_t, B_t, A_nom, B_nom, C, K, N, t)
            for m in est:
                out[which][m].append(nrmse(sim["y"], est[m]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, which, lab in ((axes[0], "a", "viscous pole a"), (axes[1], "b", "input gain b")):
        for m, ls, mk in (("linear", "--", "s"), ("spline", ":", "D"), ("model", "-.", "P"), ("bvp", "-", "o")):
            ax.plot(100 * DELTAS, out[which][m], color=COL[m], ls=ls, marker=mk, ms=4.5, label=LABEL[m])
        ax.set_xlabel(f"error in {lab} of the model used for reconstruction [%]")
        ax.set_ylabel("normalized RMSE")
        ax.set_title(f"Mismatch in {lab}", loc="left")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper center")

    # time plot, b off by +20 %
    A_t, B_t, _ = motor(MOTOR_A, MOTOR_B * 1.2)
    sim, est = case(A_t, B_t, A_nom, B_nom, C, K, N, t)
    ax = axes[2]
    ax.plot(t, sim["y"], color=COL["truth"], lw=1.0, label="True output")
    ax.plot(t, est["model"], color=COL["model"], ls="-.", label=LABEL["model"])
    ax.plot(t, est["bvp"], color=COL["bvp"], lw=2.0, label=LABEL["bvp"])
    ax.plot(np.arange(N) * TAU, sim["yk"], "o", color=COL["truth"], ms=4, label="Samples")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [rad]")
    ax.set_title("True b is 20 % larger than the model", loc="left")
    ax.legend(loc="upper right")
    savefig(fig, "exp6_model_mismatch.png")

    idx = [0, 3, 6, 9, 12]
    rows = []
    for which in ("a", "b"):
        for i in idx:
            rows.append([which, float(100 * DELTAS[i]), out[which]["linear"][i], out[which]["spline"][i],
                         out[which]["model"][i], out[which]["bvp"][i]])
    header = ["parameter", "model error [%]", "nRMSE linear", "nRMSE spline", "nRMSE model sim", "nRMSE BVP"]
    md = write_table(RES / "exp6_mismatch.csv", header, rows, fmt="{:.3g}")
    return (f"## Experiment 6 - model mismatch\n\nStable design K = {np.round(K, 5).tolist()} "
            f"(poles 0.85∠±75°, tau = {TAU}).\n\n" + md)


if __name__ == "__main__":
    print(run())
