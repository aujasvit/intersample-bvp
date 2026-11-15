"""Experiment 2 - the paper's DC-motor experiments (Table 1, Figs. 7-9), in simulation.

Two plants for each (tau, K) of Table 1:
  * ideal   : the identified linear model itself (BVP must be exact),
  * realistic: same model + Coulomb/static friction (Fc = 20 rad/s^2, Fs = 1.2 Fc)
               + 2048-count encoder quantization of the samples.
"True" output is logged at 1 kHz as in the paper; RMSE as in paper eq. (5.2).
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, MOTOR_A, MOTOR_B, PAPER_MOTOR_EXPS, RES,
                  linear_interp, motor, savefig, simulate_linear, simulate_motor_real, style,
                  write_table, plt)

FC_NOMINAL = 20.0


def motor_case(e, Fc=FC_NOMINAL, realistic=True):
    """Simulate one experiment; return dict with 1 kHz truth grid, samples and reconstructions."""
    A, B, C = motor()
    K = np.array([[e["k1"], 0.0]])
    N = int(e["T"] / e["tau"]) + 1
    t = np.arange(0, (N - 1) * e["tau"], 1e-3)
    tk = np.arange(N) * e["tau"]
    if realistic:
        r = simulate_motor_real(MOTOR_A, MOTOR_B, e["tau"], e["k1"], e["x0"], N, Fc=Fc)
        y_true = np.interp(t, r["t_fine"], r["y_fine"])
        ys, xk, uk = r["ymeas"], r["xk"], r["uk"]
    else:
        r = simulate_linear(A, B, C, K, e["tau"], e["x0"], N, t)
        y_true, ys, xk, uk = r["y"], r["yk"], r["xk"], r["uk"]
    interp = BVPInterpolator(A, B, C, K, e["tau"])
    return dict(t=t, tk=tk, y_true=y_true, ys=ys, xk=xk, uk=uk, N=N, A=A, B=B, C=C, K=K,
                y_bvp=interp.reconstruct(ys, t), y_lin=linear_interp(tk, ys, t),
                cond=interp.cond, eig=np.linalg.eigvals(interp.Acl))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run():
    style()
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for e, axrow in zip(PAPER_MOTOR_EXPS, axes):
        ideal = motor_case(e, realistic=False)
        real = motor_case(e, realistic=True)
        lam = np.abs(ideal["eig"]).max()
        rows.append([e["name"], e["tau"], e["k1"], lam,
                     e["paper_lin"], e["paper_bvp"], e["paper_lin"] / e["paper_bvp"],
                     rmse(ideal["y_lin"], ideal["y_true"]), rmse(ideal["y_bvp"], ideal["y_true"]),
                     rmse(real["y_lin"], real["y_true"]), rmse(real["y_bvp"], real["y_true"]),
                     rmse(real["y_lin"], real["y_true"]) / rmse(real["y_bvp"], real["y_true"])])
        for ax, key, lab in ((axrow[0], "y_lin", "linear"), (axrow[1], "y_bvp", "bvp")):
            ax.plot(real["t"], real["y_true"], color=COL["truth"], lw=1.0, label="True angle (1 kHz)")
            ax.plot(real["t"], real[key], color=COL[lab], lw=2.0, ls="--" if lab == "linear" else "-",
                    label=LABEL[lab])
            ax.plot(real["tk"][:-1], real["ys"][:-1], "o", color=COL["truth"], ms=4.5, label="Samples (quantized)")
            ax.set_title(f"{e['name']}  (τ = {e['tau']} s, k1 = {e['k1']}):  "
                         f"RMSE = {rmse(real[key], real['y_true']):.4f} rad", loc="left", fontsize=9.5)
            ax.set_ylabel("angle [rad]")
            ax.set_xlim(0, real["t"][-1])
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    savefig(fig, "exp2_motor_paper_figs7-9.png")

    header = ["exp", "tau [s]", "k1", "max|eig| of model closed loop",
              "paper RMSE linear", "paper RMSE BVP", "paper ratio",
              "ideal-plant RMSE linear", "ideal-plant RMSE BVP",
              "realistic RMSE linear", "realistic RMSE BVP", "realistic ratio"]
    md = write_table(RES / "exp2_motor.csv", header, rows)
    note = ("\nIdeal plant = identified linear model (BVP exact by construction). Realistic plant = model + "
            f"Coulomb friction Fc={FC_NOMINAL} rad/s^2 (stiction 1.2 Fc) + 2048-count encoder quantization. "
            "Initial angles/durations read approximately from paper Figs. 7-9.\n")
    return "## Experiment 2 - motor experiments (paper Table 1)\n\n" + md + note


if __name__ == "__main__":
    print(run())
