"""Experiment 15 - end-to-end pipeline: identify the model, then reconstruct.

Step tests (open loop, from rest) on the 'realistic' motor (friction + encoder), position
logged at 1 kHz, as in the paper's identification. Fit theta(t) = (b U / a)(t - (1 - e^{-a t})/a).
Coulomb friction biases the fit (effective gain ~ b - Fc/U). Then reconstruct the paper's
closed-loop experiments with each identified model and compare to the true linear part.
"""
import numpy as np
from scipy.optimize import curve_fit
from core import (BVPInterpolator, COL, ENCODER_STEP, MOTOR_A, MOTOR_B, PAPER_MOTOR_EXPS, RES, motor, savefig,
                  simulate_motor_real, style, write_table, plt)
from exp2_motor import FC_NOMINAL

STEPS = (0.15, 0.3, 0.6, 1.0)


def step_test(U, T=0.4):
    # constant input U from rest: reuse the closed-loop simulator with k1 = 0 and a constant offset
    dt = 1e-3
    a, b, Fc, Fs = MOTOR_A, MOTOR_B, FC_NOMINAL, 1.2 * FC_NOMINAL
    th, om = 0.0, 0.0
    ys = []
    sub = 50
    h = dt / sub
    for _ in range(int(T / dt)):
        ys.append(ENCODER_STEP * np.round(th / ENCODER_STEP))
        for _ in range(sub):
            F = b * U
            if om == 0.0 and abs(F) <= Fs:
                continue
            f = F - Fc * np.sign(om if om != 0 else F)
            om_new = om + h * (-a * om + f)
            th += h * om_new
            om = om_new
    t = np.arange(len(ys)) * dt
    return t, np.array(ys)


def model(t, a, bU):
    return (bU / a) * (t - (1 - np.exp(-a * t)) / a)


def run():
    style()
    ids = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    step_cols = [COL["bvp"], COL["spline"], COL["fast"], COL["model"]]
    for U, sc in zip(STEPS, step_cols):
        t, y = step_test(U)
        (a_hat, bU_hat), _ = curve_fit(model, t, y, p0=(10.0, 200.0 * U))
        ids.append((U, a_hat, bU_hat / U))
        axes[0].plot(t, y, color=COL["truth"], lw=0.8)
        axes[0].plot(t, model(t, a_hat, bU_hat), ls="--", lw=1.6, color=sc, label=f"U = {U}: â = {a_hat:.1f}, b̂ = {bU_hat / U:.0f}")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("angle [rad]")
    axes[0].set_title(f"(a) Step-test identification (true a = {MOTOR_A}, b = {MOTOR_B:.0f}; friction biases b̂)",
                      loc="left", fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=8)

    from exp2_motor import motor_case
    rows = []
    models = [("true linear part", MOTOR_A, MOTOR_B)] + [(f"identified, U = {U}", a, b) for U, a, b in ids]
    rm = np.zeros((len(models), len(PAPER_MOTOR_EXPS)))
    for j, e in enumerate(PAPER_MOTOR_EXPS):
        c = motor_case(e, realistic=True)
        for i, (name, a, b) in enumerate(models):
            A, B, C = motor(a, b)
            y = BVPInterpolator(A, B, C, c["K"], e["tau"]).reconstruct(c["ys"], c["t"])
            rm[i, j] = np.sqrt(np.mean((y - c["y_true"]) ** 2))
    for i, (name, a, b) in enumerate(models):
        rows.append([name, float(a), float(b)] + [float(v) for v in rm[i]])
    ax = axes[1]
    w = 0.16
    cols = [COL["truth"], COL["bvp"], COL["spline"], COL["fast"], COL["model"]]
    for i, (name, _, _) in enumerate(models):
        ax.bar(np.arange(3) + (i - 2) * w, rm[i], width=w * 0.9, color=cols[i], label=name)
    ax.set_xticks(range(3))
    ax.set_xticklabels([e["name"] for e in PAPER_MOTOR_EXPS])
    ax.set_ylabel("BVP reconstruction RMSE [rad]")
    ax.set_title("(b) Reconstruction with each identified model (realistic motor)", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    savefig(fig, "exp15_sysid_pipeline.png")
    header = ["model", "a", "b"] + [f"RMSE {e['name']}" for e in PAPER_MOTOR_EXPS]
    md = write_table(RES / "exp15_sysid.csv", header, rows, fmt="{:.4g}")
    return "## Experiment 15 - identification + reconstruction pipeline\n\n" + md


if __name__ == "__main__":
    print(run())
