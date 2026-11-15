"""Experiment 7 - Coulomb friction (reproduces the artifact of paper Experiment 3).

Paper motor Exps 1-3 on the realistic plant with friction level Fc swept (stiction 1.2 Fc,
encoder quantization on). BVP (linear model) vs linear interpolation.
"""
import numpy as np
from core import COL, LABEL, PAPER_MOTOR_EXPS, RES, nrmse, savefig, style, write_table, plt
from exp2_motor import FC_NOMINAL, motor_case

FCS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
EXP_MARK = ["o", "s", "^"]


def run():
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    rows = []
    for e, mk in zip(PAPER_MOTOR_EXPS, EXP_MARK):
        bv, li = [], []
        for Fc in FCS:
            c = motor_case(e, Fc=Fc, realistic=True)
            bv.append(nrmse(c["y_true"], c["y_bvp"]))
            li.append(nrmse(c["y_true"], c["y_lin"]))
            rows.append([e["name"], Fc, li[-1], bv[-1], li[-1] / bv[-1]])
        ax.plot(FCS, bv, color=COL["bvp"], marker=mk, ms=6, label=f"BVP – {e['name']}")
        ax.plot(FCS, li, color=COL["linear"], ls="--", marker=mk, ms=6, mfc="none", label=f"Linear – {e['name']}")
    ax.set_yscale("log")
    ax.set_xlabel("Coulomb friction level Fc [rad/s²]  (not in the model)")
    ax.set_ylabel("normalized RMSE")
    ax.set_title("Reconstruction error vs unmodelled friction", loc="left")
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    # zoom on the Exp 3 tail at nominal friction
    c = motor_case(PAPER_MOTOR_EXPS[2], Fc=FC_NOMINAL, realistic=True)
    ax = axes[1]
    sel = c["t"] >= 0.9
    ax.plot(c["t"][sel], c["y_true"][sel], color=COL["truth"], lw=1.2, label="True angle (stuck by friction)")
    ax.plot(c["t"][sel], c["y_lin"][sel], color=COL["linear"], ls="--", lw=1.6, label=LABEL["linear"])
    ax.plot(c["t"][sel], c["y_bvp"][sel], color=COL["bvp"], lw=2.0, label=LABEL["bvp"])
    ks = (c["tk"] >= 0.9) & (c["tk"] <= c["t"][-1])
    ax.plot(c["tk"][ks], c["ys"][ks], "o", color=COL["truth"], ms=5, label="Samples")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [rad]")
    ax.set_title(f"Motor Exp 3 tail, Fc = {FC_NOMINAL}: model predicts motion that friction prevents",
                 loc="left", fontsize=10)
    ax.legend(loc="upper right")
    savefig(fig, "exp7_friction.png")

    header = ["exp", "Fc [rad/s^2]", "nRMSE linear", "nRMSE BVP", "linear/BVP"]
    md = write_table(RES / "exp7_friction.csv", header, rows, fmt="{:.3g}")
    return "## Experiment 7 - unmodelled Coulomb friction\n\n" + md


if __name__ == "__main__":
    print(run())
