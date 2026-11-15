"""Experiment 3 - head-to-head comparison of intersample reconstruction methods.

Methods (what they need):
  linear, spline      : samples only
  fast3, fast10       : model + TRUE state x_k and input u_k at every sample (oracle)
  model               : model + TRUE initial state x0, simulated forward (oracle x0)
  bvp                 : model + K + samples   (the paper)
Datasets: paper Examples 1-3 (ideal) and motor Exps 1-3 (realistic plant).
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, PAPER_EXAMPLES, PAPER_MOTOR_EXPS, RES,
                  double_integrator, fast_sampling_curve, linear_interp, metrics, savefig,
                  simulate_linear, spline_interp, style, write_table, plt)
from exp2_motor import motor_case

METHODS = ["linear", "spline", "fast3", "fast10", "model", "bvp"]
MARK = {"linear": "s", "spline": "D", "fast3": "^", "fast10": "v", "model": "P", "bvp": "o"}
MCOL = {"linear": COL["linear"], "spline": COL["spline"], "fast3": COL["fast"], "fast10": COL["fast"],
        "model": COL["model"], "bvp": COL["bvp"]}


def datasets():
    A, B, C = double_integrator()
    out = []
    for ex in PAPER_EXAMPLES:
        N = 10 if "deadbeat" not in ex["name"] else 5
        t = np.linspace(0, N - 1, 3001)
        sim = simulate_linear(A, B, C, ex["K"], 1.0, ex["x0"], N, t)
        out.append(dict(name=ex["name"].split(" (")[0], A=A, B=B, C=C, K=np.atleast_2d(ex["K"]), tau=1.0,
                        x0=ex["x0"], N=N, t=t, tk=np.arange(N), y_true=sim["y"], ys=sim["yk"],
                        xk=sim["xk"], uk=sim["uk"]))
    for e in PAPER_MOTOR_EXPS:
        c = motor_case(e, realistic=True)
        out.append(dict(name="Motor " + e["name"], A=c["A"], B=c["B"], C=c["C"], K=c["K"], tau=e["tau"],
                        x0=e["x0"], N=c["N"], t=c["t"], tk=c["tk"], y_true=c["y_true"], ys=c["ys"],
                        xk=c["xk"], uk=c["uk"]))
    return out


def estimates(d):
    t, tk, ys = d["t"], d["tk"], d["ys"]
    est = {"linear": linear_interp(tk, ys, t), "spline": spline_interp(tk, ys, t)}
    for m in (3, 10):
        est[f"fast{m}"], _ = fast_sampling_curve(d["A"], d["B"], d["C"], d["tau"], d["xk"], d["uk"], m, t)
    est["model"] = simulate_linear(d["A"], d["B"], d["C"], d["K"], d["tau"], d["x0"], d["N"], t)["y"]
    est["bvp"] = BVPInterpolator(d["A"], d["B"], d["C"], d["K"], d["tau"]).reconstruct(ys, t)
    return est


def run():
    style()
    data = datasets()
    rows, rmse_tab = [], {}
    for d in data:
        est = estimates(d)
        rmse_tab[d["name"]] = {}
        for m in METHODS:
            mt = metrics(d["y_true"], est[m])
            rmse_tab[d["name"]][m] = mt["rmse"]
            rows.append([d["name"], LABEL[m], mt["rmse"], mt["maxerr"], mt["peakerr"]])
        d["est"] = est

    # --- dot plot of RMSE (log) per dataset
    # values below 1e-12 are numerically exact; draw them in a separate "exact" lane on the left
    exact_x = 3e-5
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    names = [d["name"] for d in data]
    for j, m in enumerate(METHODS):
        vals = [rmse_tab[n][m] if rmse_tab[n][m] > 1e-12 else exact_x for n in names]
        yy = np.arange(len(names)) + (j - 2.5) * 0.11
        ax.plot(vals, yy, MARK[m], color=MCOL[m], ms=8, mfc=MCOL[m] if m != "fast10" else "none",
                mew=1.6, label=LABEL[m], ls="none")
    ax.axvspan(1.4e-5, 6.5e-5, color=COL["grid"], zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1.4e-5, 4)
    ax.set_xticks([exact_x, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["exact\n(<1e-12)", "1e-3", "1e-2", "0.1", "1"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE vs true continuous output (log scale)")
    ax.set_title("Reconstruction error by method  (Examples: ideal plant;  Motor: friction + quantization)",
                 loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    savefig(fig, "exp3_rmse_dotplot.png")

    # --- overlays: Example 1 and Motor Exp 1
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5))
    for ax, d, tmax in ((axes[0], data[0], 5.0), (axes[1], data[3], 1.3)):
        sel = d["t"] <= tmax
        t = d["t"][sel]
        ax.plot(t, d["y_true"][sel], color=COL["truth"], lw=1.0, label=LABEL["truth"], zorder=6)
        ax.plot(t, d["est"]["linear"][sel], color=COL["linear"], ls="--", lw=1.6, label=LABEL["linear"])
        ax.plot(t, d["est"]["spline"][sel], color=COL["spline"], ls=":", lw=2.0, label=LABEL["spline"])
        ax.plot(t, d["est"]["model"][sel], color=COL["model"], ls="-.", lw=1.6, label=LABEL["model"])
        ax.plot(t, d["est"]["bvp"][sel], color=COL["bvp"], lw=2.4, label=LABEL["bvp"], zorder=5)
        ks = d["tk"] <= tmax
        ax.plot(d["tk"][ks], d["ys"][ks], "o", color=COL["truth"], ms=5.5, label="Samples", zorder=7)
        ax.set_title(d["name"], loc="left")
        ax.set_ylabel("output")
        ax.set_xlim(0, tmax)
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_xlabel("time (τ = 1)")
    axes[1].set_xlabel("time [s]")
    savefig(fig, "exp3_method_overlay.png")

    header = ["dataset", "method", "RMSE", "max abs error", "error in min value"]
    md = write_table(RES / "exp3_comparison.csv", header, rows)
    # compact RMSE matrix
    short = ["linear", "spline", "fast3", "fast10", "model", "bvp"]
    mat = "| dataset | " + " | ".join(short) + " |\n|" + "---|" * (len(short) + 1) + "\n"
    for n in names:
        mat += f"| {n} | " + " | ".join(f"{rmse_tab[n][m]:.3g}" for m in short) + " |\n"
    return ("## Experiment 3 - method comparison\n\nRMSE matrix (fast*/model use oracle true state / x0):\n\n"
            + mat + "\nFull metrics:\n\n" + md)


if __name__ == "__main__":
    print(run())
