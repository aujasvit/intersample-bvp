"""Experiment 14 (extension) - observer-based output feedback.

Only the angle is measured; a prediction observer estimates the velocity and u_k = -K xhat_k.
The paper's formula assumes u_k = -K x_k (true state) and therefore breaks while the observer
converges. Fixes: augmented BVP on z = [x; xhat] (4 samples), or input-aware BVP (2 samples +
logged u_k), whose Q = [C; C Ad] no longer depends on the controller at all.
"""
import numpy as np
from scipy.signal import place_poles
from core import BVPInterpolator, COL, LABEL, RES, motor, nrmse, savefig, style, write_table, zoh, plt
from core_ext import known_input_bvp, motor_stable_K, observer_model, simulate_sd

TAU = 0.089118
N = 30
SIGMA = 0.003


def run(seed=7):
    style()
    rng = np.random.default_rng(seed)
    A, B, C = motor()
    K = motor_stable_K(TAU)
    Ad, Bd = zoh(A, B, TAU)
    rows = []
    t = np.linspace(0, (N - 4) * TAU, 2000)
    tk = np.arange(N) * TAU
    results = {}
    for obs_poles in ((0.3, 0.35), (0.6, 0.65), (0.8, 0.85)):
        L = place_poles(Ad.T, C.T, list(obs_poles)).gain_matrix.T
        st = {"xh": np.zeros(2)}

        def ctrl(k, y, x, st=st, L=L):
            xh = st["xh"]
            u = -(K @ xh)[0]
            st["xh"] = Ad @ xh + Bd[:, 0] * u + L[:, 0] * (y - C[0] @ xh)
            return u

        sim = simulate_sd(A, B, C, TAU, (0.6, 0.0), N, ctrl, t)
        Rp = BVPInterpolator(A, B, C, K, TAU).operator(t, N)
        om = observer_model(A, B, C, K, L, TAU)
        Ra = om.operator(t, N)
        yki, condki = known_input_bvp(A, B, C, TAU, sim["yk"], sim["uk"], sim["uprev"], t)
        noisy = {"paper": 0.0, "aug": 0.0, "ki": 0.0}
        for _ in range(60):
            e = SIGMA * rng.standard_normal(N)
            noisy["paper"] += nrmse(sim["y"], Rp @ (sim["yk"] + e)) ** 2
            noisy["aug"] += nrmse(sim["y"], Ra @ (sim["yk"] + e)) ** 2
            noisy["ki"] += nrmse(sim["y"], known_input_bvp(A, B, C, TAU, sim["yk"] + e, sim["uk"], sim["uprev"], t)[0]) ** 2
        rows.append([str(obs_poles), nrmse(sim["y"], Rp @ sim["yk"]), max(nrmse(sim["y"], Ra @ sim["yk"]), 0.0),
                     max(nrmse(sim["y"], yki), 0.0), float(np.sqrt(noisy["paper"] / 60)),
                     float(np.sqrt(noisy["aug"] / 60)), float(np.sqrt(noisy["ki"] / 60)), om.cond, condki])
        results[obs_poles] = (sim, Rp @ sim["yk"], Ra @ sim["yk"], yki)

    sim, yp, ya, yki = results[(0.6, 0.65)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    sel = t <= 1.5
    ax.plot(t[sel], sim["y"][sel], color=COL["truth"], lw=1.0, label=LABEL["truth"])
    ax.plot(t[sel], yp[sel], color=COL["linear"], ls="--", lw=1.6, label="Paper formula (assumes u = -K x)")
    ax.plot(t[sel], yki[sel], color=COL["bvp"], lw=2.0, alpha=0.85, label="Input-aware BVP (logged u)")
    ax.plot(tk[tk <= 1.5], sim["yk"][tk <= 1.5], "o", color=COL["truth"], ms=4.5, label="Samples")
    ax.set_title("(a) Observer-based control (observer poles 0.6, 0.65)", loc="left")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [rad]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    for j, (idx, lab, col) in enumerate(((4, "Paper formula", COL["linear"]),
                                         (5, "Augmented BVP, 4 samples", COL["model"]),
                                         (6, "Input-aware BVP, 2 samples", COL["bvp"]))):
        vals = [r[idx] for r in rows]
        ax.bar(x + (j - 1) * w, vals, width=w * 0.9, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"observer poles {l}" for l in labels], fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylabel("normalized RMSE (noisy samples, σ = 3 mrad)")
    ax.set_title("(b) Error vs observer speed", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    savefig(fig, "exp14_observer_control.png")
    header = ["observer poles", "nRMSE paper", "nRMSE augmented", "nRMSE input-aware", "noisy paper",
              "noisy augmented", "noisy input-aware", "cond(O_aug)", "cond(Q_input-aware)"]
    md = write_table(RES / "exp14_observer.csv", header, rows, fmt="{:.3g}")
    return "## Experiment 14 - observer-based control (extension)\n\n" + md


if __name__ == "__main__":
    print(run())
