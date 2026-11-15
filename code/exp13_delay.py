"""Experiment 13 (extension) - computation delay.

The controller applies u_k at k tau + D instead of k tau (u_{k-1} held until then).
Three reconstructions:
  paper       : assumes D = 0                                      (2 samples)
  augmented   : z_k = [x_k; u_{k-1}], 3 samples, no logging needed (n + 1 = 3)
  input-aware : logged u_k, Q = [C; C Ad], 2 samples
"""
import numpy as np
from core import BVPInterpolator, COL, LABEL, RES, motor, nrmse, savefig, style, write_table, plt
from core_ext import delay_model, known_input_bvp, motor_stable_K, simulate_sd

TAU = 0.089118
N = 30
SIGMA = 0.003


def run(seed=6):
    style()
    rng = np.random.default_rng(seed)
    A, B, C = motor()
    K = motor_stable_K(TAU)
    fracs = np.linspace(0, 0.6, 13)
    t = np.linspace(0, (N - 3) * TAU, 2000)
    out = {m: [] for m in ("paper", "aug", "ki")}
    outn = {m: [] for m in ("paper", "aug", "ki")}
    rows = []
    for f in fracs:
        D = f * TAU
        sim = simulate_sd(A, B, C, TAU, (0.6, 0.0), N, lambda k, y, x: -(K @ x)[0], t, delay=D)
        Rp = BVPInterpolator(A, B, C, K, TAU).operator(t, N)
        aug = delay_model(A, B, C, K, TAU, D)
        Ra = aug.operator(t, N)
        yki, _ = known_input_bvp(A, B, C, TAU, sim["yk"], sim["uk"], sim["uprev"], t, delay=D)
        out["paper"].append(nrmse(sim["y"], Rp @ sim["yk"]))
        out["aug"].append(max(nrmse(sim["y"], Ra @ sim["yk"]), 1e-16))
        out["ki"].append(max(nrmse(sim["y"], yki), 1e-16))
        acc = {m: 0.0 for m in outn}
        for _ in range(40):
            e = SIGMA * rng.standard_normal(N)
            acc["paper"] += nrmse(sim["y"], Rp @ (sim["yk"] + e)) ** 2
            acc["aug"] += nrmse(sim["y"], Ra @ (sim["yk"] + e)) ** 2
            yk_, _ = known_input_bvp(A, B, C, TAU, sim["yk"] + e, sim["uk"], sim["uprev"], t, delay=D)
            acc["ki"] += nrmse(sim["y"], yk_) ** 2
        for m in outn:
            outn[m].append(np.sqrt(acc[m] / 40))
        rows.append([float(f), out["paper"][-1], out["aug"][-1], out["ki"][-1], outn["paper"][-1], outn["aug"][-1],
                     outn["ki"][-1], aug.cond])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = {"paper": (COL["linear"], "--", "s", "Paper formula (assumes no delay)"),
              "aug": (COL["model"], "-.", "P", "Augmented BVP, 3 samples (no logging)"),
              "ki": (COL["bvp"], "-", "o", "Input-aware BVP, 2 samples + logged u")}
    for ax, dat, ttl in ((axes[0], out, "(a) Noise-free samples"), (axes[1], outn, f"(b) Noisy samples σ = {SIGMA} rad")):
        for m, (c, ls, mk, lab) in styles.items():
            vals = [v if v > 1e-12 else 2e-5 for v in dat[m]]
            ax.semilogy(100 * fracs, vals, color=c, ls=ls, marker=mk, ms=4.5, label=lab)
        ax.set_xlabel("computation delay D [% of τ]")
        ax.set_ylabel("normalized RMSE")
        ax.set_title(ttl, loc="left")
        ax.set_ylim(1e-5, 1)
        ax.axhspan(1e-5, 4e-5, color=COL["grid"], zorder=0)
        ax.set_yticks([2e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        ax.set_yticklabels(["exact", "1e-4", "1e-3", "0.01", "0.1", "1"])
    axes[0].legend(loc="center right", fontsize=8)

    D = 0.4 * TAU
    sim = simulate_sd(A, B, C, TAU, (0.6, 0.0), N, lambda k, y, x: -(K @ x)[0], t, delay=D)
    ax = axes[2]
    sel = t <= 1.2
    ax.plot(t[sel], sim["y"][sel], color=COL["truth"], lw=1.0, label=LABEL["truth"])
    ax.plot(t[sel], (BVPInterpolator(A, B, C, K, TAU).reconstruct(sim["yk"], t))[sel], color=COL["linear"], ls="--",
            label="Paper formula (no delay)")
    yki, _ = known_input_bvp(A, B, C, TAU, sim["yk"], sim["uk"], sim["uprev"], t, delay=D)
    ax.plot(t[sel], yki[sel], color=COL["bvp"], lw=2, alpha=0.8, label="Input-aware BVP")
    tk = np.arange(N) * TAU
    ax.plot(tk[tk <= 1.2], sim["yk"][tk <= 1.2], "o", color=COL["truth"], ms=4.5, label="Samples")
    ax.set_title("(c) Delay D = 0.4 τ", loc="left")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [rad]")
    ax.legend(loc="upper left", fontsize=8)
    savefig(fig, "exp13_computation_delay.png")
    header = ["D/tau", "nRMSE paper", "nRMSE augmented", "nRMSE input-aware", "noisy paper", "noisy augmented",
              "noisy input-aware", "cond(O_aug)"]
    md = write_table(RES / "exp13_delay.csv", header, rows[::3], fmt="{:.3g}")
    return "## Experiment 13 - computation delay (extension)\n\n" + md


if __name__ == "__main__":
    print(run())
