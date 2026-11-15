"""Experiment 11 - Kalman (RTS) smoother as the principled baseline.

The smoother uses ALL samples with a noise model (x_{k+1} = Acl x_k + w, y_k = C x_k + v),
then fills each interval with C Phi(s) xhat_k. The BVP method uses 2 samples and trusts them fully.
(a) Example 1, exact model: RMSE vs measurement noise.
(b,c) Stable motor design, noise sigma = 3 mrad: RMSE vs smoother process-noise tuning q,
      with an exact model and with a model whose gain b is 20 % wrong.
"""
import numpy as np
from core import (BVPInterpolator, COL, LABEL, PAPER_EXAMPLES, RES, double_integrator, motor, nrmse,
                  savefig, simulate_linear, spline_interp, style, write_table, plt, Phi, MOTOR_A, MOTOR_B)
from core_ext import motor_stable_K, rts_smoother

MC = 150


def smoother_curve(A, B, C, K, tau, ys, t, q, r):
    Acl = Phi(A, B, K, tau)
    xs = rts_smoother(Acl, C, ys, q, r)
    out = np.empty(len(t))
    cache = {}
    for i, ti in enumerate(t):
        k = min(int(np.floor(ti / tau + 1e-9)), len(ys) - 1)
        s = ti - k * tau
        key = round(s, 12)
        if key not in cache:
            cache[key] = C[0] @ Phi(A, B, K, s)
        out[i] = cache[key] @ xs[k]
    return out


def run(seed=4):
    style()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = []

    # (a) noise sweep on Example 1
    A, B, C = double_integrator()
    ex = PAPER_EXAMPLES[0]
    N = 20
    b2, b4 = BVPInterpolator(A, B, C, ex["K"], 1.0), BVPInterpolator(A, B, C, ex["K"], 1.0, m=4)
    t = np.linspace(0, b4.valid_until(N), 1500)
    tk = np.arange(N)
    sim = simulate_linear(A, B, C, ex["K"], 1.0, ex["x0"], N, t)
    R2, R4 = b2.operator(t, N), b4.operator(t, N)
    sig = np.logspace(-4, -0.5, 12)
    cur = {m: [] for m in ("spline", "bvp", "bvp_ls", "rts")}
    for s_ in sig:
        acc = {m: 0.0 for m in cur}
        for _ in range(MC):
            yn = sim["yk"] + s_ * rng.standard_normal(N)
            acc["spline"] += np.mean((spline_interp(tk, yn, t) - sim["y"]) ** 2)
            acc["bvp"] += np.mean((R2 @ yn - sim["y"]) ** 2)
            acc["bvp_ls"] += np.mean((R4 @ yn - sim["y"]) ** 2)
            acc["rts"] += np.mean((smoother_curve(A, B, C, ex["K"], 1.0, yn, t, 1e-10, s_ ** 2) - sim["y"]) ** 2)
        for m in cur:
            cur[m].append(np.sqrt(acc[m] / MC))
    ax = axes[0]
    ax.loglog(sig, cur["spline"], color=COL["spline"], ls=":", marker="D", ms=4, label=LABEL["spline"])
    ax.loglog(sig, cur["bvp"], color=COL["bvp"], marker="o", ms=4, label=LABEL["bvp"])
    ax.loglog(sig, cur["bvp_ls"], color=COL["model"], ls="-.", marker="P", ms=4, label=LABEL["bvp_ls"])
    ax.loglog(sig, cur["rts"], color=COL["truth"], ls="--", marker="s", ms=4, label="Kalman/RTS smoother (all samples)")
    ax.set_xlabel("noise std σ on samples")
    ax.set_ylabel("RMSE vs true output")
    ax.set_title("(a) Example 1, exact model", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    rows.append(["Ex.1 exact model, sigma=1e-2", "", float(np.interp(1e-2, sig, cur["spline"])),
                 float(np.interp(1e-2, sig, cur["bvp"])), float(np.interp(1e-2, sig, cur["rts"]))])

    # (b,c) motor: tuning sweep, exact vs mismatched model
    A, B, C = motor()
    tau = 0.089118
    K = motor_stable_K(tau)
    N = 34
    tm = np.linspace(0, (N - 1) * tau, 2000)
    tkm = np.arange(N) * tau
    sigma = 0.003
    qs = np.logspace(-8, 4, 13)
    QSHAPE = np.diag([1e-4, 1.0])   # model error enters mainly through acceleration -> velocity
    for ax, berr, ttl in ((axes[1], 0.0, "(b) Motor, exact model"),
                          (axes[2], 0.2, "(c) Motor, model gain b 20 % wrong")):
        At, Bt, _ = motor(MOTOR_A, MOTOR_B * (1 + berr))
        sim = simulate_linear(At, Bt, C, K, tau, (0.6, 0.0), N, tm)
        Rb = BVPInterpolator(A, B, C, K, tau).operator(tm, N)
        noises = [sigma * rng.standard_normal(N) for _ in range(60)]
        bvp = np.sqrt(np.mean([nrmse(sim["y"], Rb @ (sim["yk"] + e)) ** 2 for e in noises]))
        spl = np.sqrt(np.mean([nrmse(sim["y"], spline_interp(tkm, sim["yk"] + e, tm)) ** 2 for e in noises]))
        rts = [np.sqrt(np.mean([nrmse(sim["y"], smoother_curve(A, B, C, K, tau, sim["yk"] + e, tm, q * QSHAPE, sigma ** 2)) ** 2
                                for e in noises[:30]])) for q in qs]
        ax.semilogx(qs, rts, color=COL["truth"], ls="--", marker="s", ms=4, label="Kalman/RTS smoother")
        ax.axhline(bvp, color=COL["bvp"], lw=2, label=LABEL["bvp"])
        ax.axhline(spl, color=COL["spline"], ls=":", lw=2, label=LABEL["spline"])
        ax.set_xlabel("smoother process noise q,  Q_w = q·diag(1e-4, 1)")
        ax.set_ylabel("normalized RMSE")
        ax.set_title(ttl + f",  σ = {sigma} rad", loc="left")
        ax.set_ylim(0, max(max(rts), bvp, spl) * 1.15)
        rows.append([ttl, f"best q = {qs[int(np.argmin(rts))]:.0e}", spl, bvp, float(min(rts))])
    axes[1].legend(loc="center right", fontsize=8)
    savefig(fig, "exp11_kalman_smoother.png")
    md = write_table(RES / "exp11_kalman.csv", ["case", "note", "spline", "BVP", "RTS (best)"], rows, fmt="{:.3g}")
    return "## Experiment 11 - Kalman/RTS smoother baseline\n\n" + md


if __name__ == "__main__":
    print(run())
