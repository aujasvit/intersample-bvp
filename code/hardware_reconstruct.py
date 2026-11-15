"""Reconstruct the intersample response from REAL logged data (e.g. the EE380 DC-motor rig).

Input CSV (header required):  t,y        - controller sampling instants and measured output
                         or:  t,y,u      - plus the control value the controller applied
Optional --truth CSV:         t,y        - high-rate log (e.g. 1 kHz) to score the reconstruction

Model: DC motor  theta'' = -a theta' + b u   (paper eq. 5.1 by default), y = theta,
       state feedback u_k = -(k1 theta_k + k2 omega_k).
If u is logged, the input-aware BVP is used (works for ANY controller, delay via --delay);
otherwise the paper's closed-loop formula with K = [k1, k2].

Examples
  python hardware_reconstruct.py --demo                       # validate on simulated logs
  python hardware_reconstruct.py --log run1.csv --k1 1.3578 --truth run1_1khz.csv
"""
import argparse
from pathlib import Path

import numpy as np
from core import (BVPInterpolator, COL, FIGS, LABEL, MOTOR_A, MOTOR_B, PAPER_MOTOR_EXPS, RES, linear_interp,
                  motor, savefig, simulate_motor_real, style, plt)
from core_ext import known_input_bvp


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {k: np.asarray(data[k], float) for k in data.dtype.names}


def reconstruct(log, a, b, k1, k2, delay=0.0, n_points=3000):
    A, B, C = motor(a, b)
    tk = log["t"] - log["t"][0]
    tau = float(np.median(np.diff(tk)))
    if np.max(np.abs(np.diff(tk) - tau)) > 0.02 * tau:
        print(f"warning: sampling jitter up to {np.max(np.abs(np.diff(tk) - tau)) / tau:.1%} of tau")
    t = np.linspace(0, tk[-1], n_points)
    if "u" in log:
        u = log["u"]
        uprev = np.concatenate([[0.0], u[:-1]])
        y, cond = known_input_bvp(A, B, C, tau, log["y"], u, uprev, t, delay=delay)
        method = "input-aware BVP (logged u)"
    else:
        I = BVPInterpolator(A, B, C, [[k1, k2]], tau)
        y, cond = I.reconstruct(log["y"], t), I.cond
        method = "paper BVP (closed-loop model)"
    return t + log["t"][0], y, tau, cond, method


def report(log, t, y, tau, cond, method, truth=None, name="hardware"):
    style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    msg = [f"{name}: tau = {tau:.6g} s, cond(Q) = {cond:.3g}, method = {method}"]
    if truth is not None:
        ax.plot(truth["t"], truth["y"], color=COL["truth"], lw=1.0, label="High-rate log (truth)")
        sel = (truth["t"] >= log["t"][0]) & (truth["t"] <= log["t"][-1])
        yb = np.interp(truth["t"][sel], t, y)
        yl = linear_interp(log["t"], log["y"], truth["t"][sel])
        rb = np.sqrt(np.mean((yb - truth["y"][sel]) ** 2))
        rl = np.sqrt(np.mean((yl - truth["y"][sel]) ** 2))
        msg.append(f"RMSE linear = {rl:.4g}, RMSE BVP = {rb:.4g}  (ratio {rl / rb:.2f}x)")
    ax.plot(log["t"], log["y"], color=COL["linear"], ls="--", lw=1.5, label=LABEL["linear"])
    ax.plot(t, y, color=COL["bvp"], lw=2.0, label=LABEL["bvp"])
    ax.plot(log["t"], log["y"], "o", color=COL["truth"], ms=4.5, label="Logged samples")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output")
    ax.set_title(msg[0], loc="left", fontsize=9.5)
    ax.legend(loc="upper right")
    out = savefig(fig, f"hw_{name}.png")
    print("\n".join(msg), "\nfigure ->", out)
    return msg


def demo():
    """Write simulated logs in the expected CSV format, then run the pipeline on them."""
    outdir = RES / "hardware_demo"
    outdir.mkdir(exist_ok=True)
    lines = []
    for e in PAPER_MOTOR_EXPS:
        N = int(e["T"] / e["tau"]) + 1
        r = simulate_motor_real(MOTOR_A, MOTOR_B, e["tau"], e["k1"], e["x0"], N)
        tk = np.arange(N) * e["tau"]
        tag = e["name"].replace(" ", "")
        np.savetxt(outdir / f"{tag}_samples.csv", np.c_[tk, r["ymeas"]], delimiter=",", header="t,y", comments="")
        np.savetxt(outdir / f"{tag}_samples_u.csv", np.c_[tk, r["ymeas"], r["uk"]], delimiter=",", header="t,y,u",
                   comments="")
        t1k = np.arange(0, tk[-1], 1e-3)
        np.savetxt(outdir / f"{tag}_truth_1khz.csv", np.c_[t1k, np.interp(t1k, r["t_fine"], r["y_fine"])],
                   delimiter=",", header="t,y", comments="")
        for fname, sfx in ((f"{tag}_samples.csv", ""), (f"{tag}_samples_u.csv", "_u")):
            log = load_csv(outdir / fname)
            res = reconstruct(log, MOTOR_A, MOTOR_B, e["k1"], 0.0)
            lines += report(log, *res, truth=load_csv(outdir / f"{tag}_truth_1khz.csv"), name=f"demo_{tag}{sfx}")
    return lines


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path)
    p.add_argument("--truth", type=Path)
    p.add_argument("--a", type=float, default=MOTOR_A)
    p.add_argument("--b", type=float, default=MOTOR_B)
    p.add_argument("--k1", type=float, default=0.0)
    p.add_argument("--k2", type=float, default=0.0)
    p.add_argument("--delay", type=float, default=0.0, help="computation delay [s] (input-aware mode only)")
    p.add_argument("--name", default="hardware")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        demo()
    elif args.log:
        log = load_csv(args.log)
        res = reconstruct(log, args.a, args.b, args.k1, args.k2, delay=args.delay)
        report(log, *res, truth=load_csv(args.truth) if args.truth else None, name=args.name)
    else:
        p.print_help()
