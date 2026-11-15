"""Core tools for reproducing and stress-testing Haba & Budai (2025),
"Determining the intersample response of state feedback control systems using
boundary value problems".

Notation follows the paper / writeup:
    plant            x' = A x + B u,  y = C x
    ZOH              u(t) = u_k on [k tau, (k+1) tau)
    state feedback   u_k = -K x_k
    within interval  x(k tau + s) = Phi(s) x_k,  Phi(s) = Ad(s) - Bd(s) K
    BVP              Q x_k = [y_k; y_{k+1}],  Q = [C; C Phi(tau)]
    interpolation    y(k tau + s) = C Phi(s) Q^{-1} [y_k; y_{k+1}]
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.signal import place_poles

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
RES = ROOT / "results"
FIGS.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- plotting style
# Fixed method -> colour mapping used in every figure (validated categorical palette).
COL = {
    "bvp": "#2a78d6",      # blue   - the paper's method
    "linear": "#eb6834",   # orange - linear interpolation
    "spline": "#1baf7a",   # aqua   - cubic spline
    "fast": "#eda100",     # yellow - fast sampling / lifting points
    "model": "#6250d6",    # violet - forward model simulation from x0
    "truth": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
}
LABEL = {
    "bvp": "BVP method (paper)",
    "bvp_ls": "BVP least-squares (4 samples)",
    "linear": "Linear interpolation",
    "spline": "Cubic spline",
    "fast3": "Fast sampling m=3 (needs true state)",
    "fast10": "Fast sampling m=10 (needs true state)",
    "model": "Model simulation from x0",
    "truth": "True continuous output",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "axes.edgecolor": COL["ink2"],
        "axes.labelcolor": COL["truth"], "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
        "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5,
        "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110,
    })


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=200)
    plt.close(fig)
    return FIGS / name


# ---------------------------------------------------------------- plants
def double_integrator():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


MOTOR_A, MOTOR_B = 12.1825, 282.9259   # paper eq. (5.1)


def motor(a=MOTOR_A, b=MOTOR_B):
    A = np.array([[0.0, 1.0], [0.0, -a]])
    B = np.array([[0.0], [b]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


# Paper Table 1 (+ initial angle / duration read off Figs. 7-9; approximate).
PAPER_MOTOR_EXPS = [
    dict(name="Exp 1", tau=0.089118, k1=1.3578, x0=(0.6, 0.0), T=3.0, paper_lin=0.064851, paper_bvp=0.017650),
    dict(name="Exp 2", tau=0.056322, k1=2.3937, x0=(0.5, 0.0), T=1.8, paper_lin=0.269140, paper_bvp=0.085217),
    dict(name="Exp 3", tau=0.106300, k1=1.1688, x0=(0.6, 0.0), T=2.5, paper_lin=0.083632, paper_bvp=0.038559),
]

# Paper Section 4 examples (double integrator, tau = 1).
PAPER_EXAMPLES = [
    dict(name="Example 1 (ripple)", K=(2.0, 1.5), x0=(1.0, 0.0),
         Gamma=lambda t: np.array([[1 - t**2, 4*t - 3*t**2], [-2*t, 4 - 6*t]])),
    dict(name="Example 2 (hidden oscillation)", K=(0.95, 1.75), x0=(1.0, 0.0),
         Gamma=lambda t: np.array([[1 - 4.2*t + 3.2*t**2, 8*t - 7*t**2], [-4.2 + 6.4*t, 8 - 14*t]])),
    dict(name="Example 3 (deadbeat)", K=(1.0, 1.5), x0=(1.0, -4.0),
         Gamma=lambda t: np.array([[(1 - t)**2, 4*t - 3*t**2], [2*t - 2, 4 - 6*t]])),
]


# ---------------------------------------------------------------- ZOH / transition matrices
def zoh(A, B, s):
    """Ad(s) = e^{As}, Bd(s) = int_0^s e^{A sigma} B dsigma (Van Loan augmented exponential)."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * s)
    return E[:n, :n], E[:n, n:]


def Phi(A, B, K, s):
    """Closed-loop transition inside one ZOH interval: x(k tau + s) = Phi(s) x_k."""
    Ad, Bd = zoh(A, B, s)
    return Ad - Bd @ np.atleast_2d(K)


def place_gain(A, B, tau, poles):
    """Discrete pole placement for the ZOH-discretized plant."""
    Ad, Bd = zoh(A, B, tau)
    return place_poles(Ad, Bd, poles).gain_matrix


# ---------------------------------------------------------------- the paper's method
class BVPInterpolator:
    """Intersample reconstruction from output samples.

    m = n reproduces the paper exactly (n = 2: Q = [C; C Phi(tau)]).
    m > n is the least-squares extension (uses m consecutive samples).
    """

    def __init__(self, A, B, C, K, tau, m=None):
        self.A, self.B, self.C = A, B, C
        self.K = np.atleast_2d(K)
        self.tau = tau
        self.n = A.shape[0]
        self.m = self.n if m is None else m
        self.Acl = Phi(A, B, self.K, tau)
        self.O = np.vstack([C @ np.linalg.matrix_power(self.Acl, j) for j in range(self.m)])
        self.Oinv = np.linalg.pinv(self.O)
        self.cond = np.linalg.cond(self.O)

    def Gamma(self, s):
        """Paper's BVP principal solution matrix Gamma(s) = Phi(s) Q^{-1} (m = n only)."""
        return Phi(self.A, self.B, self.K, s) @ np.linalg.inv(self.O)

    def operator(self, t, N):
        """Linear map R such that y_hat(t) = R @ y_samples (N samples)."""
        kmax = N - self.m
        R = np.zeros((len(t), N))
        cache = {}
        for i, ti in enumerate(t):
            k = min(int(np.floor(ti / self.tau + 1e-9)), kmax)
            s = ti - k * self.tau
            key = round(s, 12)
            if key not in cache:
                cache[key] = self.C @ Phi(self.A, self.B, self.K, s) @ self.Oinv
            R[i, k:k + self.m] = cache[key][0]
        return R

    def valid_until(self, N):
        return (N - self.m + 1) * self.tau

    def reconstruct(self, y_samples, t):
        return self.operator(t, len(y_samples)) @ np.asarray(y_samples)


# ---------------------------------------------------------------- baselines
def linear_interp(tk, yk, t):
    return np.interp(t, tk, yk)


def spline_interp(tk, yk, t):
    return CubicSpline(tk, yk)(t)


def fast_sampling(A, B, C, tau, xk, uk, m):
    """Fast discretization / lifting: m points per interval from the TRUE state x_k and input u_k."""
    pts_t, pts_y = [], []
    mats = [zoh(A, B, j * tau / m) for j in range(m)]
    for k in range(len(uk)):
        for j, (Ad, Bd) in enumerate(mats):
            pts_t.append(k * tau + j * tau / m)
            pts_y.append((C @ (Ad @ xk[k] + Bd[:, 0] * uk[k]))[0])
    pts_t.append(len(uk) * tau)
    pts_y.append((C @ xk[len(uk)])[0])
    return np.array(pts_t), np.array(pts_y)


def fast_sampling_curve(A, B, C, tau, xk, uk, m, t):
    pt, py = fast_sampling(A, B, C, tau, xk, uk, m)
    return np.interp(t, pt, py), (pt, py)


# ---------------------------------------------------------------- simulators
def simulate_linear(A, B, C, K, tau, x0, N, t):
    """Exact sampled-data simulation (linear plant, ZOH, u_k = -K x_k).
    Returns samples y_k (k=0..N-1), states x_k (k=0..N), inputs u_k, dense output y(t)."""
    K = np.atleast_2d(K)
    Acl = Phi(A, B, K, tau)
    xs = [np.asarray(x0, float)]
    for _ in range(N):
        xs.append(Acl @ xs[-1])
    xs = np.array(xs)
    us = np.array([-(K @ xs[k])[0] for k in range(N)])
    yk = xs[:N] @ C[0]
    y = np.empty(len(t))
    cache = {}
    for i, ti in enumerate(t):
        k = min(int(np.floor(ti / tau + 1e-9)), N - 1)
        s = ti - k * tau
        key = round(s, 12)
        if key not in cache:
            cache[key] = C @ Phi(A, B, K, s)
        y[i] = (cache[key] @ xs[k])[0]
    return dict(yk=yk, xk=xs, uk=us, y=y)


ENCODER_STEP = 2 * np.pi / 2048   # assumed 512-line encoder, quadrature decoding


def simulate_motor_real(a, b, tau, k1, x0, N, Fc=15.0, Fs=None, quant=ENCODER_STEP, dt_target=2e-5):
    """'Realistic' motor: linear model + Coulomb/static friction (in rad/s^2) + encoder quantization.

    theta'' = -a theta' + b u - Fc sign(theta')   (sliding)
    stuck while |b u| <= Fs.   Controller: u_k = -k1 * quantize(theta(k tau)).
    Each ~20 us sub-step is integrated EXACTLY for the linear part (friction force held
    constant over the sub-step); a velocity sign change inside a sub-step is treated as
    sticking. With Fc = 0 and quant = 0 this reproduces the exact linear solution.
    """
    Fs = 1.2 * Fc if Fs is None else Fs
    nsub = max(1, int(round(tau / dt_target)))
    dt = tau / nsub
    ea = np.exp(-a * dt)
    g1 = (1 - ea) / a              # int_0^dt e^{-a s} ds
    g2 = (dt - g1) / a             # int_0^dt (1 - e^{-a s})/a ds
    th, om = float(x0[0]), float(x0[1])
    th_f = np.empty(N * nsub + 1)
    th_f[0] = th
    yk, ymeas, uk, xk = [], [], [], []
    idx = 1
    for _ in range(N):
        q = quant * np.round(th / quant) if quant else th
        u = -k1 * q
        yk.append(th)
        ymeas.append(q)
        uk.append(u)
        xk.append((th, om))
        F = b * u
        for _ in range(nsub):
            if om == 0.0:
                if abs(F) <= Fs:
                    th_f[idx] = th
                    idx += 1
                    continue
                f = F - Fc * np.sign(F)
            else:
                f = F - Fc * np.sign(om)
            # exact solution of om' = -a om + f over dt
            om_new = ea * om + g1 * f
            th_new = th + g1 * om + g2 * f
            if om != 0.0 and om_new * om < 0.0 and Fc > 0:
                om_new = 0.0          # reversal inside sub-step -> stick (dt is tiny)
            th, om = th_new, om_new
            th_f[idx] = th
            idx += 1
    xk.append((th, om))
    t_f = np.arange(N * nsub + 1) * dt
    return dict(t_fine=t_f, y_fine=th_f, yk=np.array(yk), ymeas=np.array(ymeas),
                uk=np.array(uk), xk=np.array(xk), tau=tau)


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_hat):
    e = y_hat - y_true
    return dict(rmse=float(np.sqrt(np.mean(e**2))), maxerr=float(np.max(np.abs(e))),
                peakerr=float(abs(np.min(y_hat) - np.min(y_true))))


def nrmse(y_true, y_hat):
    return float(np.sqrt(np.mean((y_hat - y_true)**2)) / np.sqrt(np.mean(y_true**2)))


def write_table(path, header, rows, fmt="{:.4g}"):
    """Write a CSV and return a markdown table string."""
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + "\n")
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        md += "| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |\n"
    return md
