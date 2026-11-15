"""Extensions used by experiments 9-15.

* simulate_sd        : exact sampled-data simulation with ANY controller and an optional
                       computation delay (u_k applied at k tau + delay).
* IntervalBVP        : generalized BVP reconstruction for an augmented interval model
                       z_{k+1} = F z_k,  y_k = Cz z_k,  y(k tau + s) = Cy(s) z_k
                       (covers delay and observer-based control; m consecutive samples).
* known_input_bvp    : 'input-aware' BVP. If the controller's outputs u_k are logged, the
                       intersample state only depends on x_k, so
                       Q = [C; C Ad] (open-loop!) and y(t) = C(Ad(s) x_k + forced_k(s)).
* rts_smoother       : Kalman / Rauch-Tung-Striebel smoother on the sample grid.
"""
import numpy as np
from core import MOTOR_A, MOTOR_B, motor, place_gain, zoh


class ZohCache:
    def __init__(self, A, B):
        self.A, self.B, self.c = A, B, {}

    def __call__(self, s):
        key = round(float(s), 12)
        if key not in self.c:
            self.c[key] = zoh(self.A, self.B, s)
        return self.c[key]


def motor_stable_K(tau=0.089118):
    """Stable full-state design for the paper's motor (poles 0.85 e^{+-j75deg})."""
    A, B, _ = motor()
    th = np.deg2rad(75)
    return place_gain(A, B, tau, [0.85 * np.exp(1j * th), 0.85 * np.exp(-1j * th)])


# ------------------------------------------------------------------ simulation
def simulate_sd(A, B, C, tau, x0, N, controller, t, delay=0.0, u_init=0.0):
    """controller(k, y_k, x_k) -> u_k (may keep internal state). u_k acts from k tau + delay."""
    Z = ZohCache(A, B)
    x = np.asarray(x0, float)
    xs, xm, us, up = [x], [], [], []
    u_prev = u_init
    for k in range(N):
        u = float(controller(k, float(C[0] @ x), x))
        if delay > 0:
            Ad1, Bd1 = Z(delay)
            xmid = Ad1 @ x + Bd1[:, 0] * u_prev
            Ad2, Bd2 = Z(tau - delay)
            xn = Ad2 @ xmid + Bd2[:, 0] * u
        else:
            Ad1, Bd1 = Z(tau)
            xmid = x
            xn = Ad1 @ x + Bd1[:, 0] * u
        us.append(u)
        up.append(u_prev)
        xm.append(xmid)
        xs.append(xn)
        u_prev, x = u, xn
    xs = np.array(xs)
    y = np.empty(len(t))
    for i, ti in enumerate(t):
        k = min(int(np.floor(ti / tau + 1e-9)), N - 1)
        s = ti - k * tau
        if s < delay:
            Ad, Bd = Z(s)
            xx = Ad @ xs[k] + Bd[:, 0] * up[k]
        else:
            Ad, Bd = Z(s - delay)
            xx = Ad @ xm[k] + Bd[:, 0] * us[k]
        y[i] = C[0] @ xx
    return dict(yk=xs[:N] @ C[0], xk=xs, uk=np.array(us), uprev=np.array(up), y=y)


# ------------------------------------------------------------------ generalized BVP
class IntervalBVP:
    def __init__(self, F, Cz, Cy_fn, tau, m):
        self.F, self.Cz, self.Cy_fn, self.tau, self.m = F, Cz, Cy_fn, tau, m
        O = np.vstack([Cz @ np.linalg.matrix_power(F, j) for j in range(m)])
        self.O, self.Oinv, self.cond = O, np.linalg.pinv(O), np.linalg.cond(O)

    def valid_until(self, N):
        return (N - self.m + 1) * self.tau

    def operator(self, t, N):
        kmax = N - self.m
        R = np.zeros((len(t), N))
        cache = {}
        for i, ti in enumerate(t):
            k = min(int(np.floor(ti / self.tau + 1e-9)), kmax)
            s = ti - k * self.tau
            key = round(s, 12)
            if key not in cache:
                cache[key] = (self.Cy_fn(s) @ self.Oinv).ravel()
            R[i, k:k + self.m] = cache[key]
        return R

    def reconstruct(self, ys, t):
        return self.operator(t, len(ys)) @ np.asarray(ys)


def delay_model(A, B, C, K, tau, delay, m=None):
    """Augmented state z_k = [x_k; u_{k-1}] for computation delay 'delay' (0 <= delay < tau)."""
    n = A.shape[0]
    K = np.atleast_2d(K)
    Z = ZohCache(A, B)

    def Psi(s):
        if s < delay:
            Ad, Bd = Z(s)
            return np.hstack([Ad, Bd])
        Ad1, Bd1 = Z(delay)
        Ad2, Bd2 = Z(s - delay)
        return Ad2 @ np.hstack([Ad1, Bd1]) + Bd2 @ np.hstack([-K, np.zeros((1, 1))])

    F = np.vstack([Psi(tau), np.hstack([-K, np.zeros((1, 1))])])
    Cz = np.hstack([C, np.zeros((1, 1))])
    return IntervalBVP(F, Cz, lambda s: C @ Psi(s), tau, n + 1 if m is None else m)


def observer_model(A, B, C, K, L, tau, m=None):
    """Augmented state z_k = [x_k; xhat_k] for a prediction observer
    xhat_{k+1} = Ad xhat_k + Bd u_k + L (y_k - C xhat_k),  u_k = -K xhat_k."""
    n = A.shape[0]
    K = np.atleast_2d(K)
    L = np.reshape(L, (n, 1))
    Z = ZohCache(A, B)
    Ad, Bd = Z(tau)
    F = np.block([[Ad, -Bd @ K], [L @ C, Ad - Bd @ K - L @ C]])
    Cz = np.hstack([C, np.zeros((1, n))])

    def Cy(s):
        Ads, Bds = Z(s)
        return C @ np.hstack([Ads, -Bds @ K])

    return IntervalBVP(F, Cz, Cy, tau, 2 * n if m is None else m)


def known_input_bvp(A, B, C, tau, ys, us, uprev, t, delay=0.0):
    """Input-aware BVP (n = 2): uses logged controller outputs; valid for ANY controller."""
    Z = ZohCache(A, B)
    Ad, _ = Z(tau)
    Qi = np.linalg.inv(np.vstack([C, C @ Ad]))

    def forced(s, k):
        if s < delay:
            return Z(s)[1][:, 0] * uprev[k]
        Ad1, Bd1 = Z(delay)
        Ad2, Bd2 = Z(s - delay)
        return Ad2 @ (Bd1[:, 0] * uprev[k]) + Bd2[:, 0] * us[k]

    N = len(ys)
    X = [Qi @ np.array([ys[k], ys[k + 1] - C[0] @ forced(tau, k)]) for k in range(N - 1)]
    y = np.empty(len(t))
    for i, ti in enumerate(t):
        k = min(int(np.floor(ti / tau + 1e-9)), N - 2)
        s = ti - k * tau
        y[i] = C[0] @ (Z(s)[0] @ X[k] + forced(s, k))
    return y, np.linalg.cond(np.vstack([C, C @ Ad]))


# ------------------------------------------------------------------ Kalman / RTS smoother
def rts_smoother(F, C, ys, q, r, P0=100.0):
    """x_{k+1} = F x_k + w (cov q I, or q itself if q is a matrix), y_k = C x_k + v (var r)."""
    n, N = F.shape[0], len(ys)
    I = np.eye(n)
    xf, Pf, xp, Pp = [], [], [], []
    x, P = np.zeros(n), P0 * I
    for k in range(N):
        if k > 0:
            x, P = F @ x, F @ P @ F.T + (q * I if np.ndim(q) == 0 else q)
        xp.append(x)
        Pp.append(P)
        S = (C @ P @ C.T)[0, 0] + r
        G = (P @ C.T)[:, 0] / S
        x = x + G * (ys[k] - C[0] @ x)
        P = (I - np.outer(G, C[0])) @ P
        xf.append(x)
        Pf.append(P)
    xs = [None] * N
    xs[-1] = xf[-1]
    for k in range(N - 2, -1, -1):
        J = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + J @ (xs[k + 1] - xp[k + 1])
    return np.array(xs)


__all__ = ["ZohCache", "motor_stable_K", "simulate_sd", "IntervalBVP", "delay_model", "observer_model",
           "known_input_bvp", "rts_smoother", "MOTOR_A", "MOTOR_B"]
