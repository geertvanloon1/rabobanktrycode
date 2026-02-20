from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class BridgeContext:
    t: int
    R: int
    z: np.ndarray
    beta: np.ndarray
    eta2: float
    y_q: np.ndarray
    quarter_of_month: np.ndarray
    month_pos_in_quarter: np.ndarray

@dataclass
class BridgeObs:
    y: float
    mean_const: float
    H: np.ndarray
    var: float

class BridgeModel:
    def state_lag_count(self) -> int:
        raise NotImplementedError

    def observe(self, ctx: BridgeContext) -> BridgeObs | None:
        raise NotImplementedError

class GDPBridgeEq5(BridgeModel):
    """
    Zhang bridging equation (eq. 5):
      y_k = b0 + b1'(z*F_t) + b2'(z*F_{t-1}) + b3'(z*F_{t-2}) + b4 y_{k-1} + nu
    observed at quarter-end months (month_pos==3), for k>=1.
    """
    def state_lag_count(self) -> int:
        return 2

    def observe(self, ctx: BridgeContext) -> BridgeObs | None:
        t, R = ctx.t, ctx.R
        if ctx.month_pos_in_quarter[t] != 3:
            return None

        k = int(ctx.quarter_of_month[t])
        if k <= 0 or k >= len(ctx.y_q):
            return None

        Z = ctx.z.astype(float)
        beta = ctx.beta
        eta2 = float(max(ctx.eta2, 1e-12))

        b0 = float(beta[0])
        b1 = beta[1:1+R]
        b2 = beta[1+R:1+2*R]
        b3 = beta[1+2*R:1+3*R]
        b4 = float(beta[-1])

        yk = float(ctx.y_q[k])
        ykm1 = float(ctx.y_q[k - 1])

        mean_const = b0 + b4 * ykm1

        # augmented state is [F_t, F_{t-1}, F_{t-2}] => dim 3R
        H = np.zeros(3 * R)
        H[0:R] = b1 * Z
        H[R:2*R] = b2 * Z
        H[2*R:3*R] = b3 * Z

        return BridgeObs(y=yk, mean_const=mean_const, H=H, var=eta2)


def make_augmented_transition(a: np.ndarray, sigma2: np.ndarray, lag_count: int):
    """
    Companion form for S_t = [F_t, F_{t-1}, ..., F_{t-lag_count}]
    Returns (Tm, Qm).
    """
    R = len(a)
    A = np.diag(np.clip(a, -0.999, 0.999))
    Q = np.diag(np.maximum(sigma2, 1e-10))

    blocks = lag_count + 1
    dim = blocks * R

    Tm = np.zeros((dim, dim))
    Tm[0:R, 0:R] = A
    for ell in range(1, blocks):
        Tm[ell*R:(ell+1)*R, (ell-1)*R:ell*R] = np.eye(R)

    Qm = np.zeros((dim, dim))
    Qm[0:R, 0:R] = Q
    return Tm, Qm
