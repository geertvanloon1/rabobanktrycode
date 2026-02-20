import numpy as np


def _unpack_beta(beta: np.ndarray, R: int):
    """
    beta layout consistent with your bridge regression construction:
      [β0, β1(1..R), β2(1..R), β3(1..R), β4]
    """
    beta = np.asarray(beta, float)
    b0 = float(beta[0])
    b1 = beta[1:1 + R]
    b2 = beta[1 + R:1 + 2 * R]
    b3 = beta[1 + 2 * R:1 + 3 * R]
    b4 = float(beta[-1])
    return b0, b1, b2, b3, b4



def substitute_FT_when_xT_empty(
    F: np.ndarray,
    a: np.ndarray,
    x_T_empty: bool,
) -> np.ndarray:
    """
    Zhang (2022), Table 2 / eq. (16)–(18):
    when the current-month panel is empty (x_T = ∅),
    substitute \tilde{F}_T = A F_{T−1} for nowcasting.
    """
    F = np.asarray(F, float)
    a = np.asarray(a, float)

    if not x_T_empty:
        return F

    if F.shape[0] < 2:
        raise ValueError("Need at least 2 months of factors to form A @ F_{T-1}.")

    F_used = F.copy()
    # A is diagonal in this codebase (stored as vector `a`), so A @ F_{T-1} == a * F_{T-1}.
    F_used[-1, :] = a * F_used[-2, :]
    return F_used


def nowcast_mean_next_quarter(
    beta: np.ndarray,
    z: np.ndarray,
    a: np.ndarray,
    F: np.ndarray,
    y_last: float,
    month_pos_in_quarter_last: int,
) -> float:
    """
    Conditional mean nowcast for y_{K+1} given current factor draw path F and params.
    Mirrors Zhang eq (16)-(18) logic:
      mp=1: uses (A^2 F_T, A F_T, F_T)
      mp=2: uses (A F_T, F_T, F_{T-1})
      mp=3: uses (F_T, F_{T-1}, F_{T-2})
    """
    F = np.asarray(F, float)
    z = np.asarray(z, float)
    a = np.asarray(a, float)

    T, R = F.shape
    t = T - 1
    mp = int(month_pos_in_quarter_last)
    if mp not in (1, 2, 3):
        raise ValueError(f"month_pos_in_quarter_last must be 1/2/3, got {mp}")

    b0, b1, b2, b3, b4 = _unpack_beta(beta, R)

    fT = F[t, :]
    AfT = a * fT
    A2fT = (a * a) * fT

    if mp == 1:
        return float(
            b0
            + b1 @ (z * A2fT)
            + b2 @ (z * AfT)
            + b3 @ (z * fT)
            + b4 * float(y_last)
        )

    if mp == 2:
        if t - 1 < 0:
            raise ValueError("Need at least 2 months of factors for mp=2.")
        fTm1 = F[t - 1, :]
        return float(
            b0
            + b1 @ (z * AfT)
            + b2 @ (z * fT)
            + b3 @ (z * fTm1)
            + b4 * float(y_last)
        )

    # mp == 3
    if t - 2 < 0:
        raise ValueError("Need at least 3 months of factors for mp=3.")
    fTm1 = F[t - 1, :]
    fTm2 = F[t - 2, :]
    return float(
        b0
        + b1 @ (z * fT)
        + b2 @ (z * fTm1)
        + b3 @ (z * fTm2)
        + b4 * float(y_last)
    )


def nowcast_draw_next_quarter(
    beta: np.ndarray,
    z: np.ndarray,
    a: np.ndarray,
    sigma2: np.ndarray,
    eta2: float,
    F: np.ndarray,
    y_last: float,
    month_pos_in_quarter_last: int,
    rng=np.random,
) -> float:
    """
    Posterior predictive draw:
      - Simulate missing future monthly factors within the quarter using AR(1) shocks
      - Add GDP noise nu ~ N(0, eta2)

    This produces a draw from p(y_{K+1} | data, params-draw).
    """
    F = np.asarray(F, float)
    z = np.asarray(z, float)
    a = np.asarray(a, float)
    sigma2 = np.asarray(sigma2, float)

    T, R = F.shape
    t = T - 1
    mp = int(month_pos_in_quarter_last)
    if mp not in (1, 2, 3):
        raise ValueError(f"month_pos_in_quarter_last must be 1/2/3, got {mp}")

    b0, b1, b2, b3, b4 = _unpack_beta(beta, R)

    fT = F[t, :]
    nu = np.sqrt(max(float(eta2), 0.0)) * rng.randn()
    sig = np.sqrt(np.maximum(sigma2, 0.0))

    if mp == 1:
        # current month is 1st month of quarter: simulate months 2 and 3
        eps2 = sig * rng.randn(R)
        f2 = a * fT + eps2
        eps3 = sig * rng.randn(R)
        f3 = a * f2 + eps3
        y = (
            b0
            + b1 @ (z * f3)
            + b2 @ (z * f2)
            + b3 @ (z * fT)
            + b4 * float(y_last)
            + nu
        )
        return float(y)

    if mp == 2:
        # current month is 2nd month: simulate month 3
        if t - 1 < 0:
            raise ValueError("Need at least 2 months of factors for mp=2.")
        f1 = F[t - 1, :]
        eps3 = sig * rng.randn(R)
        f3 = a * fT + eps3
        y = (
            b0
            + b1 @ (z * f3)
            + b2 @ (z * fT)
            + b3 @ (z * f1)
            + b4 * float(y_last)
            + nu
        )
        return float(y)

    # mp == 3: quarter complete in factors
    if t - 2 < 0:
        raise ValueError("Need at least 3 months of factors for mp=3.")
    f2 = F[t - 1, :]
    f1 = F[t - 2, :]
    y = (
        b0
        + b1 @ (z * fT)
        + b2 @ (z * f2)
        + b3 @ (z * f1)
        + b4 * float(y_last)
        + nu
    )
    return float(y)
