import numpy as np
import numpy.linalg as npl


def sample_beta_eta2(
    y_q: np.ndarray,
    F: np.ndarray,
    z: np.ndarray,
    quarter_of_month: np.ndarray,
    month_pos_in_quarter: np.ndarray,
    eta2_current: float,              # NEW: current η² from the Markov chain
    alpha_h: float = 2.0,
    beta_h: float = 1e-4,
    rng=np.random,
) -> tuple[np.ndarray, float]:
    """
    What this code does (plain language):
    ------------------------------------
    This draws the GDP bridge regression parameters:
        • beta  (the regression coefficients in the GDP bridge)
        • eta2  (the GDP regression noise variance)

    Why this exists in the Zhang DFM:
    --------------------------------
    Zhang links quarterly GDP to the monthly factors via the bridge equation
    (their Eq. (5)). The GDP term appears in the MCMC factorization as the
    conditional density p(y_k | beta, F, y_{k-1}, eta2) (their Eq. (10),
    referenced inside their Eq. (11)).

    In our codebase we use the "selected factors" Z ⊙ F (z is 0/1 per factor),
    so the regressors are Z*F_{3k}, Z*F_{3k-1}, Z*F_{3k-2}, and y_{k-1}.

    Priors (Zhang):
    --------------
      beta ~ N(0, I)
      eta2 ~ Inverse-Gamma(alpha_h, beta_h) 

    How the sampling works (Gibbs step):
    -----------------------------------
    Because of conjugacy, we can do:
      (1) beta | eta2, data   ~ Normal
      (2) eta2 | beta, data   ~ Inverse-Gamma

    IMPORTANT:
    ----------
    beta's conditional distribution depends on the *current* eta2 from the chain.
    That is why eta2_current is an input.
    """
    y_q = np.asarray(y_q, float)
    F = np.asarray(F, float)
    z = np.asarray(z, int)
    quarter_of_month = np.asarray(quarter_of_month)
    month_pos_in_quarter = np.asarray(month_pos_in_quarter)

    K = len(y_q)           # number of quarters in y_q
    T, R = F.shape         # monthly factor path length and number of factors
    Z = z.astype(float)

    # ------------------------------------------------------------------
    # Step A: Build the regression dataset (Xreg, yreg)
    #
    # For each quarter k, Zhang uses the three months of that quarter:
    #   F_{3k}, F_{3k-1}, F_{3k-2}
    # plus lagged GDP y_{k-1}.
    #
    # Our calendar arrays (quarter_of_month, month_pos_in_quarter) tell us:
    #   - which months belong to quarter k
    #   - which month is position 1/2/3 inside that quarter
    #
    # Each row of Xreg is:
    #   [ 1,
    #     Z*F_{t3}, Z*F_{t2}, Z*F_{t1},
    #     y_{k-1} ]
    # where t3 is the quarter-end month, t2 the middle month, t1 the first month.
    # ------------------------------------------------------------------
    rows = []
    targ = []

    for k in range(1, K):
        # months that belong to quarter k
        ts = np.where(quarter_of_month == k)[0]
        if ts.size < 3:
            # not enough monthly observations to form the 3-month bridge
            continue
        ts = np.sort(ts)

        # Prefer using explicit month positions (1,2,3) to avoid off-by-one mistakes
        t3_c = ts[month_pos_in_quarter[ts] == 3]
        t2_c = ts[month_pos_in_quarter[ts] == 2]
        t1_c = ts[month_pos_in_quarter[ts] == 1]

        if t3_c.size > 0 and t2_c.size > 0 and t1_c.size > 0:
            t3 = int(t3_c[-1])   # quarter-end month index in the monthly timeline
            t2 = int(t2_c[-1])
            t1 = int(t1_c[-1])
        else:
            # Fallback: last three months tagged as quarter k
            # (should be consistent if quarter tagging is consistent)
            t1, t2, t3 = map(int, ts[-3:])

        # Safety guard: t3 must be within the factor sample
        if not (0 <= t1 < T and 0 <= t2 < T and 0 <= t3 < T):
            continue

        xrow = np.r_[
            1.0,
            Z * F[t3, :],
            Z * F[t2, :],
            Z * F[t1, :],
            y_q[k - 1],
        ]
        rows.append(xrow)
        targ.append(y_q[k])

    # beta layout is always [β0, β1(R), β2(R), β3(R), β4]
    p = 1 + 3 * R + 1

    if len(rows) == 0:
        # No usable GDP rows: fall back to priors (rare, only very short samples)
        beta = rng.randn(p)  # N(0,I)
        eta2_new = 1.0 / rng.gamma(alpha_h, 1.0 / (beta_h + 1e-12))
        return beta, float(eta2_new)

    Xreg = np.vstack(rows)          # shape (nobs, p)
    yreg = np.asarray(targ, float)  # shape (nobs,)
    nobs = yreg.shape[0]

    # ------------------------------------------------------------------
    # Step B: Draw beta | eta2_current
    #
    # Model: yreg ~ N(Xreg beta, eta2 I)
    # Prior: beta ~ N(0, I)
    #
    # => posterior is Normal:
    #    Vn_inv = I + (X'X)/eta2
    #    mn     = Vn * (X'y)/eta2
    # ------------------------------------------------------------------
    eta2 = float(max(eta2_current, 1e-12))

    V0_inv = np.eye(p)          # prior precision for beta (since prior covariance = I)
    XtX = Xreg.T @ Xreg
    Xty = Xreg.T @ yreg

    Vn_inv = V0_inv + XtX / eta2
    Vn = npl.inv(Vn_inv)
    mn = Vn @ (Xty / eta2)

    L = npl.cholesky(Vn + 1e-12 * np.eye(p))
    beta = mn + L @ rng.randn(p)

    # ------------------------------------------------------------------
    # Step C: Draw eta2 | beta
    #
    # Residuals: e = y - X beta
    # With IG prior, posterior is IG:
    #    alpha_post = alpha_h + n/2
    #    beta_post  = beta_h  + 0.5 * sum(e^2)
    # ------------------------------------------------------------------
    resid = yreg - Xreg @ beta
    alpha_post = alpha_h + 0.5 * nobs
    beta_post = beta_h + 0.5 * float(resid @ resid)

    eta2_new = 1.0 / rng.gamma(alpha_post, 1.0 / (beta_post + 1e-12))
    return beta, float(eta2_new)




def _unpack_beta(beta: np.ndarray, R: int):
    b0 = float(beta[0])
    b1 = beta[1:1+R]
    b2 = beta[1+R:1+2*R]
    b3 = beta[1+2*R:1+3*R]
    b4 = float(beta[-1])
    return b0, b1, b2, b3, b4

def nowcast_mean_next_quarter(beta, z, a, F, y_last, month_pos_in_quarter_last):
    """
    Zhang eq (16)-(18): conditional mean E[y_{K+1} | F_T, params]
    month_pos_in_quarter_last is 1/2/3 for the *current* month T.
    """
    F = np.asarray(F)
    z = np.asarray(z).astype(float)
    a = np.asarray(a).astype(float)

    T, R = F.shape
    t = T - 1
    b0, b1, b2, b3, b4 = _unpack_beta(beta, R)

    fT = F[t, :]
    AfT  = a * fT
    A2fT = (a * a) * fT

    mp = int(month_pos_in_quarter_last)

    if mp == 1:
        # eq (16)
        return (b0
                + b1 @ (z * A2fT)
                + b2 @ (z * AfT)
                + b3 @ (z * fT)
                + b4 * float(y_last))
    elif mp == 2:
        # eq (17) needs F_{T-1}
        if t - 1 < 0:
            raise ValueError("Need at least 2 months of factors for mp=2.")
        fTm1 = F[t-1, :]
        return (b0
                + b1 @ (z * AfT)
                + b2 @ (z * fT)
                + b3 @ (z * fTm1)
                + b4 * float(y_last))
    elif mp == 3:
        # eq (18) needs F_{T-1}, F_{T-2}
        if t - 2 < 0:
            raise ValueError("Need at least 3 months of factors for mp=3.")
        fTm1 = F[t-1, :]
        fTm2 = F[t-2, :]
        return (b0
                + b1 @ (z * fT)
                + b2 @ (z * fTm1)
                + b3 @ (z * fTm2)
                + b4 * float(y_last))
    else:
        raise ValueError(f"month_pos must be 1/2/3, got {mp}")

def nowcast_draw_next_quarter(beta, z, a, sigma2, eta2, F, y_last, month_pos_in_quarter_last, rng=np.random):
    """
    Posterior predictive draw:
      - simulate missing future monthly factors in the quarter using AR(1) shocks
      - then simulate GDP noise nu ~ N(0, eta2)
    This is what you typically score/evaluate vs realized GDP.
    """
    F = np.asarray(F)
    z = np.asarray(z).astype(float)
    a = np.asarray(a).astype(float)
    sigma2 = np.asarray(sigma2).astype(float)

    T, R = F.shape
    t = T - 1
    b0, b1, b2, b3, b4 = _unpack_beta(beta, R)
    mp = int(month_pos_in_quarter_last)

    fT = F[t, :]
    nu = np.sqrt(max(float(eta2), 0.0)) * rng.randn()

    if mp == 1:
        # current month is first month of the quarter; need simulate months 2 and 3
        eps1 = np.sqrt(np.maximum(sigma2, 0.0)) * rng.randn(R)
        f2 = a * fT + eps1
        eps2 = np.sqrt(np.maximum(sigma2, 0.0)) * rng.randn(R)
        f3 = a * f2 + eps2
        y = (b0
             + b1 @ (z * f3)
             + b2 @ (z * f2)
             + b3 @ (z * fT)
             + b4 * float(y_last)
             + nu)
        return float(y)

    if mp == 2:
        # current month is second month; need simulate month 3
        if t - 1 < 0:
            raise ValueError("Need at least 2 months of factors for mp=2.")
        f1 = F[t-1, :]
        eps1 = np.sqrt(np.maximum(sigma2, 0.0)) * rng.randn(R)
        f3 = a * fT + eps1
        y = (b0
             + b1 @ (z * f3)
             + b2 @ (z * fT)
             + b3 @ (z * f1)
             + b4 * float(y_last)
             + nu)
        return float(y)

    if mp == 3:
        # quarter complete in factors; no future factor simulation needed
        if t - 2 < 0:
            raise ValueError("Need at least 3 months of factors for mp=3.")
        f2 = F[t-1, :]
        f1 = F[t-2, :]
        y = (b0
             + b1 @ (z * fT)
             + b2 @ (z * f2)
             + b3 @ (z * f1)
             + b4 * float(y_last)
             + nu)
        return float(y)

    raise ValueError(f"month_pos must be 1/2/3, got {mp}")
