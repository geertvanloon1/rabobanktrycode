# zhangfunctions/kalman_ffbs.py
from __future__ import annotations

import numpy as np
import numpy.linalg as npl

from zhangnowcast.model.bridging import BridgeContext, make_augmented_transition, BridgeModel


# ============================================================
# Helper: stochastic-volatility covariance "sandwich"
# ============================================================
def sv_cov_from_psi(Psi_o: np.ndarray, omega_o_t: np.ndarray) -> np.ndarray:
    """
    Zhang Section 2.1, Eq. (2):
        Σ_t = D_t Ψ D_t
    where D_t = diag(exp(ω_it)).

    In your notation:
      - Psi_o is the *correlation* submatrix Ψ for the observed series at time t
      - omega_o_t are the log standard deviations ω_it for the same observed series

    We return:
        R_o = D Psi_o D
    which is the measurement covariance for the observed block at time t.

    Note: since D is diagonal, elementwise multiplication is efficient:
        (D Ψ D)_{ij} = Ψ_{ij} * exp(ω_i) * exp(ω_j)
    """
    s = np.exp(omega_o_t)  # exp(ω) = std devs
    return Psi_o * (s[:, None] * s[None, :])


# ============================================================
# FFBS for monthly panel only (no quarterly GDP measurement)
# ============================================================
def ffbs_factors(
    X: np.ndarray,          # (T,n) NaNs ok
    mu: np.ndarray,         # (n,)
    Theta: np.ndarray,      # (n,R) loadings Λ in Zhang Eq. (6)
    z: np.ndarray,          # (R,) selection indicators (diag of Z in Zhang Eq. (6))
    omega: np.ndarray,      # (T,n) log-vols (ignored in Step 2 constant-vol version; pass zeros)
    Psi: np.ndarray,        # (n,n) correlation matrix Ψ in Zhang Eq. (2)
    a: np.ndarray,          # (R,) AR(1) coefficients in Zhang Eq. (4)/(6)
    sigma2: np.ndarray,     # (R,) innovation variances in Zhang Eq. (4)/(6)
    rng=np.random,
):
    """
    This does Forward-Filtering Backward-Sampling (FFBS) for the latent factors F_t
    in a standard linear Gaussian state space model.

    Zhang connections:
    - Monthly measurement equation (Zhang Eq. (6), also Eq. (8) conceptually):
          x_t = μ + Θ (Z ⊙ F_t) + ε_t
      Here we implement it row-wise with missing data. (Z ⊙ F_t) is achieved by
      multiplying columns of Θ by z (binary 0/1).
    - Factor state equation (Zhang Eq. (4)/(6)):
          F_t = A F_{t-1} + u_t,  u_t ~ N(0, diag(σ^2))

    Missing data:
    - At each month t we subset the observed series and run the Kalman update
      on that observed block only (this mirrors Zhang Eq. (12) idea: selecting
      available entries).

    Note:
    - This function is the "monthly-only" smoother. Your joint XY version below
      extends this idea by augmenting the state with lags and adding the GDP bridge
      observation at quarter-end months, consistent with the factorization in
      Zhang Eq. (11). :contentReference[oaicite:2]{index=2}
    """
    X = np.asarray(X, float)
    T, n = X.shape
    R = len(z)

    Z = z.astype(float)

    # A and Q correspond to Zhang Eq. (6) factor transition parameters
    A = np.diag(np.clip(a, -0.999, 0.999))
    Q = np.diag(np.maximum(sigma2, 1e-10))

    # Prior for F_0 (not specified explicitly in Zhang; typical diffuse-ish prior)
    m_prev = np.zeros(R)
    P_prev = np.eye(R) * 10.0

    # Store predicted and filtered moments for backward simulation
    a_pred = np.zeros((T, R))
    P_pred = np.zeros((T, R, R))
    a_filt = np.zeros((T, R))
    P_filt = np.zeros((T, R, R))

    I_R = np.eye(R)

   
    # Forward Kalman filter
   
    for t in range(T):
        # Predict step
        m_pr = A @ m_prev
        P_pr = A @ P_prev @ A.T + Q

        a_pred[t], P_pred[t] = m_pr, P_pr

        obs = ~np.isnan(X[t])
        if not np.any(obs):
            # No measurement update this month
            a_filt[t], P_filt[t] = m_pr, P_pr
            m_prev, P_prev = m_pr, P_pr
            continue

        x_o = X[t, obs]
        mu_o = mu[obs]

        # H_o = Θ_obs * z, i.e. mask factor loadings by selected factors
        H_o = Theta[obs, :] * Z[None, :]   # (n_obs,R)

        # Measurement covariance: in this monthly-only FFBS we use Ψ (or its observed block)
        # This corresponds to constant-vol case; your XY/loglik code uses SV via sv_cov_from_psi.
        R_o = Psi[np.ix_(obs, obs)]
        R_o = 0.5 * (R_o + R_o.T) + 1e-10 * np.eye(R_o.shape[0])

        # Innovation
        v = x_o - (mu_o + H_o @ m_pr)
        S = H_o @ P_pr @ H_o.T + R_o
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(S.shape[0])

       
        # Improvement A: gain via linear solve / Cholesky (no explicit inverse)
       
        cholS = npl.cholesky(S)
        # Solve S^{-1} H P by two triangular solves:
        # K = P H' S^{-1}
        # We compute K' = S^{-1} (H P)' then transpose.
        HPt = H_o @ P_pr
        # Solve cholS * y = HPt  => y = cholS^{-1} HPt
        y = npl.solve(cholS, HPt)
        # Solve cholS.T * w = y => w = S^{-1} HPt
        w = npl.solve(cholS.T, y)
        # Now w = S^{-1} (H P); so K = P H' S^{-1} = (S^{-1} H P)'.
        K = w.T

        m_po = m_pr + K @ v

       
        # Improvement B: Joseph-form covariance update (more PSD-stable)
        # P_po = (I-KH)P_pr(I-KH)' + K R K'
       
        I_KH = I_R - K @ H_o
        P_po = I_KH @ P_pr @ I_KH.T + K @ R_o @ K.T
        P_po = 0.5 * (P_po + P_po.T)

        a_filt[t], P_filt[t] = m_po, P_po
        m_prev, P_prev = m_po, P_po

   
    # Backward simulation (Carter-Kohn / simulation smoother)
   
    F = np.zeros((T, R))

    L = npl.cholesky(P_filt[T - 1] + 1e-10 * I_R)
    F[T - 1] = a_filt[T - 1] + L @ rng.randn(R)

    for t in range(T - 2, -1, -1):
        Pp = P_pred[t + 1]
        # Smoother gain J = P_filt[t] A' P_pred[t+1]^{-1}
        J = (P_filt[t] @ A.T) @ npl.solve(Pp, I_R)

        mean = a_filt[t] + J @ (F[t + 1] - a_pred[t + 1])
        cov = P_filt[t] - J @ Pp @ J.T
        cov = 0.5 * (cov + cov.T)

        L = npl.cholesky(cov + 1e-10 * I_R)
        F[t] = mean + L @ rng.randn(R)

    return F


# ============================================================
# Log p(X | ...) integrating out factors via Kalman filter
# (monthly panel only, with SV covariance)
# ============================================================
def kalman_loglik_X(
    X, mu, Theta, z, omega, Psi, a, sigma2
) -> float:
    """
    Computes:
        log p(X | mu, Theta, z, omega, Psi, a, sigma2)
    by integrating out the latent factors F using the Kalman filter.

    Zhang connection:
    - This is the "collapsed likelihood" idea inside the posterior factorization
      in Zhang Eq. (11): integrate out latent F to evaluate likelihood terms
      for selection steps, etc. :contentReference[oaicite:3]{index=3}
    - Measurement covariance uses SV sandwich (Zhang Eq. (2)-(3)).

    Missing data:
    - At each t, use only observed monthly series.

    Improvement C:
    - Use Cholesky for quadratic form and logdet, and use Joseph form in update.
    """
    X = np.asarray(X, float)
    T, n = X.shape
    R = Theta.shape[1]
    Z = z.astype(float)

    A = np.diag(np.clip(a, -0.999, 0.999))
    Q = np.diag(np.maximum(sigma2, 1e-10))

    m_prev = np.zeros(R)
    P_prev = np.eye(R) * 10.0

    ll = 0.0
    log2pi = np.log(2.0 * np.pi)

    I_R = np.eye(R)

    for t in range(T):
        # predict
        m_pr = A @ m_prev
        P_pr = A @ P_prev @ A.T + Q

        obs = ~np.isnan(X[t])
        m = int(obs.sum())
        if m == 0:
            m_prev, P_prev = m_pr, P_pr
            continue

        x_o = X[t, obs]
        mu_o = mu[obs]
        H_o = Theta[obs, :] * Z[None, :]  # (m, R)

        Psi_o = Psi[np.ix_(obs, obs)]
        Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(m)

        omega_o_t = omega[t, obs]
        R_o = sv_cov_from_psi(Psi_o, omega_o_t)
        R_o = 0.5 * (R_o + R_o.T) + 1e-10 * np.eye(m)

        v = x_o - (mu_o + H_o @ m_pr)
        S = H_o @ P_pr @ H_o.T + R_o
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(m)

        # log N(v;0,S): use Cholesky
        cholS = npl.cholesky(S)
        y = npl.solve(cholS, v)
        quad = float(y @ y)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cholS))))

        ll += -0.5 * (m * log2pi + logdet + quad)

       
        # Improvement A: gain via solves, not explicit inverse
       
        HPt = H_o @ P_pr
        y2 = npl.solve(cholS, HPt)
        w2 = npl.solve(cholS.T, y2)
        K = w2.T

        m_po = m_pr + K @ v

       
        # Improvement B: Joseph-form covariance update
       
        I_KH = I_R - K @ H_o
        P_po = I_KH @ P_pr @ I_KH.T + K @ R_o @ K.T
        P_po = 0.5 * (P_po + P_po.T)

        m_prev, P_prev = m_po, P_po

    return float(ll)


# ============================================================
# Joint loglik: monthly X + quarterly GDP bridge, integrate out factors
# ============================================================
def kalman_loglik_XY(
    X, y_q, quarter_of_month, month_pos_in_quarter,
    mu, Theta, z, omega, Psi, a, sigma2,
    beta, eta2,
    bridge: BridgeModel,
) -> float:
    """
    Joint log-likelihood:
        log p(X, Y | params)
    integrating out the latent factors via a Kalman filter on an augmented state.

    Zhang connection:
    - Monthly equation: Zhang Eq. (6)/(8)/(12)
    - Quarterly bridge: Zhang Eq. (5) (rewritten using Z in Eq. (6))
    - The product structure is in Zhang Eq. (11); here we evaluate the joint
      measurement likelihood by stacking the monthly measurement block and the
      quarterly measurement (when present) into one combined update per month. :contentReference[oaicite:4]{index=4}

    How the bridge is inserted:
    - Your BridgeModel.observe(ctx) returns a scalar observation at quarter-end months
      (month_pos == 3), with a linear measurement row H over the augmented state.

    Improvements A/B/C applied here too (stable gain, Joseph form, Cholesky).
    """
    X = np.asarray(X, float)
    y_q = np.asarray(y_q, float)
    quarter_of_month = np.asarray(quarter_of_month, int)
    month_pos_in_quarter = np.asarray(month_pos_in_quarter, int)

    T, n = X.shape
    R = Theta.shape[1]
    Z = z.astype(float)

    lag_count = int(bridge.state_lag_count())
    Tm, Qm = make_augmented_transition(a, sigma2, lag_count)
    dim = Tm.shape[0]

    # Prior for augmented state S_0 = [F_0, F_-1, F_-2, ...] (diffuse-ish)
    m_prev = np.zeros(dim)
    P_prev = np.eye(dim) * 10.0

    ll = 0.0
    log2pi = np.log(2.0 * np.pi)
    I_dim = np.eye(dim)

    for t in range(T):
        # predict
        m_pr = Tm @ m_prev
        P_pr = Tm @ P_prev @ Tm.T + Qm

        # monthly observed block size
        obs = ~np.isnan(X[t])
        mX = int(obs.sum())

        # quarterly bridge observation at this month?
        ctx = BridgeContext(
            t=t, R=R, z=z, beta=beta, eta2=float(eta2),
            y_q=y_q, quarter_of_month=quarter_of_month,
            month_pos_in_quarter=month_pos_in_quarter
        )
        bobs = bridge.observe(ctx)
        mY = 1 if bobs is not None else 0

        mTot = mX + mY
        if mTot == 0:
            m_prev, P_prev = m_pr, P_pr
            continue

        # Build stacked measurement system:
        #   y_meas = mean_const + H * state + eps
        y_meas = np.zeros(mTot)
        mean_meas = np.zeros(mTot)
        H = np.zeros((mTot, dim))
        Rm = np.zeros((mTot, mTot))

        # --- Monthly X block ---
        if mX > 0:
            idx = np.where(obs)[0]
            x_o = X[t, idx]
            mu_o = mu[idx]
            Hx = Theta[idx, :] * Z[None, :]  # (mX, R)

            y_meas[0:mX] = x_o
            mean_meas[0:mX] = mu_o
            H[0:mX, 0:R] = Hx  # monthly depends on current factor block only

            Psi_o = Psi[np.ix_(idx, idx)]
            Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(mX)

            omega_o_t = omega[t, idx]
            R_o = sv_cov_from_psi(Psi_o, omega_o_t)
            R_o = 0.5 * (R_o + R_o.T) + 1e-10 * np.eye(mX)

            Rm[0:mX, 0:mX] = R_o

        # --- Quarterly GDP bridge block (one scalar) ---
        if bobs is not None:
            row = mX
            if bobs.H.shape[0] != dim:
                raise ValueError(f"Bridge H has dim {bobs.H.shape[0]} but state dim is {dim}")

            y_meas[row] = float(bobs.y)
            mean_meas[row] = float(bobs.mean_const)
            H[row, :] = bobs.H
            Rm[row, row] = float(max(bobs.var, 1e-12))

        # innovation
        v = y_meas - (mean_meas + H @ m_pr)
        S = H @ P_pr @ H.T + Rm
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(mTot)

        # loglik via Cholesky (Improvement C)
        cholS = npl.cholesky(S)
        ytmp = npl.solve(cholS, v)
        quad = float(ytmp @ ytmp)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cholS))))
        ll += -0.5 * (mTot * log2pi + logdet + quad)

       
        # Improvement A: stable gain
       
        HPt = H @ P_pr
        y2 = npl.solve(cholS, HPt)
        w2 = npl.solve(cholS.T, y2)
        K = w2.T  # (dim, mTot)

        m_po = m_pr + K @ v

       
        # Improvement B: Joseph-form update
       
        I_KH = I_dim - K @ H
        P_po = I_KH @ P_pr @ I_KH.T + K @ Rm @ K.T
        P_po = 0.5 * (P_po + P_po.T)

        m_prev, P_prev = m_po, P_po

    return float(ll)


# ============================================================
# FFBS for joint monthly X + quarterly GDP bridge
# (sample factors by smoothing on the augmented state)
# ============================================================
def ffbs_factors_XY(
    X, y_q, quarter_of_month, month_pos_in_quarter,
    mu, Theta, z, omega, Psi, a, sigma2,
    beta, eta2,
    bridge: BridgeModel,
    rng=np.random,
):
    """
    This is the joint FFBS sampler used in your main MCMC.

    Concept:
    - We augment the state to carry factor lags needed by Zhang’s bridge Eq. (5):
        y_k depends on F_{3k}, F_{3k-1}, F_{3k-2}.
      So we build an augmented state:
        S_t = [F_t, F_{t-1}, F_{t-2}]  (lag_count=2)
      and do Kalman filtering + backward sampling on S_t.

    Zhang connection:
    - The idea matches the mixed-frequency likelihood factorization (Eq. (11)):
      we treat GDP as an additional measurement that arrives at quarter-end months,
      and we handle ragged-edge monthly data by subsetting observed X series each month. :contentReference[oaicite:5]{index=5}

    Returns:
    - Only the current factor block F_t = S_t[0:R] for all t (shape (T,R)),
      because the rest are just lags used for the bridge measurement.
    """
    X = np.asarray(X, float)
    y_q = np.asarray(y_q, float)
    quarter_of_month = np.asarray(quarter_of_month, int)
    month_pos_in_quarter = np.asarray(month_pos_in_quarter, int)

    T, n = X.shape
    R = Theta.shape[1]
    Z = z.astype(float)

    lag_count = int(bridge.state_lag_count())
    Tm, Qm = make_augmented_transition(a, sigma2, lag_count)
    dim = Tm.shape[0]

    # prior for S_0
    m_prev = np.zeros(dim)
    P_prev = np.eye(dim) * 10.0

    a_pred = np.zeros((T, dim))
    P_pred = np.zeros((T, dim, dim))
    a_filt = np.zeros((T, dim))
    P_filt = np.zeros((T, dim, dim))

    I_dim = np.eye(dim)

   
    # Forward filter
   
    for t in range(T):
        # predict
        m_pr = Tm @ m_prev
        P_pr = Tm @ P_prev @ Tm.T + Qm

        a_pred[t], P_pred[t] = m_pr, P_pr

        obs = ~np.isnan(X[t])
        mX = int(obs.sum())

        # bridge observation?
        ctx = BridgeContext(
            t=t, R=R, z=z, beta=beta, eta2=float(eta2),
            y_q=y_q, quarter_of_month=quarter_of_month,
            month_pos_in_quarter=month_pos_in_quarter
        )
        bobs = bridge.observe(ctx)
        mY = 1 if bobs is not None else 0

        mTot = mX + mY
        if mTot == 0:
            a_filt[t], P_filt[t] = m_pr, P_pr
            m_prev, P_prev = m_pr, P_pr
            continue

        y_meas = np.zeros(mTot)
        mean_meas = np.zeros(mTot)
        H = np.zeros((mTot, dim))
        Rm = np.zeros((mTot, mTot))

        # monthly X block
        if mX > 0:
            idx = np.where(obs)[0]
            x_o = X[t, idx]
            mu_o = mu[idx]
            Hx = Theta[idx, :] * Z[None, :]  # (mX, R)

            y_meas[0:mX] = x_o
            mean_meas[0:mX] = mu_o
            H[0:mX, 0:R] = Hx

            Psi_o = Psi[np.ix_(idx, idx)]
            Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(mX)

            omega_o_t = omega[t, idx]
            R_o = sv_cov_from_psi(Psi_o, omega_o_t)
            R_o = 0.5 * (R_o + R_o.T) + 1e-10 * np.eye(mX)

            Rm[0:mX, 0:mX] = R_o

        # quarterly bridge scalar
        if bobs is not None:
            row = mX
            if bobs.H.shape[0] != dim:
                raise ValueError(f"Bridge H has dim {bobs.H.shape[0]} but state dim is {dim}")
            y_meas[row] = float(bobs.y)
            mean_meas[row] = float(bobs.mean_const)
            H[row, :] = bobs.H
            Rm[row, row] = float(max(bobs.var, 1e-12))

        # update
        v = y_meas - (mean_meas + H @ m_pr)
        S = H @ P_pr @ H.T + Rm
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(mTot)

        # Improvement A/C: use Cholesky for gain stability
        cholS = npl.cholesky(S)
        HPt = H @ P_pr
        y2 = npl.solve(cholS, HPt)
        w2 = npl.solve(cholS.T, y2)
        K = w2.T

        m_po = m_pr + K @ v

        # Improvement B: Joseph-form covariance update
        I_KH = I_dim - K @ H
        P_po = I_KH @ P_pr @ I_KH.T + K @ Rm @ K.T
        P_po = 0.5 * (P_po + P_po.T)

        a_filt[t], P_filt[t] = m_po, P_po
        m_prev, P_prev = m_po, P_po

   
    # Backward sampling
   
    Sdraw = np.zeros((T, dim))

    L = npl.cholesky(P_filt[T - 1] + 1e-10 * I_dim)
    Sdraw[T - 1] = a_filt[T - 1] + L @ rng.randn(dim)

    for t in range(T - 2, -1, -1):
        Pp = P_pred[t + 1]
        J = (P_filt[t] @ Tm.T) @ npl.solve(Pp, I_dim)

        mean = a_filt[t] + J @ (Sdraw[t + 1] - a_pred[t + 1])
        cov = P_filt[t] - J @ Pp @ J.T
        cov = 0.5 * (cov + cov.T)

        L = npl.cholesky(cov + 1e-10 * I_dim)
        Sdraw[t] = mean + L @ rng.randn(dim)

    # Return the current factor block F_t from augmented state S_t
    return Sdraw[:, 0:R]
