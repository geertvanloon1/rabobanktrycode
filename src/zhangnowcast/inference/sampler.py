from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import numpy.linalg as npl
import math
import pandas as pd

from zhangnowcast.data.data import ZhangData
from zhangnowcast.inference.kalman_ffbs import (
    ffbs_factors,
    kalman_loglik_X,
    kalman_loglik_XY,
    ffbs_factors_XY,
)
from zhangnowcast.model.gdp_block import sample_beta_eta2
from zhangnowcast.model.factor_params import sample_a_sigma2, enforce_ident_order
from zhangnowcast.model.selection import sample_z_p_pi
from zhangnowcast.model.lkj_psi import sample_Psi_LKJ1_columnwise_rw
from zhangnowcast.inference.sv_pgbs import (
    compute_monthly_residuals,
    sample_omega_pgbs,
    sample_tau2,
)
from zhangnowcast.nowcast.nowcast import nowcast_draw_next_quarter, substitute_FT_when_xT_empty
from zhangnowcast.model.bridging import GDPBridgeEq5

bridge = GDPBridgeEq5()


@dataclass
class BayResult:
    # bookkeeping
    draws_kept: int

    # posterior summaries (standardized scale unless standardize=False)
    F_mean: np.ndarray           # (T, R)
    F_sd: np.ndarray             # (T, R)
    mu_mean: np.ndarray          # (n,)
    Theta_mean: np.ndarray       # (n, R)   (loadings Lambda)
    Theta_sd: np.ndarray         # (n, R)

    # factor selection
    z_mean: np.ndarray           # (R,)
    z_draws: np.ndarray          # (S, R) int8
    r_draws: np.ndarray          # (S,)

    # GDP bridge regression + factor dynamics
    beta_mean: np.ndarray        # (1 + 3R + 1,)
    a_mean: np.ndarray           # (R,)
    sigma2_mean: np.ndarray      # (R,)
    eta2_mean: float             # scalar

    # correlated errors / SV
    Psi_mean: np.ndarray         # (n, n)
    omega_mean: np.ndarray       # (T, n)
    omega_sd: np.ndarray         # (T, n)
    tau2_mean: float             # scalar

    # ---- multi-target nowcast panel ----
    nowcast_targets: List[str]        # length J
    nowcast_draws_matrix: np.ndarray  # (S, J)
    nowcast_means_vector: np.ndarray  # (J,)
    nowcast_sds_vector: np.ndarray    # (J,)

    # backward compatibility (current-quarter only; last target)
    nowcast_mean: float
    nowcast_sd: float
    nowcast_draws: np.ndarray         # (S,)

    # standardization params (for interpretability in artifacts)
    standardize_mean: np.ndarray      # (n,)
    standardize_sd: np.ndarray        # (n,)

    # diagnostics + config snapshot
    diagnostics: Dict[str, Any]
    config: Dict[str, Any]


@dataclass
class BayHyperParams:
    """
    Hyperparameters used by BAY / BAY-SV.

    Parameterization note: all Inverse-Gamma priors in this code use
    IG(shape=alpha, scale=beta), sampled via:
        x = 1 / Gamma(shape=alpha, scale=1/beta)
    """
    alpha_s: float
    beta_s: float

    alpha_h: float
    beta_h: float

    omega1_mean: float
    omega1_var: float

    alpha_l: float
    beta_l: float

    n_particles: int

    alpha_p: float = 1.0
    beta_p: float = 3.0
    alpha_pi: float = 2.0
    beta_pi: float = 2.0


def resolve_bay_hyperparams(*, R: int, n: int, paper_defaults: bool) -> BayHyperParams:
    """
    Zhang (2022) uses:
      (alpha_s, beta_s)=(2, R+2), (alpha_h, beta_h)=(2, 0.0001),
      omega_{i1}~N(0,1), (alpha_l, beta_l)=(2, n-1),
      and P=5 particles in Simulation 2 (SV).
    """
    if paper_defaults:
        return BayHyperParams(
            alpha_s=2.0,
            beta_s=float(R + 2),
            alpha_h=2.0,
            beta_h=1e-4,
            omega1_mean=0.0,
            omega1_var=1.0,
            alpha_l=2.0,
            beta_l=0.2,  # float(max(n - 1, 1))
            n_particles=20,  # 5
        )

    # legacy / stability defaults
    return BayHyperParams(
        alpha_s=2.0,
        beta_s=0.5,
        alpha_h=2.0,
        beta_h=1e-4,
        omega1_mean=0.0,
        omega1_var=10.0,
        alpha_l=2.0,
        beta_l=0.2,
        n_particles=10,
    )


def _standardize_panel(X: np.ndarray):
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0, ddof=0)
    s = np.where(s == 0, 1.0, s)
    Xs = (X - m[None, :]) / s[None, :]
    return Xs, m, s


def _loglik_X_given_F(X: np.ndarray, mu: np.ndarray, Theta: np.ndarray, z: np.ndarray, F: np.ndarray) -> float:
    """
    For BAY base we assume Σ=I (so diagonal, unit variance).
    Likelihood: sum_{obs} -0.5[(x-mean)^2 + log(2π)]
    """
    T, n = X.shape
    Z = z.astype(float)
    ll = 0.0
    c = -0.5 * np.log(2 * np.pi)
    for t in range(T):
        obs = ~np.isnan(X[t])
        if not np.any(obs):
            continue
        mean = mu[obs] + (Theta[obs, :] * Z[None, :]) @ F[t, :]
        e = X[t, obs] - mean
        ll += np.sum(c - 0.5 * e * e)
    return float(ll)


def _loglik_toggle_zj(X, mu, Theta, z, F, j):
    """
    Return loglik under z_j=0 and z_j=1, keeping F fixed (this is the standard conditional
    used in a Gibbs-within-Gibbs scheme; then F is re-sampled next iteration).
    """
    z0 = z.copy(); z0[j] = 0
    z1 = z.copy(); z1[j] = 1
    ll0 = _loglik_X_given_F(X, mu, Theta, z0, F)
    ll1 = _loglik_X_given_F(X, mu, Theta, z1, F)
    return ll0, ll1


def _sample_mu_theta(X: np.ndarray, F: np.ndarray, z: np.ndarray, rng=np.random):
    """
    BAY base block with Σ=I:
      X_{t,i} = mu_i + Theta_i (z ⊙ F_t) + eps, eps~N(0,1)
    Priors:
      mu_i ~ N(0,1)
      Theta_{i,:} ~ N(0, I_R)  (equivalent to MN(0,I_n,I_R) row-wise)
    """
    T, n = X.shape
    R = F.shape[1]
    ZF = F * z.astype(float)[None, :]   # (T,R)

    mu = np.zeros(n)
    Theta = np.zeros((n, R))

    for i in range(n):
        obs = ~np.isnan(X[:, i])
        yi = X[obs, i]
        Xi = ZF[obs, :]  # (n_obs,R)

        D = np.column_stack([np.ones(len(yi)), Xi])  # (n_obs, 1+R)

        P0 = np.eye(1 + R)
        P0[0, 0] = 1.0

        Pn = P0 + D.T @ D
        Vn = npl.inv(Pn)
        mn = Vn @ (D.T @ yi)

        L = np.linalg.cholesky(Vn + 1e-12 * np.eye(1 + R))
        draw = mn + L @ rng.randn(1 + R)

        mu[i] = draw[0]
        Theta[i, :] = draw[1:]

    return mu, Theta


def run_bay(
    D: ZhangData,
    R_max: int = 4,
    n_iter: int = 2000,
    burn: int = 1000,
    thin: int = 5,
    standardize: bool = True,
    seed: Optional[int] = 123,
    paper_defaults: bool = True,
    use_sv: bool = True,
    do_selection: bool = True,
):
    rng = np.random.RandomState(seed) if seed is not None else np.random

    X = D.X_m.copy()

    # Standardize monthly panel to match priors scaling
    if standardize:
        X, mX, sX = _standardize_panel(X)
    else:
        mX = np.zeros(X.shape[1])
        sX = np.ones(X.shape[1])

    T, n = X.shape
    R = R_max

    h = resolve_bay_hyperparams(R=R, n=n, paper_defaults=paper_defaults)

    # Initialize parameters
    z = np.ones(R, dtype=int)
    z[0] = 1
    p = np.zeros(R)
    p[0] = 1.0
    p[1:] = 0.5
    s = np.ones(R - 1, dtype=int)
    pi = 0.5

    mu = np.zeros(n)
    Theta = 0.1 * rng.randn(n, R)
    a = 0.5 * np.ones(R)
    sigma2 = 0.5 * np.ones(R)

    F = 0.1 * rng.randn(T, R)

    beta = np.zeros(1 + 3 * R + 1)  # [β0, β1(R), β2(R), β3(R), β4]
    eta2 = 1.0

    kept = 0

    F_sum = np.zeros_like(F)
    F_sumsq = np.zeros_like(F)

    mu_sum = np.zeros_like(mu)

    Theta_sum = np.zeros_like(Theta)
    Theta_sumsq = np.zeros_like(Theta)

    z_sum = np.zeros(R)
    beta_sum = np.zeros_like(beta)

    a_sum = np.zeros_like(a)
    sigma2_sum = np.zeros_like(sigma2)
    eta2_sum = 0.0

    Psi = np.eye(n)
    Psi_sum = np.zeros_like(Psi)

    psi_accepts = 0
    psi_trials = 0

    # SV state
    omega = np.zeros((T, n))
    omega_sum = np.zeros_like(omega)
    omega_sumsq = np.zeros_like(omega)

    tau2 = 0.1
    tau2_sum = 0.0

    # factor selection draws
    z_draws_list = []
    r_draws_list = []

    # ---------------------------------------------------------
    # MULTI-TARGET PANEL DEFINITIONS (constant within this run)
    # ---------------------------------------------------------
    asof_month = pd.Timestamp(D.dates_m[-1])
    asof_q = asof_month.to_period("Q")

    if len(D.dates_q) == 0 or len(D.y_q) == 0:
        raise ValueError("D.dates_q / D.y_q is empty; cannot define last released GDP quarter.")

    dates_q = pd.to_datetime(np.asarray(D.dates_q))
    y_q = np.asarray(D.y_q, dtype=float)

    # Released quarters = those with finite GDP
    released_mask = np.isfinite(y_q)
    if not released_mask.any():
        raise ValueError("No released GDP in this vintage.")

    last_obs_idx = np.where(released_mask)[0][-1]
    last_released_q = dates_q[last_obs_idx].to_period("Q")

    # Targets = quarters after last released up to asof_q
    targets: List[pd.Period] = []
    q_iter = last_released_q + 1
    while q_iter <= asof_q:
        targets.append(q_iter)
        q_iter += 1

    if len(targets) == 0:
        # This means GDP is already released up to asof quarter; nothing to nowcast.
        # We keep behavior explicit rather than silently returning nonsense.
        raise ValueError(
            f"No active nowcast targets: last_released_q={last_released_q}, asof_q={asof_q}."
        )

    nowcast_targets = [str(q) for q in targets]
    J = len(targets)

    # ragged-edge / empty x_T adjustment is also constant per run
    x_T_empty = bool(np.all(np.isnan(D.X_m[-1, :])))
    mp_asof = int(D.month_pos_in_quarter[-1])

    # ---------------------------------------------------------
    # IMPORTANT FIX: anchor bridge months to the TARGET quarter
    # ---------------------------------------------------------
    # nowcast_draw_next_quarter() always uses the LAST rows of the provided F
    # (e.g., F[t], F[t-1], F[t-2] when month_pos_in_quarter_last==3).
    #
    # Therefore, when nowcasting a completed earlier quarter (qtar < asof_q),
    # we MUST truncate F so its last row corresponds to the QUARTER-END MONTH
    # of qtar (e.g., March for Q1), so the bridge uses (Mar, Feb, Jan),
    # not (Apr, Mar, Feb) in an April vintage.
    dm = pd.to_datetime(np.asarray(D.dates_m)).normalize().to_period("M").to_timestamp()  # month starts
    m_to_idx = {pd.Timestamp(m): i for i, m in enumerate(dm)}

    target_tidx: List[int] = []
    target_mp: List[int] = []
    for qtar in targets:
        if qtar < asof_q:
            # quarter-end month for qtar, converted to month-start timestamp (e.g., 2016-03-01)
            m_end = qtar.asfreq("M", "end").to_timestamp().normalize().replace(day=1)
            if m_end not in m_to_idx:
                raise ValueError(
                    f"Quarter-end month {m_end.date()} for target {qtar} not found in D.dates_m. "
                    f"(Check date conventions: D.dates_m uses month starts.)"
                )
            target_tidx.append(int(m_to_idx[m_end]))
            target_mp.append(3)
        else:
            # current as-of quarter: anchor at the as-of month
            target_tidx.append(int(len(dm) - 1))
            target_mp.append(int(mp_asof))

    # store per-kept-draw vectors -> matrix (S,J)
    nowcast_draws_matrix_list: List[np.ndarray] = []

    # -----------------
    # MCMC loop
    # -----------------
    for it in range(n_iter):
        # 1) F | rest via FFBS
        F = ffbs_factors_XY(
            X=X,
            y_q=D.y_q,
            quarter_of_month=D.quarter_of_month,
            month_pos_in_quarter=D.month_pos_in_quarter,
            mu=mu,
            Theta=Theta,
            z=z,
            omega=omega,
            Psi=Psi,
            a=a,
            sigma2=sigma2,
            beta=beta,
            eta2=eta2,
            bridge=bridge,
            rng=rng,
        )

        # 2) mu, Theta
        mu, Theta = sample_mu_theta_correlated(X, F, z, omega, Psi, rng=rng)

        def ll_Psi(Psi_try):
            return _loglik_X_given_F_correlated(X, mu, Theta, z, F, omega, Psi_try)

        Psi, acc, tr = sample_Psi_LKJ1_columnwise_rw(Psi, ll_Psi, n_sweeps=1, step=0.05, rng=rng)
        psi_accepts += acc
        psi_trials += tr

        if use_sv:
            E = compute_monthly_residuals(X, mu, Theta, z, F)

            omega = sample_omega_pgbs(
                E=E,
                Psi=Psi,
                tau2=tau2,
                omega_ref=omega,
                P=h.n_particles,
                omega1_mean=h.omega1_mean,
                omega1_var=h.omega1_var,
                rng=rng
            )
            tau2 = sample_tau2(omega, alpha_l=h.alpha_l, beta_l=h.beta_l, rng=rng)
        else:
            omega.fill(0.0)

        # 3) factor AR params a, sigma2
        a, sigma2 = sample_a_sigma2(F, alpha_s=h.alpha_s, beta_s=h.beta_s, rng=rng)

        # 4) GDP regression beta, eta2
        beta, eta2 = sample_beta_eta2(
            D.y_q, F, z,
            D.quarter_of_month, D.month_pos_in_quarter,
            eta2_current=eta2,
            alpha_h=h.alpha_h, beta_h=h.beta_h, rng=rng
        )

        # 5) Identification / ordering
        F, Theta, a, sigma2, z, beta, p, s = enforce_ident_order(F, Theta, a, sigma2, z, beta, p=p, s=s)
        z[0] = 1
        p[0] = 1.0

        # 6) Selection update (z,p,pi)
        if do_selection:
            loglik0 = np.zeros(R)
            loglik1 = np.zeros(R)

            for j in range(1, R):
                z0 = z.copy(); z0[j] = 0
                z1 = z.copy(); z1[j] = 1

                ll0 = kalman_loglik_XY(
                    X, D.y_q, D.quarter_of_month, D.month_pos_in_quarter,
                    mu, Theta, z0, omega, Psi, a, sigma2,
                    beta, eta2, bridge=bridge
                )
                ll1 = kalman_loglik_XY(
                    X, D.y_q, D.quarter_of_month, D.month_pos_in_quarter,
                    mu, Theta, z1, omega, Psi, a, sigma2,
                    beta, eta2, bridge=bridge
                )
                loglik0[j] = ll0
                loglik1[j] = ll1

            z, p, s, pi = sample_z_p_pi(
                z=z, p=p, s=s, pi=pi,
                loglik0=loglik0, loglik1=loglik1,
                alpha_p=h.alpha_p, beta_p=h.beta_p,
                alpha_pi=h.alpha_pi, beta_pi=h.beta_pi,
                rng=rng
            )
        else:
            z[:] = 1
            z[0] = 1
            p[:] = 1.0
            pi = 1.0

        if (it + 1) % 200 == 0:
            print("post-selection z:", z.astype(int))

        # -----------------
        # store
        # -----------------
        if it >= burn and ((it - burn) % thin == 0):
            kept += 1

            F_sum += F
            F_sumsq += F * F

            mu_sum += mu

            Theta_sum += Theta
            Theta_sumsq += Theta * Theta

            z_sum += z
            beta_sum += beta

            a_sum += a
            sigma2_sum += sigma2
            eta2_sum += float(eta2)

            Psi_sum += Psi

            omega_sum += omega
            omega_sumsq += omega * omega
            tau2_sum += float(tau2)

            z_draws_list.append(z.astype(np.int8).copy())
            r_draws_list.append(int(np.sum(z)))

            # -----------------------------
            # MULTI-TARGET SEQUENTIAL NOWCAST
            # -----------------------------
            F_for_nowcast = substitute_FT_when_xT_empty(F, a=a, x_T_empty=x_T_empty)

            y_prev = float(y_q[last_obs_idx])
            y_targets = np.zeros(J, dtype=float)

            for jj, _qtar in enumerate(targets):
                mp = int(target_mp[jj])
                tidx = int(target_tidx[jj])

                # Truncate factor path so the bridge is anchored at the correct month
                # for THIS target (quarter-end month for completed quarters).
                F_target = F_for_nowcast[: (tidx + 1), :]

                y_draw = nowcast_draw_next_quarter(
                    beta=beta,
                    z=z,
                    a=a,
                    sigma2=sigma2,
                    eta2=eta2,
                    F=F_target,
                    y_last=y_prev,
                    month_pos_in_quarter_last=mp,
                    rng=rng
                )
                y_targets[jj] = y_draw
                y_prev = y_draw

            nowcast_draws_matrix_list.append(y_targets)

    # -----------------
    # finalize posterior summaries
    # -----------------
    denom = max(kept, 1)

    F_mean = F_sum / denom
    F_var = np.maximum(F_sumsq / denom - F_mean * F_mean, 0.0)

    Theta_mean = Theta_sum / denom
    Theta_var = np.maximum(Theta_sumsq / denom - Theta_mean * Theta_mean, 0.0)

    omega_mean = omega_sum / denom
    omega_var = np.maximum(omega_sumsq / denom - omega_mean * omega_mean, 0.0)

    psi_accept_rate = float(psi_accepts / psi_trials) if psi_trials > 0 else float("nan")

    # nowcast matrix + summaries
    nowcast_draws_matrix = np.asarray(nowcast_draws_matrix_list, dtype=float)  # (S,J)
    if nowcast_draws_matrix.ndim != 2 or nowcast_draws_matrix.shape[1] != J:
        raise RuntimeError(
            f"nowcast_draws_matrix has unexpected shape {nowcast_draws_matrix.shape}, expected (S,{J})."
        )

    nowcast_means_vector = nowcast_draws_matrix.mean(axis=0)
    nowcast_sds_vector = nowcast_draws_matrix.std(axis=0, ddof=0)

    # backward-compatible current-quarter = last target
    nowcast_mean = float(nowcast_means_vector[-1])
    nowcast_sd = float(nowcast_sds_vector[-1])
    nowcast_draws = nowcast_draws_matrix[:, -1].copy()

    return BayResult(
        draws_kept=int(kept),

        F_mean=F_mean,
        F_sd=np.sqrt(F_var),

        mu_mean=mu_sum / denom,

        Theta_mean=Theta_mean,
        Theta_sd=np.sqrt(Theta_var),

        z_mean=z_sum / denom,
        z_draws=np.asarray(z_draws_list, dtype=np.int8),
        r_draws=np.asarray(r_draws_list, dtype=np.int16),

        beta_mean=beta_sum / denom,
        a_mean=a_sum / denom,
        sigma2_mean=sigma2_sum / denom,
        eta2_mean=float(eta2_sum / denom),

        Psi_mean=Psi_sum / denom,

        omega_mean=omega_mean,
        omega_sd=np.sqrt(omega_var),
        tau2_mean=float(tau2_sum / denom),

        nowcast_targets=nowcast_targets,
        nowcast_draws_matrix=nowcast_draws_matrix,
        nowcast_means_vector=nowcast_means_vector,
        nowcast_sds_vector=nowcast_sds_vector,

        nowcast_mean=nowcast_mean,
        nowcast_sd=nowcast_sd,
        nowcast_draws=nowcast_draws,

        standardize_mean=mX.astype(float),
        standardize_sd=sX.astype(float),

        diagnostics=dict(
            psi_accept_rate=psi_accept_rate,
            psi_accepts=int(psi_accepts),
            psi_trials=int(psi_trials),
        ),
        config=dict(
            R_max=R_max, n_iter=n_iter, burn=burn, thin=thin, standardize=standardize,
            paper_defaults=paper_defaults,
            use_sv=use_sv,
            hyperparams=h.__dict__,
        ),
    )


def _loglik_gdp(y, F, z, beta, eta2):
    y = np.asarray(y, float)
    K = len(y)
    R = F.shape[1]
    Z = z.astype(float)

    b0 = beta[0]
    b1 = beta[1:1+R]
    b2 = beta[1+R:1+2*R]
    b3 = beta[1+2*R:1+3*R]
    b4 = beta[-1]

    ll = 0.0
    c = -0.5 * np.log(2*np.pi*eta2)

    for k in range(1, K):
        t3 = 3*(k+1) - 1
        mean = (b0
                + b1 @ (Z * F[t3,:])
                + b2 @ (Z * F[t3-1,:])
                + b3 @ (Z * F[t3-2,:])
                + b4 * y[k-1])
        e = y[k] - mean
        ll += c - 0.5*(e*e)/eta2
    return float(ll)


import numpy as np

def _loglik_X_given_F_correlated(X, mu, Theta, z, F, omega, Psi):
    X = np.asarray(X, float)
    F = np.asarray(F, float)
    omega = np.asarray(omega, float)
    Psi = np.asarray(Psi, float)

    T, n = X.shape
    Z = z.astype(float)

    ll = 0.0
    const2pi = float(np.log(2.0 * np.pi))

    for t in range(T):
        obs = ~np.isnan(X[t])
        m = int(obs.sum())
        if m == 0:
            continue

        mean = mu[obs] + (Theta[obs, :] * Z[None, :]) @ F[t, :]
        e = X[t, obs] - mean

        omega_o_t = omega[t, obs]
        inv_scale = np.exp(-omega_o_t)
        e_scaled = inv_scale * e

        Psi_o = Psi[np.ix_(obs, obs)]
        Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(m)
        L = np.linalg.cholesky(Psi_o)

        y = np.linalg.solve(L, e_scaled)
        quad = float(y @ y)

        logdet_Psi = 2.0 * float(np.sum(np.log(np.diag(L))))
        logdet = logdet_Psi + 2.0 * float(np.sum(omega_o_t))

        ll += -0.5 * (m * const2pi + logdet + quad)

    return float(ll)


def sample_mu_theta_correlated(X: np.ndarray, F: np.ndarray, z: np.ndarray, omega: np.ndarray, Psi: np.ndarray, rng=np.random):
    X = np.asarray(X, float)
    T, n = X.shape
    R = F.shape[1]
    ZF = F * z.astype(float)[None, :]

    K = n + n * R
    I_K = np.eye(K)

    Ds = []
    ys = []

    for t in range(T):
        obs = ~np.isnan(X[t])
        m = int(obs.sum())
        if m == 0:
            continue

        obs_idx = np.where(obs)[0]
        x_o = X[t, obs_idx]

        Psi_o = Psi[np.ix_(obs_idx, obs_idx)]
        Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(m)
        L = np.linalg.cholesky(Psi_o)

        inv_scale = np.exp(-omega[t, obs_idx])
        x_scaled = inv_scale * x_o

        y_t = np.linalg.solve(L, x_scaled)

        D_t = np.zeros((m, K), dtype=float)

        for rr, i in enumerate(obs_idx):
            D_t[rr, i] = 1.0

        zft = ZF[t, :]
        base_theta = n
        for rr, i in enumerate(obs_idx):
            start = base_theta + i * R
            D_t[rr, start:start + R] = zft

        D_t *= inv_scale[:, None]
        D_t = np.linalg.solve(L, D_t)

        Ds.append(D_t)
        ys.append(y_t)

    if len(Ds) == 0:
        mu = rng.randn(n)
        Theta = rng.randn(n, R)
        return mu, Theta

    D_big = np.vstack(Ds)
    y_big = np.concatenate(ys)

    Pn = I_K + D_big.T @ D_big
    Vn = np.linalg.solve(Pn, I_K)
    mn = Vn @ (D_big.T @ y_big)

    Lp = np.linalg.cholesky(Vn + 1e-12 * np.eye(K))
    b = mn + Lp @ rng.randn(K)

    mu = b[:n]
    Theta = b[n:].reshape(n, R)
    return mu, Theta


def logml_monthly_collapsed(X, F, z, Psi):
    X = np.asarray(X, float)
    T, n = X.shape
    R = F.shape[1]
    ZF = F * z.astype(float)[None, :]

    K = n + n * R
    Dy = np.zeros(K)
    DD = np.zeros((K, K))
    yy = 0.0
    m_total = 0

    for t in range(T):
        obs = ~np.isnan(X[t])
        m = int(obs.sum())
        if m == 0:
            continue

        obs_idx = np.where(obs)[0]
        x_o = X[t, obs_idx]

        Psi_o = Psi[np.ix_(obs_idx, obs_idx)]
        Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(m)
        L = np.linalg.cholesky(Psi_o)

        y_t = np.linalg.solve(L, x_o)
        yy += float(y_t @ y_t)
        m_total += m

        D_t = np.zeros((m, K), dtype=float)

        for rr, i in enumerate(obs_idx):
            D_t[rr, i] = 1.0

        zft = ZF[t, :]
        base = n
        for rr, i in enumerate(obs_idx):
            start = base + i * R
            D_t[rr, start:start+R] = zft

        D_t = np.linalg.solve(L, D_t)

        Dy += D_t.T @ y_t
        DD += D_t.T @ D_t

    P = np.eye(K) + DD
    cholP = np.linalg.cholesky(P + 1e-12 * np.eye(K))

    v = np.linalg.solve(cholP, Dy)
    quad = float(v @ v)
    logdetP = 2.0 * float(np.sum(np.log(np.diag(cholP))))

    const = -0.5 * m_total * np.log(2.0 * np.pi)
    return const - 0.5 * (yy - quad + logdetP)


def logml_gdp_collapsed(y, F, z, alpha_h=2.0, beta_h=1e-4):
    y = np.asarray(y, float)
    K = len(y)
    R = F.shape[1]
    Z = z.astype(float)

    rows = []
    targ = []

    for k in range(1, K):
        t3 = 3 * (k + 1) - 1
        xrow = np.r_[
            1.0,
            Z * F[t3, :],
            Z * F[t3 - 1, :],
            Z * F[t3 - 2, :],
            y[k - 1],
        ]
        rows.append(xrow)
        targ.append(y[k])

    Xreg = np.vstack(rows)
    yreg = np.asarray(targ)

    n = len(yreg)
    p = Xreg.shape[1]

    V0_inv = np.eye(p)

    XtX = Xreg.T @ Xreg
    Xty = Xreg.T @ yreg

    Vn_inv = V0_inv + XtX
    Vn = np.linalg.inv(Vn_inv)
    mn = Vn @ Xty

    quad = yreg @ yreg - mn @ Vn_inv @ mn

    alpha_post = alpha_h + 0.5 * n
    beta_post = beta_h + 0.5 * quad

    sign0, logdetV0 = np.linalg.slogdet(np.eye(p))
    signn, logdetVn = np.linalg.slogdet(Vn)

    logml = (
        -0.5 * n * np.log(2 * np.pi)
        + 0.5 * (logdetVn - logdetV0)
        + alpha_h * np.log(beta_h)
        - alpha_post * np.log(beta_post)
        + math.lgamma(alpha_post)
        - math.lgamma(alpha_h)
    )

    return float(logml)