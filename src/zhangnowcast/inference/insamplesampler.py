from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import numpy.linalg as npl
import math

from zhangnowcast.data.data import ZhangData
from zhangnowcast.inference.kalman_ffbs import ffbs_factors, kalman_loglik_X, kalman_loglik_XY, ffbs_factors_XY
from zhangnowcast.model.gdp_block import sample_beta_eta2
from zhangnowcast.model.factor_params import sample_a_sigma2, enforce_ident_order
from zhangnowcast.model.selection import sample_z_p_pi
from zhangnowcast.model.lkj_psi import sample_Psi_LKJ1_columnwise_rw
from zhangnowcast.inference.sv_pgbs import compute_monthly_residuals, sample_omega_pgbs, sample_tau2
from zhangnowcast.nowcast.nowcast import nowcast_draw_next_quarter, substitute_FT_when_xT_empty
from zhangnowcast.model.bridging import GDPBridgeEq5
bridge = GDPBridgeEq5()

@dataclass
class BayResult:
    draws_kept: int
    F_mean: np.ndarray           # (T, R)
    F_sd: np.ndarray             # (T, R)
    mu_mean: np.ndarray          # (n,)
    Theta_mean: np.ndarray       # (n, R)   (loadings Lambda)
    Theta_sd: np.ndarray         # (n, R)
    z_mean: np.ndarray           # (R,)
    z_draws: np.ndarray          # (S, R) int8
    r_draws: np.ndarray          # (S,)
    beta_mean: np.ndarray        # (1 + 3R + 1,)
    a_mean: np.ndarray           # (R,)
    sigma2_mean: np.ndarray      # (R,)
    eta2_mean: float             # scalar
    Psi_mean: np.ndarray         # (n, n)
    omega_mean: np.ndarray       # (T, n)
    omega_sd: np.ndarray         # (T, n)
    tau2_mean: float             # scalar
    nowcast_mean: float
    nowcast_sd: float
    nowcast_draws: np.ndarray    # (S,)
    standardize_mean: np.ndarray # (n,)
    standardize_sd: np.ndarray   # (n,)
    diagnostics: Dict[str, Any]
    config: Dict[str, Any]

@dataclass
class BayHyperParams:
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
    if paper_defaults:
        return BayHyperParams(
            alpha_s=2.0,
            beta_s=float(R + 2),
            alpha_h=2.0,
            beta_h=1e-4,
            omega1_mean=0.0,
            omega1_var=1.0,
            alpha_l=2.0,
            beta_l=0.2,
            n_particles=20,
        )
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


def _loglik_X_given_F_correlated(X, mu, Theta, z, F, omega, Psi):
    """
    Log-likelihood of monthly panel under eps_t ~ N(0, D_t Psi D_t) with missing data,
    where D_t = diag(exp(omega[t,:])).

    X: (T,n) with NaNs for missing
    mu: (n,)
    Theta: (n,K)
    z: (K,) {0,1}
    F: (T,K)
    omega: (T,n)
    Psi: (n,n) correlation matrix
    """
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

        # mean for observed series
        mean = mu[obs] + (Theta[obs, :] * Z[None, :]) @ F[t, :]
        e = X[t, obs] - mean  # (m,)

        # SV scaling: e_scaled = D^{-1} e, with D = diag(exp(omega))
        omega_o_t = omega[t, obs]          # (m,)
        inv_scale = np.exp(-omega_o_t)     # (m,)
        e_scaled = inv_scale * e           # (m,)

        # Work with Psi_o (correlation), not R_o
        Psi_o = Psi[np.ix_(obs, obs)]
        Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(m)
        L = np.linalg.cholesky(Psi_o)

        # quad form under Psi_o^{-1}
        y = np.linalg.solve(L, e_scaled)
        quad = float(y @ y)

        # log |D Psi D| = log|Psi| + 2*sum(omega)
        logdet_Psi = 2.0 * float(np.sum(np.log(np.diag(L))))
        logdet = logdet_Psi + 2.0 * float(np.sum(omega_o_t))

        ll += -0.5 * (m * const2pi + logdet + quad)

    return float(ll)



def sample_mu_theta_correlated(X: np.ndarray, F: np.ndarray, z: np.ndarray, omega: np.ndarray, Psi: np.ndarray, rng=np.random):
    """
    Correct (mu, Theta) draw under:
      x_t ~ N(mu + Theta (z ⊙ F_t), Psi)
    with missing data.

    Parameter vector b = [mu (n), vec(Theta) (n*R)] length K = n + nR.
    Uses per-time whitening with chol(Psi_obs) and stacks into a single Gaussian regression.

    Prior: b ~ N(0, I_K)  (same scale as your current BAY step; can be configured later)
    """
    X = np.asarray(X, float)
    T, n = X.shape
    R = F.shape[1]
    ZF = F * z.astype(float)[None, :]  # (T,R)

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

        
        inv_scale = np.exp(-omega[t, obs_idx])     # (m,)
        x_scaled = inv_scale * x_o                # elementwise

        # whiten y
        y_t = np.linalg.solve(L, x_scaled)

        # Build fixed-width design D_t: (m, K)
        D_t = np.zeros((m, K), dtype=float)

        # mu part: for each observed series i, that equation uses mu_i
        # => place 1 in column i
        for rr, i in enumerate(obs_idx):
            D_t[rr, i] = 1.0

        # theta part: for each observed series i, that equation uses Theta[i,:] @ ZF_t
        zft = ZF[t, :]  # (R,)
        base_theta = n  # start index of Theta vector in b
        for rr, i in enumerate(obs_idx):
            start = base_theta + i * R
            D_t[rr, start:start + R] = zft

        # scale rows of design the same way, then whiten
        D_t *= inv_scale[:, None]
        D_t = np.linalg.solve(L, D_t)

        Ds.append(D_t)
        ys.append(y_t)

    if len(Ds) == 0:
        # No data at all (shouldn't happen)
        mu = rng.randn(n)
        Theta = rng.randn(n, R)
        return mu, Theta

    D_big = np.vstack(Ds)          # (sum m_t, K)
    y_big = np.concatenate(ys)     # (sum m_t,)

    # Posterior b | ... ~ N(mn, Vn)
    Pn = I_K + D_big.T @ D_big
    # use solve for stability
    Vn = np.linalg.solve(Pn, I_K)
    mn = Vn @ (D_big.T @ y_big)

    Lp = np.linalg.cholesky(Vn + 1e-12 * np.eye(K))
    b = mn + Lp @ rng.randn(K)

    mu = b[:n]
    Theta = b[n:].reshape(n, R)
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

    # Standardize the entire panel to match priors scaling
    if standardize:
        X, mX, sX = _standardize_panel(X)
    else:
        mX = np.zeros(X.shape[1])
        sX = np.ones(X.shape[1])

    T, n = X.shape
    R = R_max

    # Hyperparameters for model fitting
    h = resolve_bay_hyperparams(R=R, n=n, paper_defaults=paper_defaults)
    alpha_l = h.alpha_l
    beta_l = h.beta_l
    omega1_mean = h.omega1_mean
    omega1_var = h.omega1_var
    n_particles = h.n_particles

    # Initialize parameters for the entire dataset
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

    # Initialize factors
    F = 0.1 * rng.randn(T, R)

    beta = np.zeros(1 + 3 * R + 1)  # [β0, β1(R), β2(R), β3(R), β4]
    eta2 = 1.0

    # Accumulators for results
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
    now_sum = 0.0
    now_sumsq = 0.0
    now_draws = []
    omega = np.zeros((T, n))
    omega_sum = np.zeros_like(omega)
    omega_sumsq = np.zeros_like(omega)
    tau2 = 0.1  # starting value
    tau2_sum = 0.0
    z_draws_list = []
    r_draws_list = []

    # MCMC loop for in-sample analysis
    for it in range(n_iter):
        # Step 1: Estimate the factors (F) using FFBS
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

        # Step 2: Sample mu and Theta (parameters of the factors)
        mu, Theta = sample_mu_theta_correlated(X, F, z, omega, Psi, rng=rng)

        # Step 3: Sample Psi
        def ll_Psi(Psi_try):
            return _loglik_X_given_F_correlated(X, mu, Theta, z, F, omega, Psi_try)

        Psi, acc, tr = sample_Psi_LKJ1_columnwise_rw(Psi, ll_Psi, n_sweeps=1, step=0.05, rng=rng)
        psi_accepts += acc
        psi_trials += tr

        # Step 4: Stochastic Volatility (if applicable)
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
            omega.fill(0.0)  # No stochastic volatility (constant)

        # Step 5: Sample factor AR parameters (a and sigma2)
        a, sigma2 = sample_a_sigma2(F, alpha_s=h.alpha_s, beta_s=h.beta_s, rng=rng)

        # Step 6: Sample GDP regression parameters (beta and eta2)
        beta, eta2 = sample_beta_eta2(
            D.y_q, F, z,
            D.quarter_of_month, D.month_pos_in_quarter,
            eta2_current=eta2,
            alpha_h=h.alpha_h, beta_h=h.beta_h, rng=rng
        )

        # Step 7: Enforce identification to control label switching
        F, Theta, a, sigma2, z, beta, p, s = enforce_ident_order(F, Theta, a, sigma2, z, beta, p=p, s=s)
        z[0] = 1
        p[0] = 1.0

        # Step 8: Update factor selection (z, p, pi)
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
            z[:] = 1  # No factor selection, keep all factors
            z[0] = 1
            p[:] = 1.0
            pi = 1.0

        # Step 9: Store the results
        if (it + 1) % 200 == 0:
            print("post-selection z:", z.astype(int))

        # Store after burn-in and thinning
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

            y_last = float(D.y_q[-1])
            mp_last = int(D.month_pos_in_quarter[-1])

            # Substitute missing factors and do the nowcasting
            x_T_empty = bool(np.all(np.isnan(D.X_m[-1, :])))
            F_for_nowcast = substitute_FT_when_xT_empty(F, a=a, x_T_empty=x_T_empty)

            y_hat = nowcast_draw_next_quarter(
                beta=beta, z=z, a=a, sigma2=sigma2, eta2=eta2,
                F=F_for_nowcast, y_last=y_last, month_pos_in_quarter_last=mp_last,
                rng=rng
            )

            now_draws.append(y_hat)
            now_sum += y_hat
            now_sumsq += y_hat * y_hat

    # Finalize posterior summaries
    denom = max(kept, 1)

    now_mean = now_sum / denom
    now_var = max(now_sumsq / denom - now_mean**2, 0.0)
    now_sd = float(np.sqrt(now_var))

    F_mean = F_sum / denom
    F_var = np.maximum(F_sumsq / denom - F_mean * F_mean, 0.0)

    Theta_mean = Theta_sum / denom
    Theta_var = np.maximum(Theta_sumsq / denom - Theta_mean * Theta_mean, 0.0)

    omega_mean = omega_sum / denom
    omega_var = np.maximum(omega_sumsq / denom - omega_mean * omega_mean, 0.0)

    psi_accept_rate = float(psi_accepts / psi_trials) if psi_trials > 0 else float("nan")

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

        nowcast_mean=float(now_mean),
        nowcast_sd=float(now_sd),
        nowcast_draws=np.asarray(now_draws, dtype=float),

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
