# sv_pgbs.py
import numpy as np


def compute_monthly_residuals(X: np.ndarray, mu: np.ndarray, Theta: np.ndarray,
                              z: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Residuals E[t,i] = X[t,i] - mu[i] - Theta[i,:]*(z) @ F[t,:]
    Preserves NaNs where X has NaNs.
    Shapes:
      X: (T,n), mu: (n,), Theta: (n,K), z: (K,), F: (T,K)
    Returns:
      E: (T,n) with NaNs matching X
    """
    X = np.asarray(X, float)
    mu = np.asarray(mu, float)
    Theta = np.asarray(Theta, float)
    z = np.asarray(z).astype(float)
    F = np.asarray(F, float)

    T, n = X.shape
    # mean_all: (T,n)
    mean_all = mu[None, :] + F @ ((Theta * z[None, :]).T)
    E = X - mean_all
    # ensure NaNs propagate
    E[np.isnan(X)] = np.nan
    return E


def sample_tau2(omega: np.ndarray, alpha_l: float, beta_l: float, rng=np.random) -> float:
    """
    tau2 ~ IG(alpha_l, beta_l) under RW omega_t = omega_{t-1} + e_t, e_t~N(0, tau2 I)
    Parameterization: IG(shape=alpha, scale=beta)
    """
    omega = np.asarray(omega, float)
    d = np.diff(omega, axis=0)  # (T-1,n)
    ss = float(np.sum(d * d))
    Tm1, n = d.shape

    alpha_post = alpha_l + 0.5 * n * Tm1
    beta_post  = beta_l  + 0.5 * ss

    g = rng.gamma(shape=alpha_post, scale=1.0 / beta_post)
    return float(1.0 / g)


def _systematic_resample(weights: np.ndarray, rng=np.random) -> np.ndarray:
    """
    Systematic resampling. weights must sum to 1.
    Returns ancestor indices of length P.
    """
    P = len(weights)
    positions = (rng.random() + np.arange(P)) / P
    cumsum = np.cumsum(weights)
    idx = np.zeros(P, dtype=int)
    i = j = 0
    while i < P:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def _get_psio_cache(Psi: np.ndarray, obs_idx: np.ndarray, cache: dict):
    """
    Cache Cholesky and logdet(Psi_o) by observed index pattern.
    """
    key = tuple(obs_idx.tolist())
    if key in cache:
        return cache[key]

    Psi_o = Psi[np.ix_(obs_idx, obs_idx)]
    m = Psi_o.shape[0]
    Psi_o = 0.5 * (Psi_o + Psi_o.T) + 1e-10 * np.eye(m)
    L = np.linalg.cholesky(Psi_o)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    cache[key] = (L, logdet)
    return L, logdet


def _log_weight_t(e_t: np.ndarray, omega_t: np.ndarray, Psi: np.ndarray,
                  cache: dict) -> float:
    """
    Log p(e_t | omega_t, Psi) up to constant (but we keep full Gaussian log-lik).
    Missing entries in e_t are ignored (based on NaNs).
    Uses: Cov = D Psi D, D=diag(exp(omega_t))
    """
    obs = ~np.isnan(e_t)
    m = int(obs.sum())
    if m == 0:
        return 0.0  # no information at this time

    obs_idx = np.where(obs)[0]
    e_o = e_t[obs_idx]
    omega_o = omega_t[obs_idx]

    # scale residual: e_scaled = D^{-1} e
    e_scaled = np.exp(-omega_o) * e_o

    L, logdet_Psi = _get_psio_cache(Psi, obs_idx, cache)
    y = np.linalg.solve(L, e_scaled)
    quad = float(y @ y)

    # log|D Psi D| = log|Psi| + 2*sum(omega_o)
    logdet = logdet_Psi + 2.0 * float(np.sum(omega_o))

    const2pi = float(np.log(2.0 * np.pi))
    return -0.5 * (m * const2pi + logdet + quad)


def _log_transition_density(omega_next: np.ndarray, omega_curr: np.ndarray, tau2: float) -> float:
    """
    RW transition: omega_next | omega_curr ~ N(omega_curr, tau2 I)
    """
    n = omega_curr.size
    diff = omega_next - omega_curr
    quad = float(diff @ diff) / tau2
    logdet = n * float(np.log(tau2))
    const2pi = float(np.log(2.0 * np.pi))
    return -0.5 * (n * const2pi + logdet + quad)


def sample_omega_pgbs(E: np.ndarray, Psi: np.ndarray, tau2: float,
                      omega_ref: np.ndarray,
                      P: int = 10,
                      omega1_mean: float = 0.0,
                      omega1_var: float = 10.0,
                      rng=np.random) -> np.ndarray:
    """
    Particle Gibbs with Backward Simulation (PGBS) for omega (T,n).
    - Conditional SMC with reference trajectory omega_ref.
    - Backward simulation smoothing using RW transition density.

    Inputs:
      E: (T,n) residuals with NaNs
      Psi: (n,n) correlation matrix
      tau2: scalar RW innovation variance
      omega_ref: (T,n) current omega path (reference)
      P: number of particles (start with 5 or 10)
      omega1_mean, omega1_var: prior for omega[0,i] ~ N(mean, var)

    Returns:
      omega_new: (T,n)
    """
    E = np.asarray(E, float)
    Psi = np.asarray(Psi, float)
    omega_ref = np.asarray(omega_ref, float)

    T, n = E.shape
    assert omega_ref.shape == (T, n)
    assert P >= 2
    assert tau2 > 0.0

    cache = {}

    # storage
    particles = np.zeros((T, P, n), float)
    ancestors = np.zeros((T, P), dtype=int)
    logw_hist = np.zeros((T, P), float)

    # --- t=0 initialize
    # sample P-1 particles from prior
    particles[0, :P-1, :] = rng.normal(loc=omega1_mean, scale=np.sqrt(omega1_var), size=(P-1, n))
    # reference particle
    particles[0, P-1, :] = omega_ref[0]
    ancestors[0, :] = np.arange(P)

    # weights at t=0
    for i in range(P):
        logw_hist[0, i] = _log_weight_t(E[0], particles[0, i], Psi, cache)

    # normalize for resampling
    lw = logw_hist[0].copy()
    lw -= _logsumexp(lw)
    w = np.exp(lw)

    # --- t=1..T-1
    for t in range(1, T):
        # resample ancestors for non-reference particles
        anc = _systematic_resample(w, rng=rng)
        # propagate non-reference
        for i in range(P-1):
            a_i = anc[i]
            ancestors[t, i] = a_i
            particles[t, i, :] = particles[t-1, a_i, :] + rng.normal(0.0, np.sqrt(tau2), size=n)

        # reference particle stays fixed on reference path, with fixed ancestor = reference index
        particles[t, P-1, :] = omega_ref[t]

        logp = np.empty(P, float)
        for j in range(P):
            logp[j] = np.log(w[j] + 1e-300) + _log_transition_density(
                omega_ref[t], particles[t-1, j, :], tau2
            )
        logp -= _logsumexp(logp)
        p = np.exp(logp)

        ancestors[t, P-1] = int(rng.choice(P, p=p))

        # compute weights
        for i in range(P):
            logw_hist[t, i] = _log_weight_t(E[t], particles[t, i], Psi, cache)

        lw = logw_hist[t].copy()
        lw -= _logsumexp(lw)
        w = np.exp(lw)

    # --- Backward simulation
    omega_new = np.zeros((T, n), float)

    # sample terminal index
    lwT = logw_hist[T-1].copy()
    lwT -= _logsumexp(lwT)
    wT = np.exp(lwT)
    k = int(rng.choice(P, p=wT))
    omega_new[T-1] = particles[T-1, k]

    # backward recursion
    for t in range(T-2, -1, -1):
        omega_next = omega_new[t+1]

        # smoothing weights proportional to w_t(i) * p(omega_{t+1} | omega_t(i))
        logp = np.empty(P, float)
        for i in range(P):
            logp[i] = logw_hist[t, i] + _log_transition_density(omega_next, particles[t, i], tau2)

        logp -= _logsumexp(logp)
        p = np.exp(logp)
        k = int(rng.choice(P, p=p))
        omega_new[t] = particles[t, k]

    return omega_new
