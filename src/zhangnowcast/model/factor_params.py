import numpy as np


def sample_a_sigma2(F: np.ndarray, alpha_s: float, beta_s: float, rng=np.random):
    """
    What this code does:
    --------------------
    Updates the AR(1) parameters for each latent factor:
        a_j   (persistence)
        sigma2_j (innovation variance)

    Why we need it (DFM context):
    -----------------------------
    In Zhang's DFM, the monthly latent factors evolve over time, and those evolving
    factors are what later feed into the GDP bridge equation (Eq. (5)).
    So we must sample the factor dynamics parameters inside the MCMC.

    Where this comes from in Zhang:
    -------------------------------
    • Factor state equation: Zhang Eq. (4)
        f_{j,t} = a_j f_{j,t-1} + sigma_j u_{j,t},  u_{j,t} ~ N(0,1)
      which is equivalent to:  f_t = a f_{t-1} + e_t,  e_t ~ N(0, sigma2)

    • Priors (stated in Zhang's prior specification section):
        a_j ~ N(0,1) truncated to [-1,1]
        sigma2_j ~ Inverse-Gamma(alpha_s, beta_s) 

    Inputs/Outputs:
    ---------------
    DO NOT CHANGE (matches your current pipeline):
      input:  F (T x R), alpha_s, beta_s, rng
      output: a (R,), sigma2 (R,)
    """
    T, R = F.shape
    a = np.zeros(R)
    sigma2 = np.zeros(R)

    # For AR(1) we have (T-1) transitions: t=2..T
    nobs = T - 1

    for j in range(R):
        # This factor's time series
        y = F[1:, j]     # f_{t}
        x = F[:-1, j]    # f_{t-1}

        # Precompute sums used by the Gaussian regression formulas
        xx = float(x @ x)   # sum f_{t-1}^2
        xy = float(x @ y)   # sum f_{t-1} * f_t

        # --- (A) Start from a reasonable value (not the actual "posterior draw") ---
        # This just helps stability if the chain starts from a weird state.
        aj = xy / (xx + 1e-12)
        aj = np.clip(aj, -0.99, 0.99)

        # --- (B) Sample sigma2_j given a_j (this matches Eq. (4) + IG prior) ---
        # Given aj, the residuals are: e_t = f_t - a f_{t-1}
        # With a Gaussian error model and IG prior, sigma2 has an IG conditional.
        resid = y - aj * x
        sse = float(np.sum(resid ** 2))

        alpha_post = alpha_s + 0.5 * nobs
        beta_post = beta_s + 0.5 * sse

        # IG sampling using Gamma trick:
        # if G ~ Gamma(alpha_post, scale=1/beta_post), then 1/G ~ IG(alpha_post, beta_post)
        sigma2_j = 1.0 / rng.gamma(alpha_post, 1.0 / (beta_post + 1e-12))

        # --- (C) Sample a_j given sigma2_j (this is the correct Gibbs step) ---
        # This is just Bayesian linear regression:
        #   y = a x + noise, noise variance = sigma2_j
        # Prior a ~ N(0,1), then posterior is Normal, then we truncate to [-1,1]
        V = 1.0 / (1.0 + xx / sigma2_j)      # posterior variance
        m = V * (xy / sigma2_j)              # posterior mean

        # Enforce Zhang's stationarity restriction |a_j| < 1 (they state |a_j| < 1) :contentReference[oaicite:3]{index=3}
        # We do it via rejection sampling from the Normal posterior.
        while True:
            draw = m + np.sqrt(V) * rng.randn()
            if -1.0 <= draw <= 1.0:
                a[j] = draw
                break

        # --- (D) Resample sigma2_j using the FINAL a_j ---
        # This makes sure the returned sigma2[j] corresponds to the returned a[j].
        resid = y - a[j] * x
        sse = float(np.sum(resid ** 2))
        beta_post = beta_s + 0.5 * sse
        sigma2[j] = 1.0 / rng.gamma(alpha_post, 1.0 / (beta_post + 1e-12))

    return a, sigma2


def enforce_ident_order(F, Theta, a, sigma2, z, beta, p=None, s=None):
    R = len(a)
    score = sigma2 / np.maximum(1.0 - a**2, 1e-6)
    perm = np.argsort(-score)

    F2 = F[:, perm]
    Theta2 = Theta[:, perm]
    a2 = a[perm]
    sigma22 = sigma2[perm]
    z2 = z[perm].copy()

    # permute GDP betas
    b0 = beta[0]
    b1 = beta[1:1+R][perm]
    b2 = beta[1+R:1+2*R][perm]
    b3 = beta[1+2*R:1+3*R][perm]
    b4 = beta[-1]
    beta2 = np.r_[b0, b1, b2, b3, b4]

    # Zhang: z1 = 1
    z2[0] = 1

    out = [F2, Theta2, a2, sigma22, z2, beta2]

    if p is not None:
        p2 = p[perm].copy()
        p2[0] = 1.0
        out.append(p2)

    if s is not None:
        # s corresponds to factors j=2..R.
        # Build mapping old factor index -> new position:
        invperm = np.empty(R, dtype=int)
        invperm[perm] = np.arange(R)

        # new factor positions for old factors 2..R:
        new_pos = invperm[1:]  # positions (0..R-1) of old factors 2..R
        # but s must be in order of new factors 2..R, so we need to reorder s accordingly.
        # We want s2[k] = s[old_factor_that_is_now_(k+1)]
        # Easier: create s2 for new factors 2..R by pulling old s entries via perm:
        # perm tells which old factor is now at each new position.
        # For new position j (>=1), old factor index is perm[j], corresponding old s index = perm[j]-1.
        s2 = np.zeros(R-1, dtype=int)
        for new_j in range(1, R):
            old_factor = perm[new_j]
            if old_factor == 0:
                continue
            s2[new_j-1] = s[old_factor-1]
        out.append(s2)

    return tuple(out)
