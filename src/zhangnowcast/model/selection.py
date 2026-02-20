import numpy as np
from math import lgamma

def _log_beta_pdf(x, a, b):
    """log Beta(a,b) density at x (up to normalization already included)."""
    if x <= 0 or x >= 1:
        return -np.inf
    return (lgamma(a + b) - lgamma(a) - lgamma(b)
            + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x))

def _logit(x):
    """logit transform with clipping to avoid log(0)."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return np.log(x) - np.log(1 - x)

def _inv_logit(z):
    """
    Numerically stable inverse-logit.
    Avoids overflow in exp() for extreme z.
    """
    if np.isscalar(z):
        if z >= 0:
            ez = np.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = np.exp(z)
            return ez / (1.0 + ez)
    else:
        z = np.asarray(z)
        out = np.empty_like(z, dtype=float)
        pos = (z >= 0)
        ez = np.exp(-z[pos])
        out[pos] = 1.0 / (1.0 + ez)
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

def mh_update_pi(pi, s_vec, alpha_pi, beta_pi, step=0.10, rng=np.random):
    """
    Zhang Eq. (15): pi ~ Beta(alpha_pi, beta_pi)
    and Zhang Eq. (14) implies latent 'spike/slab' indicators:
        s_k ~ Bernoulli(pi^k) for k=2..R  (your s_vec corresponds to k=2..R)

    We update pi using a Random-Walk MH step on the logit scale:
        z = logit(pi)
        z' = z + step * N(0,1)
        pi' = inv_logit(z')

    Because we propose symmetrically in z (not in pi), the MH target on z-scale
    must include the Jacobian |d pi / d z| = pi*(1-pi), i.e. add log(pi)+log(1-pi).
    """
    z = _logit(pi)
    z_prop = z + step * rng.randn()
    pi_prop = _inv_logit(z_prop)

    def logpost_pi(pi_val):
        # prior on pi
        if not (0 < pi_val < 1):
            return -np.inf
        lp = _log_beta_pdf(pi_val, alpha_pi, beta_pi)

        # likelihood from s_k ~ Bern(pi^k), k=2..R
        # enumerate(..., start=2) makes idx = k
        for k, s_k in enumerate(s_vec, start=2):
            w = np.clip(pi_val ** k, 1e-300, 1 - 1e-16)  # pi^k
            lp += s_k * np.log(w) + (1 - s_k) * np.log(1 - w)
        return lp

    # target density in z-space = target(pi) * |dpi/dz|
    def logpost_z(pi_val):
        return logpost_pi(pi_val) + np.log(pi_val) + np.log(1.0 - pi_val)

    if np.log(rng.rand()) < (logpost_z(pi_prop) - logpost_z(pi)):
        return pi_prop
    return pi

def sample_z_p_pi(
    z, p, s, pi,
    loglik0, loglik1,
    alpha_p=1.0, beta_p=3.0,
    alpha_pi=2.0, beta_pi=2.0,
    rng=np.random
):
    """
    This block implements Zhang Eq. (14)-(15) for factor selection. :contentReference[oaicite:6]{index=6}

    Model pieces:
      - Eq. (7): z_j | p_j ~ Bernoulli(p_j)
      - Eq. (14): p_j is spike-and-slab:
            with prob (1 - pi^j): p_j = 0 (spike)  -> forces z_j=0
            with prob (pi^j):     p_j ~ Beta(alpha_p, beta_p) (slab)
        for j = 2..R, and Zhang fixes z_1 = 1.
      - Eq. (15): pi ~ Beta(alpha_pi, beta_pi)

    We use latent s_j (j=2..R) to indicate spike/slab:
      s_j=0 => spike => p_j=0 and z_j=0
      s_j=1 => slab  => p_j ~ Beta(...) and z_j | p_j uses likelihood + prior

    Inputs loglik0/loglik1 should be:
      loglik0[j] = log p(data | z_j=0, others fixed)
      loglik1[j] = log p(data | z_j=1, others fixed)
    """
    R = len(z)

    # Zhang convention: first factor always included
    z[0] = 1
    p[0] = 1.0

    # Prior predictive under slab:
    # P(z=0 | slab) = E[1-p] = beta_p/(alpha_p+beta_p)
    slab_p_z0 = beta_p / (alpha_p + beta_p)

    #  Step 1: update spike/slab indicators s_j (for j=2..R) 
    for j in range(1, R):  # python j=1..R-1 corresponds to factor index k=j+1
        k = j + 1

        if z[j] == 1:
            # if z=1, must be slab
            s[j - 1] = 1
        else:
            # if z=0, posterior P(s=1 | z=0) only depends on mixture weights + slab predictive
            w = np.clip(pi ** k, 1e-300, 1 - 1e-16)  # pi^k
            num = w * slab_p_z0
            den = num + (1.0 - w)
            prob_s1 = num / den
            s[j - 1] = 1 if (rng.rand() < prob_s1) else 0

        if s[j - 1] == 0:
            # spike component: p_j = 0 => z_j = 0 deterministically
            p[j] = 0.0
            z[j] = 0

    #  Step 2: for slab components, update z_j and then p_j 
    for j in range(1, R):
        if s[j - 1] == 0:
            continue

        # ensure p_j is valid if coming from a bad init
        if not (0.0 < p[j] < 1.0):
            p[j] = rng.beta(alpha_p, beta_p)

        # Posterior log-odds for z_j:
        # logit P(z=1|...) = logit(p_j) + [loglik1 - loglik0]
        dLL = loglik1[j] - loglik0[j]
        log_odds = np.log(p[j]) - np.log(1.0 - p[j]) + dLL

        # stable sigmoid for probability
        if log_odds >= 0:
            prob1 = 1.0 / (1.0 + np.exp(-log_odds))
        else:
            e = np.exp(log_odds)
            prob1 = e / (1.0 + e)

        z[j] = 1 if (rng.rand() < prob1) else 0

        # Conjugate update p_j | z_j (slab):
        p[j] = rng.beta(alpha_p + z[j], beta_p + 1 - z[j])

    #  Step 3: MH update for pi using s_j likelihood + Beta prior 
    pi = mh_update_pi(pi, s, alpha_pi, beta_pi, rng=rng)

    return z, p, s, pi
