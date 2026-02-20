from __future__ import annotations
import numpy as np
import numpy.linalg as npl


def _build_Psi_from_r(Psi: np.ndarray, i: int, idx: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Replace row/column i of a correlation matrix Psi with a new correlation vector r,
    while enforcing symmetry and unit diagonal.

    This assumes:
      - Psi is already a correlation matrix
      - r corresponds to correlations between variable i and all others (idx)
    """
    Psi_new = Psi.copy()

    # Insert the new correlations symmetrically
    Psi_new[i, idx] = r
    Psi_new[idx, i] = r

    # Correlation matrices must have ones on the diagonal
    np.fill_diagonal(Psi_new, 1.0)

    # Numerical safety: enforce exact symmetry
    Psi_new = 0.5 * (Psi_new + Psi_new.T)

    return Psi_new


def propose_column_rw_LKJ1(
    Psi: np.ndarray,
    i: int,
    step: float,
    rng=np.random,
) -> tuple[np.ndarray, bool]:
    """
    Propose a Metropolis–Hastings update for row/column i of a correlation matrix
    under an LKJ(1) prior.

    Key facts used here:
    --------------------
    • Under LKJ(1), the conditional distribution of the correlation vector r
      (row i without the diagonal) given the rest of the matrix is UNIFORM on
      the ellipsoid:
          r' R^{-1} r < 1
      where R is the submatrix of Psi excluding row/column i.

    • Writing R = L L' (Cholesky factorization), the transformation
          r = L u
      maps this ellipsoid to the unit ball:
          ||u|| < 1

    • We therefore do a symmetric random walk in u-space and reject proposals
      that leave the unit ball.

    Because:
      - the proposal is symmetric
      - the LKJ(1) prior is constant on its support
    the MH acceptance probability depends ONLY on the likelihood.
    """
    Psi = np.asarray(Psi, float)
    n = Psi.shape[0]

    # Indices of all variables except i
    idx = np.array([k for k in range(n) if k != i], dtype=int)

    # Extract the (n-1)x(n-1) submatrix R corresponding to "other" variables
    # This is the matrix that defines the ellipsoid constraint
    R0 = Psi[np.ix_(idx, idx)]

    # Enforce symmetry numerically (important for Cholesky)
    R0 = 0.5 * (R0 + R0.T)

    # Compute Cholesky factor of R0.
    # We only add a tiny jitter if strictly necessary to avoid numerical failure.
    try:
        L = npl.cholesky(R0)
    except npl.LinAlgError:
        eps = 1e-12
        try:
            L = npl.cholesky(R0 + eps * np.eye(n - 1))
        except npl.LinAlgError:
            # If this fails, we cannot safely propose from this state
            return Psi, False

    # Current correlation vector between variable i and all others
    r_cur = Psi[i, idx]

    # Convert r to u-space: r = L u  ⇒  u = L^{-1} r
    u_cur = npl.solve(L, r_cur)

    # Numerical guard:
    # Due to floating-point error, u_cur may lie *slightly* outside the unit ball
    # even if Psi is positive definite. Pull it slightly inside if needed.
    nu = npl.norm(u_cur)
    if nu >= 1.0:
        u_cur = u_cur / (nu + 1e-15) * (1.0 - 1e-12)

    # Random-walk proposal in u-space
    u_prop = u_cur + step * rng.randn(n - 1)

    # Reject immediately if proposal leaves the unit ball
    if npl.norm(u_prop) >= 1.0:
        return Psi, False

    # Map back to r-space
    r_prop = L @ u_prop

    # Build the proposed full correlation matrix
    Psi_prop = _build_Psi_from_r(Psi, i, idx, r_prop)

    # Final and definitive support check:
    # The proposed Psi MUST be positive definite.
    # If not, treat it as an "outside" proposal.
    try:
        _ = npl.cholesky(Psi_prop)
    except npl.LinAlgError:
        return Psi, False

    return Psi_prop, True


def sample_Psi_LKJ1_columnwise_rw(
    Psi_current: np.ndarray,
    loglik_fn,
    n_sweeps: int = 1,
    step: float = 0.05,
    rng=np.random,
) -> tuple[np.ndarray, int, int]:
    """
    Perform column-wise Metropolis–Hastings updates of a correlation matrix Psi
    under an LKJ(1) prior.

    Each sweep:
      - loops over all rows/columns i
      - proposes an update using propose_column_rw_LKJ1
      - accepts/rejects based ONLY on the likelihood

    Returns:
      Psi     : the updated correlation matrix
      accepts : number of accepted MH moves
      trials  : number of proposal attempts (including automatic rejections)
    """
    Psi = np.asarray(Psi_current, float)
    ll_cur = loglik_fn(Psi)

    n = Psi.shape[0]
    accepts = 0
    trials = 0

    for _ in range(n_sweeps):
        for i in range(n):
            Psi_prop, inside = propose_column_rw_LKJ1(Psi, i=i, step=step, rng=rng)
            trials += 1

            # If proposal was outside the support, skip MH step
            if not inside:
                continue

            ll_prop = loglik_fn(Psi_prop)

            # Standard MH acceptance step (likelihood-only for LKJ(1))
            if np.log(rng.rand()) < (ll_prop - ll_cur):
                Psi = Psi_prop
                ll_cur = ll_prop
                accepts += 1

    return Psi, accepts, trials
