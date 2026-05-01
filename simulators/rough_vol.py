"""
Rough Fractional Stochastic Volatility (RFSV) model.

Log-volatility follows a scaled fractional Brownian motion (fBm):

    log(sigma_t) = log(sigma0) + nu * W^H_t

    dS_t = mu * S_t * dt + sigma_t * S_t * dW_t
    corr(dW_t, dW^H_t) ≈ rho  (contemporaneous correlation)

W^H is fBm with Hurst exponent H.  H < 0.5 gives "rough" paths (empirically
H ≈ 0.1 for equity volatility).  The fBm is simulated via the Davies-Harte
algorithm for exact covariance structure.

The price-vol correlation is approximated by mixing the normalised fBm
increment with an independent normal at each step.

Parameters
----------
mu     : drift of the asset (annualised)
H      : Hurst exponent (0 < H < 0.5 for rough volatility)
nu     : vol-of-vol scaling applied to the fBm
rho    : leverage correlation between price BM and fBm
sigma0 : initial instantaneous volatility
"""

import numpy as np
from .utils import fgn_davies_harte


def rough_vol_model(
    mu: float,
    H: float,
    nu: float,
    rho: float,
    sigma0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the RFSV model.

    Parameters
    ----------
    return_latent : include the stochastic volatility path in the output

    Returns
    -------
    dict with keys:
        'log_returns' : np.ndarray, shape (T,)
        'volatility'  : np.ndarray, shape (T,)  — present when return_latent=True
    """
    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)

    # Unit-variance FGN increments; scale to fBm increments with var = dt^(2H)
    xi = fgn_davies_harte(H, T, rng) * dt**H

    # Contemporaneous price-vol correlation via mixing
    Z_indep = rng.standard_normal(T)
    Z_S = rho * xi / np.std(xi) + np.sqrt(1.0 - rho**2) * Z_indep

    # Log-vol path: log(sigma_t) = log(sigma0) + nu * cumsum(xi)
    log_sigma = np.log(sigma0) + nu * np.cumsum(xi)
    log_sigma = np.concatenate([[np.log(sigma0)], log_sigma[:-1]])  # align: sigma at t drives step t

    log_returns = np.empty(T)
    volatility = np.empty(T) if return_latent else None

    for t in range(T):
        sigma = np.exp(log_sigma[t])
        if return_latent:
            volatility[t] = sigma
        log_returns[t] = (mu - 0.5 * sigma**2) * dt + sigma * sqdt * Z_S[t]

    out = {"log_returns": log_returns}
    if return_latent:
        out["volatility"] = volatility
    return out
