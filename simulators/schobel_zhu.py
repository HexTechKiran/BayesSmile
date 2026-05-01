"""
Schöbel-Zhu (1999) stochastic volatility model.

    dS_t     = mu * S_t * dt + sigma_t * S_t * dW_t
    dsigma_t = kappa * (theta - sigma_t) * dt + xi * dB_t
    corr(dW, dB) = rho

Volatility (not variance) follows a mean-reverting Ornstein-Uhlenbeck process.
sigma_t can become negative; the instantaneous variance is sigma_t^2 (always >= 0).
Euler-Maruyama discretisation for sigma_t.

Reparameterized for identifiability:
    half_life : vol mean-reversion half-life in years  (kappa = log(2) / half_life)

Parameters
----------
mu         : asset drift (annualised)
half_life  : volatility mean-reversion half-life in years; > 0
theta      : long-run volatility level; > 0
xi         : vol-of-vol — diffusion coefficient of the OU process; > 0
rho        : correlation between asset and vol Brownian motions
sigma0     : initial instantaneous volatility
"""

import numpy as np
from .utils import correlated_normals


def schobel_zhu_model(
    mu: float,
    half_life: float,
    theta: float,
    xi: float,
    rho: float,
    sigma0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict:
    """Simulate one path under the Schöbel-Zhu model."""
    kappa = np.log(2.0) / half_life

    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)

    Z_S, Z_sigma = correlated_normals(rho, T, rng)

    log_returns = np.empty(T)
    volatility = np.empty(T) if return_latent else None

    sigma = sigma0

    for t in range(T):
        if return_latent:
            volatility[t] = sigma

        log_returns[t] = (mu - 0.5 * sigma ** 2) * dt + sigma * sqdt * Z_S[t]

        sigma = sigma + kappa * (theta - sigma) * dt + xi * sqdt * Z_sigma[t]

    out = {"log_returns": log_returns}
    if return_latent:
        out["volatility"] = volatility
    return out
