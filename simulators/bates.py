"""
Bates (1996) stochastic volatility model with jumps.

    dlog S_t = (mu - 0.5*v_t - lambda_j*k_bar) dt + sqrt(v_t) dW_t + J dN_t
    dv_t     = kappa*(theta - v_t) dt + xi*sqrt(v_t) dB_t
    corr(dW, dB) = rho

Variance uses full-truncation Euler-Maruyama.
Log-jump sizes ~ N(mu_j, sigma_j^2), Poisson intensity lambda_j.
k_bar = exp(mu_j + 0.5*sigma_j^2) - 1.

Reparameterized for identifiability:
    long_run_vol : long-run annualized volatility  (theta = long_run_vol^2)
    half_life    : variance half-life in years      (kappa = log(2) / half_life)
    vol0         : initial annualized volatility    (v0 = vol0^2)

Parameters
----------
mu           : asset drift
half_life    : variance mean-reversion half-life in years; > 0
long_run_vol : long-run volatility level; > 0
xi           : vol-of-vol
rho          : price-variance correlation
vol0         : initial instantaneous volatility; > 0
lambda_j     : Poisson jump intensity, jumps per year
mu_j         : mean log-jump size
sigma_j      : std of log-jump size
"""

import numpy as np
from .utils import correlated_normals


def bates_model(
    mu: float,
    half_life: float,
    long_run_vol: float,
    xi: float,
    rho: float,
    vol0: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict:
    """Simulate one path under the Bates (SVJ) model."""
    kappa = np.log(2.0) / half_life
    theta = long_run_vol ** 2
    v0 = vol0 ** 2

    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)

    k_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    Z_S, Z_v = correlated_normals(rho, T, rng)
    n_jumps = rng.poisson(lambda_j * dt, size=T)

    jump_component = np.zeros(T)
    for t in np.where(n_jumps > 0)[0]:
        jump_component[t] = rng.normal(mu_j, sigma_j, int(n_jumps[t])).sum()

    log_returns = np.empty(T)
    variance = np.empty(T) if return_latent else None

    v = v0
    for t in range(T):
        v_pos = max(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        if return_latent:
            variance[t] = v_pos

        log_returns[t] = (
            (mu - 0.5 * v_pos - lambda_j * k_bar) * dt
            + sqrt_v * sqdt * Z_S[t]
            + jump_component[t]
        )

        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqdt * Z_v[t]

    out = {"log_returns": log_returns}
    if return_latent:
        out["variance"] = variance
        out["jump_component"] = jump_component
    return out
