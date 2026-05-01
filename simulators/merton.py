"""
Merton (1976) jump-diffusion model.

    dS_t = (mu - lambda_j * k_bar) * S_t * dt + sigma * S_t * dW_t + J * S_t * dN_t

N_t is a Poisson process with intensity lambda_j (jumps per year).
Log-jump sizes ~ N(mu_j, sigma_j^2).
k_bar = E[J] = exp(mu_j + 0.5 * sigma_j^2) - 1 is the mean jump (risk-neutral drift correction).

Parameters
----------
mu       : asset drift (annualised)
sigma    : diffusion volatility (annualised, > 0)
lambda_j : Poisson jump intensity, jumps per year (> 0)
mu_j     : mean log-jump size
sigma_j  : std of log-jump size (> 0)
"""

import numpy as np


def merton_model(
    mu: float,
    sigma: float,
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
    """Simulate one path under the Merton jump-diffusion model.

    Returns
    -------
    dict with keys:
        'log_returns'    : np.ndarray, shape (T,)
        'jump_component' : np.ndarray, shape (T,)  — present when return_latent=True
    """
    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)

    k_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    Z = rng.standard_normal(T)
    n_jumps = rng.poisson(lambda_j * dt, size=T)

    jump_component = np.zeros(T)
    for t in np.where(n_jumps > 0)[0]:
        jump_component[t] = rng.normal(mu_j, sigma_j, int(n_jumps[t])).sum()

    log_returns = (
        (mu - 0.5 * sigma ** 2 - lambda_j * k_bar) * dt
        + sigma * sqdt * Z
        + jump_component
    )

    out = {"log_returns": log_returns}
    if return_latent:
        out["jump_component"] = jump_component
    return out
