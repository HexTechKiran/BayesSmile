"""
Continuous-time GARCH(1,1) stochastic volatility model.

    dS_t  = mu * S_t * dt + sqrt(nu_t) * S_t * dW_t
    dnu_t = kappa * (omega - nu_t) * dt + xi * nu_t * dB_t
    corr(dW, dB) = rho

Differs from Heston in the diffusion term of nu: xi * nu_t instead of
xi * sqrt(nu_t).  This makes the variance noise scale with its own level.
Full-truncation Euler-Maruyama for nu; log-Euler for S.

Reparameterized for identifiability:
    long_run_vol  : long-run annualized volatility  (omega = long_run_vol^2)
    half_life     : variance half-life in years      (kappa = log(2) / half_life)
    vol0          : initial annualized volatility    (nu0 = vol0^2)

Parameters
----------
mu           : drift of the asset (annualised)
long_run_vol : long-run volatility level (annualised); > 0
half_life    : mean-reversion half-life in years; > 0
xi           : volatility of variance
rho          : correlation between price and variance Brownian motions
vol0         : initial instantaneous volatility; > 0
"""

import numpy as np
from .utils import correlated_normals


def garch_model(
    mu: float,
    long_run_vol: float,
    half_life: float,
    xi: float,
    rho: float,
    vol0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    as_volatility: bool = False,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the continuous GARCH model."""
    omega = long_run_vol ** 2
    theta = np.log(2.0) / half_life
    nu0 = vol0 ** 2

    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)
    Z_S, Z_nu = correlated_normals(rho, T, rng)

    log_returns = np.empty(T)
    variance = np.empty(T) if return_latent else None

    log_S = np.log(S0)
    nu = nu0

    for t in range(T):
        nu_pos = max(nu, 0.0)
        sqrt_nu = np.sqrt(nu_pos)

        if return_latent:
            variance[t] = nu_pos
        log_returns[t] = (mu - 0.5 * nu_pos) * dt + sqrt_nu * sqdt * Z_S[t]
        log_S += log_returns[t]

        nu = nu + theta * (omega - nu_pos) * dt + xi * nu_pos * sqdt * Z_nu[t]

    out = {"log_returns": log_returns}
    if return_latent:
        if as_volatility:
            out["volatility"] = np.sqrt(variance)
        else:
            out["variance"] = variance
    return out
