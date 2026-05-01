"""
3/2 stochastic volatility model.

    dS_t  = mu * S_t * dt + sqrt(nu_t) * S_t * dW_t
    dnu_t = nu_t * (omega - theta * nu_t) * dt + xi * nu_t^(3/2) * dB_t
    corr(dW, dB) = rho

The diffusion of nu scales as nu^(3/2), making variance shocks larger at
high variance levels (opposite asymmetry to Heston's sqrt(nu)).
Full-truncation Euler-Maruyama for nu; log-Euler for S.

Note: 1/nu follows a CIR process under this specification.

Reparameterized for identifiability:
    vol0 : initial annualized volatility  (nu0 = vol0^2)

Parameters
----------
mu    : drift of the asset (annualised)
omega : controls the long-run level; steady-state variance is omega / theta
theta : mean-reversion strength (pulls nu toward omega / theta)
xi    : volatility of variance
rho   : correlation between price and variance Brownian motions
vol0  : initial annualized volatility; > 0
"""

import numpy as np
from .utils import correlated_normals


def model_32(
    mu: float,
    omega: float,
    theta: float,
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
    """Simulate one path under the 3/2 model."""
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

        nu = (
            nu
            + nu_pos * (omega - theta * nu_pos) * dt
            + xi * nu_pos * sqrt_nu * sqdt * Z_nu[t]
        )

    out = {"log_returns": log_returns}
    if return_latent:
        if as_volatility:
            out["volatility"] = np.sqrt(variance)
        else:
            out["variance"] = variance
    return out
