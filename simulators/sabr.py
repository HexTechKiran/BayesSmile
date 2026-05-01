"""
SABR (Stochastic Alpha Beta Rho) model.

    dF_t    = sigma_t * F_t^beta * dW_t
    dsigma_t = alpha * sigma_t * dZ_t
    corr(dW, dZ) = rho

F_t is the forward price (no drift — martingale under the forward measure).
sigma uses the log-Euler scheme (geometric BM); F uses Euler-Maruyama.
F is absorbed at zero for beta < 1.

Parameters
----------
beta   : CEV exponent for the forward (0 <= beta <= 1)
alpha  : volatility of volatility
rho    : correlation between forward and vol Brownian motions
sigma0 : initial stochastic volatility
"""

import numpy as np
from .utils import correlated_normals


def sabr_model(
    beta: float,
    alpha: float,
    rho: float,
    sigma0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    F0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the SABR model.

    Parameters
    ----------
    return_latent : include the stochastic volatility path in the output

    Returns
    -------
    dict with keys:
        'log_returns' : np.ndarray, shape (T,)  — log(F_{t+1}/F_t)
        'volatility'  : np.ndarray, shape (T,)  — present when return_latent=True
    """
    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)
    Z_F = rng.standard_normal(T)
    Z_sigma = rho * Z_F + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(T)

    log_returns = np.empty(T)
    volatility = np.empty(T) if return_latent else None

    F = F0
    log_sigma = np.log(sigma0)

    for t in range(T):
        sigma = np.exp(log_sigma)
        F_pos = max(F, 0.0)

        if return_latent:
            volatility[t] = sigma

        dF = sigma * F_pos**beta * sqdt * Z_F[t]
        F_new = max(F_pos + dF, 0.0)

        log_returns[t] = np.log(F_new / F_pos) if F_pos > 0 else 0.0
        F = F_new

        # Log-Euler for sigma (exact for geometric BM)
        log_sigma += -0.5 * alpha**2 * dt + alpha * sqdt * Z_sigma[t]

    out = {"log_returns": log_returns}
    if return_latent:
        out["volatility"] = volatility
    return out
