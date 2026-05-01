"""
Constant Elasticity of Variance (CEV) model.

    dS_t = mu * S_t * dt + sigma * S_t^gamma * dW_t

Volatility is a deterministic function of the price level, not a separate
stochastic process (hence some argue it is not truly an SV model).
Euler-Maruyama on S directly; S is absorbed at zero if it goes non-positive.

Parameters
----------
mu    : drift of the asset (annualised)
sigma : volatility scale coefficient
gamma : elasticity exponent
          gamma < 1  -> leverage effect (vol falls as price rises)
          gamma = 1  -> reduces to geometric Brownian motion
          gamma > 1  -> vol rises with price
"""

import numpy as np


def cev_model(
    mu: float,
    sigma: float,
    gamma: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the CEV model.

    Parameters
    ----------
    return_latent : include the instantaneous volatility path in the output

    Returns
    -------
    dict with keys:
        'log_returns' : np.ndarray, shape (T,)
        'volatility'  : np.ndarray, shape (T,)  — present when return_latent=True
    """
    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)
    Z = rng.standard_normal(T)

    log_returns = np.empty(T)
    volatility = np.empty(T) if return_latent else None

    S = S0

    for t in range(T):
        S_pos = max(S, 0.0)

        if return_latent:
            volatility[t] = sigma * S_pos ** (gamma - 1.0)
        dS = mu * S_pos * dt + sigma * S_pos**gamma * sqdt * Z[t]
        S_new = max(S_pos + dS, 0.0)

        log_returns[t] = np.log(S_new / S_pos) if S_pos > 0 else 0.0
        S = S_new

    out = {"log_returns": log_returns}
    if return_latent:
        out["volatility"] = volatility
    return out
