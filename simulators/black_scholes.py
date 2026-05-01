"""
Black-Scholes (1973) model.

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Equivalently, log-returns are i.i.d. Gaussian:

    log(S_{t+1}/S_t) ~ N((mu - 0.5*sigma^2)*dt, sigma^2*dt)

Price uses the exact log-Euler scheme (no discretisation error).

Parameters
----------
mu    : drift of the asset (annualised)
sigma : constant volatility (annualised)
"""

import numpy as np


def black_scholes_model(
    mu: float,
    sigma: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the Black-Scholes model.

    Returns
    -------
    dict with keys:
        'log_returns' : np.ndarray, shape (T,)
    """
    rng = np.random.default_rng(rng)
    Z = rng.standard_normal(T)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return {"log_returns": log_returns}
