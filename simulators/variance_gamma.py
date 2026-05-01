"""
Variance Gamma (VG) model (Madan, Carr & Chang, 1998).

Log-returns are driven by a Brownian motion subordinated to a Gamma process:

    r_t = mu * dt + theta * (g_t - dt) + sigma * sqrt(g_t) * z_t

    g_t ~ Gamma(dt / nu, nu)          # mean = dt, variance = nu * dt
    z_t ~ N(0, 1)

The Gamma time-change g_t introduces stochastic activity time: g_t > dt means
more activity (larger moves), g_t < dt means less.  theta controls skewness
(negative for equity crash risk), nu controls excess kurtosis.

Because g_t and z_t are drawn independently at each step, log-returns are i.i.d.
(exchangeable).  There is no serial dependence — this is a pure distributional
model, not a stochastic volatility model.

Parameters
----------
mu    : mean annualised log-return drift
sigma : diffusion coefficient of the subordinated Brownian motion (> 0)
nu    : variance rate of the Gamma subordinator; controls kurtosis (> 0)
theta : drift of the arithmetic BM component; controls skewness (< 0 for equities)
"""

import numpy as np


def variance_gamma_model(
    mu: float,
    sigma: float,
    nu: float,
    theta: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict:
    """Simulate one path under the Variance Gamma model.

    Returns
    -------
    dict with keys:
        'log_returns'     : np.ndarray, shape (T,)
        'gamma_increments': np.ndarray, shape (T,)  — present when return_latent=True
    """
    rng = np.random.default_rng(rng)

    # Gamma increments: shape = dt/nu, scale = nu → mean dt, variance nu*dt
    g = rng.gamma(shape=dt / nu, scale=nu, size=T)
    Z = rng.standard_normal(T)

    # VG increment: centre the Gamma to have zero mean contribution
    log_returns = mu * dt + theta * (g - dt) + sigma * np.sqrt(g) * Z

    out = {"log_returns": log_returns}
    if return_latent:
        out["gamma_increments"] = g
    return out
