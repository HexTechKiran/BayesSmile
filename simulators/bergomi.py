"""
Bergomi (2005) 1-factor stochastic volatility model.

    dS_t = mu * S_t * dt + sqrt(V_t) * S_t * dB_t
    V_t  = V0 * exp(omega * X_t - 0.5 * omega^2 * var_X(t))
    dX_t = -kappa * X_t * dt + dW_t          (Ornstein-Uhlenbeck factor)
    corr(dB, dW) = rho

V_t is the spot variance driven by a mean-reverting OU factor X_t that starts
at zero.  With a flat initial forward variance curve (xi_0(t) = V0 for all t),
the martingale correction var_X(t) = (1 - exp(-2*kappa*t)) / (2*kappa) ensures
E[V_t] = V0 at every horizon.

X is simulated via the exact OU transition (no Euler error); S uses
the log-Euler scheme.

Reparameterized for identifiability:
    vol0      : initial spot volatility  (V0 = vol0^2)
    half_life : OU mean-reversion half-life in years  (kappa = log(2) / half_life)

Parameters
----------
mu        : drift of the asset (annualised)
vol0      : initial spot volatility; > 0
half_life : mean-reversion half-life of the OU factor in years; > 0
omega     : vol-of-vol (scales the OU factor's impact on log-variance)
rho       : correlation between the price BM and the OU factor BM
"""

import numpy as np
from .utils import correlated_normals


def bergomi_model(
    mu: float,
    vol0: float,
    half_life: float,
    omega: float,
    rho: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    as_volatility: bool = False,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the 1-factor Bergomi model."""
    V0 = vol0 ** 2
    kappa = np.log(2.0) / half_life

    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)
    Z_S, Z_X = correlated_normals(rho, T, rng)

    if kappa > 0:
        e_kdt = np.exp(-kappa * dt)
        ou_std = np.sqrt((1.0 - e_kdt**2) / (2.0 * kappa))
    else:
        e_kdt = 1.0
        ou_std = sqdt

    log_returns = np.empty(T)
    variance = np.empty(T) if return_latent else None

    X = 0.0

    for t in range(T):
        t_now = t * dt
        if kappa > 0:
            var_X = (1.0 - np.exp(-2.0 * kappa * t_now)) / (2.0 * kappa)
        else:
            var_X = t_now

        V = V0 * np.exp(omega * X - 0.5 * omega**2 * var_X)
        V_pos = max(V, 0.0)

        if return_latent:
            variance[t] = V_pos

        log_returns[t] = (mu - 0.5 * V_pos) * dt + np.sqrt(V_pos) * sqdt * Z_S[t]

        X = e_kdt * X + ou_std * Z_X[t]

    out = {"log_returns": log_returns}
    if return_latent:
        if as_volatility:
            out["volatility"] = np.sqrt(variance)
        else:
            out["variance"] = variance
    return out
