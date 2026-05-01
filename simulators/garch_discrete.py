"""
Discrete-time GARCH(1,1) model.

    r_t      = mu * dt + sqrt(nu_t * dt) * z_t
    nu_{t+1} = omega + alpha * nu_t * z_t^2 + beta * nu_t

nu_t is the annualised conditional variance.  Stationarity requires alpha + beta < 1.
Long-run variance: E[nu] = omega / (1 - alpha - beta) = long_run_vol^2.

Reparameterized for identifiability:
    long_run_vol : long-run annualized volatility; > 0
                   omega = long_run_vol^2 * (1 - persistence)
    persistence  : alpha + beta in (0, 1)
    arch_share   : alpha / persistence in (0, 1)
                   alpha = persistence * arch_share
                   beta  = persistence * (1 - arch_share)
    vol0         : initial annualized volatility; > 0  (nu0 = vol0^2)

Parameters
----------
mu           : drift of the asset (annualised)
long_run_vol : long-run volatility level; > 0
persistence  : total GARCH persistence = alpha + beta; in (0, 1)
arch_share   : fraction of persistence attributable to ARCH; in (0, 1)
vol0         : initial annualized volatility; > 0
"""

import numpy as np


def garch_discrete_model(
    mu: float,
    long_run_vol: float,
    persistence: float,
    arch_share: float,
    vol0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    as_volatility: bool = False,
    rng: np.random.Generator = None,
) -> dict[str, np.ndarray]:
    """Simulate one path under the discrete GARCH(1,1) model."""
    alpha = persistence * arch_share
    beta = persistence * (1.0 - arch_share)
    omega = long_run_vol ** 2 * (1.0 - persistence)
    nu0 = vol0 ** 2

    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)
    Z = rng.standard_normal(T)

    log_returns = np.empty(T)
    variance = np.empty(T) if return_latent else None

    nu = nu0

    for t in range(T):
        if return_latent:
            variance[t] = nu

        log_returns[t] = mu * dt + np.sqrt(max(nu, 0.0) * dt) * Z[t]

        nu = omega + (alpha * Z[t] ** 2 + beta) * nu

    out = {"log_returns": log_returns}
    if return_latent:
        if as_volatility:
            out["volatility"] = np.sqrt(variance)
        else:
            out["variance"] = variance
    return out
