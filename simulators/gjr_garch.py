"""
GJR-GARCH(1,1) model (Glosten, Jagannathan & Runkle, 1993).

    r_t      = mu * dt + sqrt(nu_t * dt) * z_t
    nu_{t+1} = omega + (alpha + gamma * I_t) * nu_t * z_t^2 + beta * nu_t

nu_t is the annualised conditional variance.  I_t = 1(z_t < 0) is the leverage
indicator.  Stationarity requires alpha + gamma/2 + beta < 1.
Long-run variance: E[nu] = omega / (1 - alpha - gamma/2 - beta) = long_run_vol^2.

Reparameterized for identifiability:
    long_run_vol    : long-run annualized volatility; > 0
                      omega = long_run_vol^2 * (1 - persistence)
    persistence     : alpha + gamma/2 + beta in (0, 1)
    arch_share      : (alpha + gamma/2) / persistence in (0, 1)
    leverage_ratio  : gamma / (2*(alpha + gamma/2)) in (0, 1)
                      alpha = persistence * arch_share * (1 - leverage_ratio)
                      gamma = 2 * persistence * arch_share * leverage_ratio
                      beta  = persistence * (1 - arch_share)
    vol0            : initial annualized volatility; > 0  (nu0 = vol0^2)

Parameters
----------
mu              : drift of the asset (annualised)
long_run_vol    : long-run volatility level; > 0
persistence     : total GARCH persistence; in (0, 1)
arch_share      : fraction of persistence from ARCH + leverage; in (0, 1)
leverage_ratio  : fraction of ARCH response attributable to leverage; in (0, 1)
vol0            : initial annualized volatility; > 0
"""

import numpy as np


def gjr_garch_model(
    mu: float,
    long_run_vol: float,
    persistence: float,
    arch_share: float,
    leverage_ratio: float,
    vol0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    as_volatility: bool = False,
    rng: np.random.Generator = None,
) -> dict:
    """Simulate one path under the GJR-GARCH(1,1) model."""
    alpha = persistence * arch_share * (1.0 - leverage_ratio)
    gamma = 2.0 * persistence * arch_share * leverage_ratio
    beta = persistence * (1.0 - arch_share)
    omega = long_run_vol ** 2 * (1.0 - persistence)
    nu0 = vol0 ** 2

    rng = np.random.default_rng(rng)
    Z = rng.standard_normal(T)

    log_returns = np.empty(T)
    variance = np.empty(T) if return_latent else None

    nu = nu0

    for t in range(T):
        if return_latent:
            variance[t] = nu

        log_returns[t] = mu * dt + np.sqrt(max(nu, 0.0) * dt) * Z[t]

        leverage = 1.0 if Z[t] < 0 else 0.0
        nu = omega + (alpha + gamma * leverage) * nu * Z[t] ** 2 + beta * nu

    out = {"log_returns": log_returns}
    if return_latent:
        if as_volatility:
            out["volatility"] = np.sqrt(variance)
        else:
            out["variance"] = variance
    return out
