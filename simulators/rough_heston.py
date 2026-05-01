"""
Rough Heston model (El Euch & Rosenbaum, 2019).

Variance is driven by a fractional Brownian motion with Hurst exponent H < 0.5:

    v_t = v_0 + (1/Γ(H+½)) * ∫_0^t (t-s)^{H-½} [κ(θ-v_s) ds + ξ√v_s dB_s]

    dlog S_t = (μ - ½ v_t) dt + √v_t dW_t,   corr(dW, dB) = ρ

Discretized via the Euler scheme for Volterra equations (O(T²) per path).
Variance is full-truncated at zero.

Reparameterized for identifiability:
    long_run_vol : long-run annualized volatility  (theta = long_run_vol^2)
    half_life    : variance half-life in years      (kappa = log(2) / half_life)
    vol0         : initial annualized volatility    (v0 = vol0^2)

Parameters
----------
mu           : asset drift
half_life    : variance mean-reversion half-life in years; > 0
long_run_vol : long-run volatility level; > 0
xi           : vol-of-vol; > 0
rho          : price-variance correlation (< 0 for leverage)
H            : Hurst exponent (0 < H < 0.5 for rough volatility)
vol0         : initial instantaneous volatility; > 0
"""

import math
import numpy as np


def rough_heston_model(
    mu: float,
    half_life: float,
    long_run_vol: float,
    xi: float,
    rho: float,
    H: float,
    vol0: float,
    *,
    T: int = 252,
    dt: float = 1.0 / 252,
    S0: float = 1.0,
    return_latent: bool = True,
    rng: np.random.Generator = None,
) -> dict:
    """Simulate one path under the Rough Heston model."""
    kappa = np.log(2.0) / half_life
    theta = long_run_vol ** 2
    v0 = vol0 ** 2

    rng = np.random.default_rng(rng)
    sqdt = np.sqrt(dt)

    Z_B = rng.standard_normal(T)
    Z_S = rho * Z_B + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(T)

    Gamma_H = math.gamma(H + 0.5)
    lags = np.arange(1, T + 1, dtype=np.float64)
    kernel = (lags * dt) ** (H - 0.5) / Gamma_H

    v_arr = np.empty(T + 1)
    v_arr[0] = v0
    innovations = np.empty(T)
    log_returns = np.empty(T)

    for n in range(T):
        v_n = max(v_arr[n], 0.0)
        sqrt_v_n = np.sqrt(v_n)

        log_returns[n] = (mu - 0.5 * v_n) * dt + sqrt_v_n * sqdt * Z_S[n]
        innovations[n] = kappa * (theta - v_n) * dt + xi * sqrt_v_n * sqdt * Z_B[n]

        v_arr[n + 1] = v0 + np.dot(kernel[n::-1], innovations[: n + 1])

    out = {"log_returns": log_returns}
    if return_latent:
        out["variance"] = v_arr[:T]
    return out
