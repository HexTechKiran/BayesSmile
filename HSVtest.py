import pandas as pd
import numpy as np
import jax
import bayesflow as bf
import keras
import QuantLib as ql
import matplotlib.pyplot as plt

SPOT_MEAN = 100
SPOT_STD = 15

VOL_MEAN = 0.04
VOL_STD = 0.002

KAPPA_MEAN = 1.0
KAPPA_STD = 0.25

SIGMA_MEAN = 0.66
SIGMA_STD = 0.15

RHO_ALPHA = 2
RHO_BETA = 5

def to_lognormal(mean, std) :
    sigma = np.sqrt(np.log(1 + std**2 / mean**2))
    mu = np.log(mean) - sigma**2 / 2
    return mu, sigma

def to_spot(norm_draw):
    log_mean, log_std = to_lognormal(SPOT_MEAN, SPOT_STD)
    scaled_norm = norm_draw * log_std + log_mean
    return np.exp(scaled_norm)

def to_v0(norm_draw):
    log_mean, log_std = to_lognormal(VOL_MEAN, VOL_STD)
    scaled_norm = norm_draw * (log_std * 2) + log_mean
    return np.exp(scaled_norm)

def to_theta(norm_draw):
    log_mean, log_std = to_lognormal(VOL_MEAN, VOL_STD)
    scaled_norm = norm_draw * log_std + log_mean
    return np.exp(scaled_norm)

def to_kappa(norm_draw):
    log_mean, log_std = to_lognormal(KAPPA_MEAN, KAPPA_STD)
    scaled_norm = norm_draw * log_std + log_mean
    return np.exp(scaled_norm)

def draw_sigma(upper_bound) :
    epsilon = 10 ** (-6)
    return np.random.beta(6, 6) * (0.9 * upper_bound) + epsilon

def draw_rho() :
    # Here we force this value negative and bound it at -0.9 to avoid over-correlating our processes
    return np.random.beta(RHO_ALPHA, RHO_BETA) * -0.9 - 0.05

def draw_priors() :
    spot = np.random.normal()
    v0 = np.random.normal()
    theta = np.random.normal()
    kappa = np.random.normal()
    sigma = draw_sigma(upper_bound=np.sqrt(2*to_kappa(kappa)*to_theta(theta)))
    rho = draw_rho()
    return {"spot":spot,
            "v0":v0,
            "theta":theta,
            "kappa":kappa,
            "sigma":sigma,
            "rho":rho}

def single_heston_draw(spot, v0, theta, kappa, sigma, rho, steps=2520, dt=1/2520):
    # Transform draws for spot, v0, theta, and kappa into the correct space
    spot = to_spot(spot)
    v0 = to_v0(v0)
    theta = to_theta(theta)
    kappa = to_kappa(kappa)

    spot_quote = ql.SimpleQuote(spot)
    yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.UnitedStates(ql.UnitedStates.NYSE), 0.05, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.UnitedStates(ql.UnitedStates.NYSE), 0.0, ql.Actual365Fixed()))

    process = ql.HestonProcess(yield_ts, div_ts, ql.QuoteHandle(spot_quote), v0, kappa, theta, sigma, rho, ql.HestonProcess.QuadraticExponentialMartingale)

    # Initialize path arrays
    x = np.zeros(steps + 1)
    v = np.zeros(steps + 1)
    x[0], v[0] = spot, v0

    state = ql.Array(2)
    # we use log prices because the Heston Process uses log prices internally
    state[0], state[1] = np.log(spot), v0

    dw = np.random.standard_normal((steps, 2)) # Standard Normal shocks

    t = 0.0
    for i in range(1, steps + 1):
        state = process.evolve(t, state, dt, ql.Array([dw[i-1, 0], dw[i-1, 1]]))
        x[i] = np.exp(state[0])
        v[i] = state[1]
        t += dt

    return dict(path=x, v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho)

priors = np.asarray([list(draw_priors().values()) for _ in range(1000)]).T
prior_samples = dict(spot=priors[0], v0=priors[1], theta=priors[2], kappa=priors[3], sigma=priors[4], rho=priors[5])
paths = [single_heston_draw(*prior) for prior in priors.T]

"""
grid2 = bf.diagnostics.plots.pairs_samples(
    prior_samples, variable_keys=["spot", "v0", "theta", "kappa", "sigma", "rho"]
)

plt.show()
"""

log_paths = []
raw_paths = [path['path'] for path in paths]
for path in paths: log_paths.append(np.log(path["path"]))
for path in raw_paths: plt.plot(path)

plt.show()
