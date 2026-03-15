import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import pandas as pd
import numpy as np
import bayesflow as bf
import keras
import jax
import QuantLib as ql
import matplotlib.pyplot as plt

jax.config.update('jax_default_device', jax.devices('cpu')[0])

SPOT_MEAN = 100
SPOT_STD = 15

VOL_MEAN = 0.04
VOL_STD = 0.002

KAPPA_MEAN = 1.0
KAPPA_STD = 0.25

SIGMA_MEAN = 0.66
SIGMA_STD = 0.15
SIGMA_SCALE = 10 # Linear scaling parameter for model fitting

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
    return (np.random.beta(6, 6) * (0.9 * upper_bound) + epsilon) * SIGMA_SCALE

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

def single_heston_draw(spot, v0, theta, kappa, sigma, rho, steps=2520, dt=1/2520, maturity=1.0):
    # Transform draws for spot, v0, theta, and kappa into the correct space
    spot = to_spot(spot)
    v0 = to_v0(v0)
    theta = to_theta(theta)
    kappa = to_kappa(kappa)
    sigma = sigma / SIGMA_SCALE

    spot_quote = ql.SimpleQuote(spot)
    yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.UnitedStates(ql.UnitedStates.NYSE), 0.05, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.UnitedStates(ql.UnitedStates.NYSE), 0.0, ql.Actual365Fixed()))

    process = ql.HestonProcess(yield_ts, div_ts, ql.QuoteHandle(spot_quote), v0, kappa, theta, sigma, rho, ql.HestonProcess.QuadraticExponentialMartingale)

    # Create a gaussian random sequence
    sequence_generator = ql.UniformRandomSequenceGenerator(2 * steps, ql.UniformRandomGenerator())
    gaussian_sequence_generator = ql.GaussianRandomSequenceGenerator(sequence_generator)

    # Define a path generator for this process
    time_grid = ql.TimeGrid(maturity, steps)
    path_generator = ql.GaussianMultiPathGenerator(
    process,
    time_grid,
    gaussian_sequence_generator,
    False # Brownian bridge flag
    )

    # Generate the price path
    sample = path_generator.next()
    multi_path = sample.value()

    price_path = np.array([multi_path[0][i] for i in range(steps + 1)])

    # Calculate and return log prices
    log_returns = np.log(price_path[1:] / price_path[:-1])

    return dict(log_returns=log_returns)

simulator = bf.make_simulator([draw_priors, single_heston_draw])

prior_samples = simulator.simulators[0].sample(1000)

grid = bf.diagnostics.plots.pairs_samples(
    prior_samples, variable_keys=["spot", "v0", "theta", "kappa", "sigma", "rho"]
)

plt.savefig("HSVtest_diagnostic_pairplot.png")

posterior_samples = simulator.sample(1000)

paths = posterior_samples['log_returns']
price_paths = []


for path in paths:
    spot = np.random.normal(100, 2)
    price_path = spot * np.exp(np.cumsum(np.concatenate([[0], path])))
    plt.plot(price_path)

plt.savefig("HSVtest_price_path.png")

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .as_time_series("log_returns")
    .standardize(include="log_returns", mean=0, std=0.004)
    .standardize(include="rho", mean=-0.307, std=0.1435)
    .concatenate(["v0", "theta", "kappa", "sigma", "rho"], into="inference_variables")
    .rename("log_returns", "summary_variables")
)

summary_net = bf.networks.TimeSeriesNetwork(dropout=0.1)

inference_net = bf.networks.DiffusionModel(dropout=0.1)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    summary_network=summary_net,
    inference_network=inference_net,
    checkpoint_path="motion_workflow/"
)

train = workflow.simulate(10000)
validation = workflow.simulate(300)

history = workflow.fit_offline(data=train,
                               epochs=100,
                               batch_size=32,
                               validation_data=validation)

f = bf.diagnostics.plots.loss(history)
plt.savefig("HSVtest_loss.png")

num_datasets = 300
num_samples = 1000

# Simulate 300 scenarios
print("Running simulations")
test_sims = workflow.simulate(num_datasets)

# Obtain num_samples posterior samples per scenario
print("Sampling")
samples = workflow.sample(conditions=test_sims, num_samples=num_samples)

print("Making plots")
f = bf.diagnostics.plots.recovery(samples, test_sims)

plt.savefig("HSVtest_posterior_plot.png")