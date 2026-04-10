import numpy as np
import bayesflow as bf
import QuantLib as ql
import matplotlib.pyplot as plt
import jax

jax.config.update('jax_default_device', jax.devices('cpu')[0])

prefix = "HSV_noTK_logpath_"

import jax
jax.devices()

SPOT_MEAN = 100
SPOT_STD = 15

VOL_MEAN = 0.04
VOL_STD = 0.02

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

def draw_priors():
    #spot = np.random.normal()
    # spot = 100
    v0 = np.random.normal()
    #theta = np.random.normal()
    #kappa = np.random.normal()
    sigma = draw_sigma(upper_bound=np.sqrt(2*1*0.04))
    rho = draw_rho()
    return {#"spot":spot,
            "v0":v0,
            #"theta":theta,
            #"kappa":kappa,
            "sigma":sigma,
            "rho":rho}

def single_heston_draw(v0, sigma, rho, spot=100, steps=500, dt=1/500, maturity=1.0):
    # Transform draws for spot, v0, theta, and kappa into the correct space
    # spot = to_spot(spot)
    v0 = to_v0(v0)
    theta = 0.04#to_theta(theta)
    kappa = 1#to_kappa(kappa)
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

    return dict(price_path=log_returns)

simulator = bf.make_simulator([draw_priors, single_heston_draw])

prior_samples = simulator.simulators[0].sample(1000)

grid = bf.diagnostics.plots.pairs_samples(
    prior_samples, variable_keys=["v0", "sigma", "rho"]
)

plt.savefig(prefix + "diagnostic_pairplot.png")

plt.close()

posterior_samples = simulator.sample(1000)

print(posterior_samples["price_path"].shape)

paths = posterior_samples['price_path']
price_paths = []

for path in paths:
    plt.plot(path)

plt.savefig(prefix + "_paths.png")

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .as_time_series("price_path")
    .standardize(include="price_path", mean=0, std=0.004)
    .standardize(include="rho", mean=-0.307, std=0.1435)
    .concatenate(["v0", "sigma", "rho"], into="inference_variables")
    .rename("price_path", "summary_variables")
)

adapted_sims = adapter(simulator.sample(100))

for k, v in adapted_sims.items():
    print(k, v.shape)

summary_net = bf.networks.TimeSeriesNetwork(dropout=0.1)

inference_net = bf.networks.FlowMatching(dropout=0.1)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    summary_network=summary_net,
    inference_network=inference_net,
    checkpoint_filepath="checkpoints/hsv_workflow",
    checkpoint_name="hsv_logpath_tsn_dm"
)

# train = workflow.simulate(10000)
validation = workflow.simulate(300)

history = workflow.fit_online(
    epochs=500,
    num_batches_per_epoch=100,
    batch_size=32,
    validation_data=validation
)

f = bf.diagnostics.plots.loss(history)
plt.plot()

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

plt.savefig(prefix + "_recoveries.png")

labels = ["v0", "sigma", "rho"]

truths = np.asarray([test_sims[labels[x]][0].item() for x in range(len(labels))])

out_samples = np.asarray([samples[labels[x]][0].flatten() for x in range(len(labels))]).T

d = out_samples.shape[1]
fig, axes = plt.subplots(d, d, figsize=(8, 8))

for i in range(d):
    for j in range(d):
        ax = axes[i, j]
        if i == j:
            ax.set_facecolor("white")  # set background blue
            ax.hist(out_samples[:, i], bins=40, histtype="step", color="lightblue")
            ax.axvline(truths[i], color="red")
            ax.set_xlabel(labels[i])
        elif i < j:
            ax.set_facecolor("midnightblue")  # set background blue
            h = ax.hist2d(out_samples[:, j], out_samples[:, i],
                          bins=50, cmap="viridis")
            ax.plot(truths[j], truths[i], "o", color="red")
        else:
            ax.axis("off")

plt.tight_layout()

plt.savefig(prefix + "_posterior_plot.png")