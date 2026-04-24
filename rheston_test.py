import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import bayesflow as bf
import jax
import matplotlib.pyplot as plt
from rHeston import rHeston

jax.config.update("jax_default_device", jax.devices("cpu")[0])

N_STEPS  = 500
T        = 1.0

LAMBDA   = 2.0
THETA    = 0.04
V0       = 0.04

def draw_rh_priors():
    rho   = -(np.random.beta(2, 2) / np.sqrt(2))           # in (-1/sqrt(2), 0)
    nu    = float(np.clip(np.random.gamma(4, 0.125), 0.05, 2.0))
    #alpha = float(np.random.beta(4, 6) * 0.48 + 0.51)      # in (0.51, 0.99)
    alpha = 0.99
    return dict(rho=float(rho), nu=float(nu), alpha=float(alpha))

def sample_rh(rho, nu, alpha, n=N_STEPS, T=T):
    rh = rHeston(
        n       = n,
        N       = 1,           # single path per simulation draw
        T       = T,
        alpha   = alpha,
        lambda_ = LAMBDA,
        rho     = rho,
        nu      = nu,
        theta   = THETA,
        V0      = V0,
    )

    dW1, dW2 = rh.dW()
    V_path   = rh.V(dW1)
    dW_p     = rh.dB_price(dW1, dW2)
    S_path   = rh.S(V_path, dW_p, S0=1.0)[0]   # shape (n+1,), take first (only) path

    #log_returns = np.log(S_path[1:] / S_path[:-1]).astype(np.float32)  # shape (n,)
    log_returns = S_path
    return dict(log_returns=log_returns)

simulator = bf.make_simulator([draw_rh_priors, sample_rh])

print("Drawing prior samples for diagnostic pairplot...")
prior_samples = simulator.simulators[0].sample(2000)

grid = bf.diagnostics.pairs_samples(
    prior_samples,
    variable_keys=["rho", "nu", "alpha"],
    variable_names=[r"$\rho$", r"$\nu$", r"$\alpha$"],
)
grid.figure.suptitle("Rough Heston prior samples")
plt.savefig("rheston_prior_pairplot.png")
plt.close()
print("  Saved rheston_prior_pairplot.png")

output_samples = simulator.sample(1000)['log_returns']

for path in output_samples:
    #prices = 100 * np.exp(np.cumsum(np.concatenate([[0], path])))
    plt.plot(path)


grid.figure.suptitle("Rough Heston Output samples")
plt.savefig("rheston_output_samples_a99.png")
plt.close()

"""

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .as_time_series("log_returns")                          # (T, 1) time-series tensor
    .standardize(include="rho",   mean=-0.354, std=0.158)
    .standardize(include="nu",    mean=0.500,  std=0.249)
    .standardize(include="alpha", mean=0.702,  std=0.071)
    .concatenate(["rho", "nu", "alpha"], into="inference_variables")
    .rename("log_returns", "summary_variables")
)

summary_net = bf.networks.TimeSeriesNetwork(dropout=0.05)
inference_net = bf.networks.DiffusionModel(dropout=0.05)

workflow = bf.BasicWorkflow(
    simulator        = simulator,
    adapter          = adapter,
    summary_network  = summary_net,
    inference_network= inference_net,
    checkpoint_filepath = "rheston_workflow/",
)

print("Simulating training data (10 000 draws)...")
train = workflow.simulate((10_000,))

print("Simulating validation data (500 draws)...")
validation = workflow.simulate((500,))

print("Training the amortised posterior...")
history = workflow.fit_offline(
    data            = train,
    epochs          = 100,
    batch_size      = 32,
    validation_data = validation,
)

fig = bf.diagnostics.loss(history)
plt.savefig("rheston_loss.png")
plt.close()
print("  Saved rheston_loss.png")

NUM_TEST    = 300
NUM_SAMPLES = 1000

print(f"Simulating {NUM_TEST} test scenarios...")
test_sims = workflow.simulate((NUM_TEST,))

print(f"Drawing {NUM_SAMPLES} posterior samples per scenario...")
samples = workflow.sample(conditions=test_sims, num_samples=NUM_SAMPLES)

fig = bf.diagnostics.recovery(
    estimates       = samples,
    targets         = test_sims,
    variable_keys   = ["rho", "nu", "alpha"],
    variable_names  = [r"$\rho$", r"$\nu$", r"$\alpha$"],
)
plt.savefig("rheston_recovery.png")
plt.close()
print("  Saved rheston_recovery.png")

labels     = ["rho", "nu", "alpha"]
tex_labels = [r"$\rho$", r"$\nu$", r"$\alpha$"]

truths     = np.array([float(test_sims[k][0]) for k in labels])
out_samples = np.column_stack([samples[k][0].flatten() for k in labels])  # (num_samples, 3)

d   = len(labels)
fig, axes = plt.subplots(d, d, figsize=(9, 9))

for i in range(d):
    for j in range(d):
        ax = axes[i, j]
        if i == j:
            ax.set_facecolor("white")
            ax.hist(out_samples[:, i], bins=40, histtype="step", color="#132a70", linewidth=1.5)
            ax.axvline(truths[i], color="crimson", linewidth=2, label="truth")
            ax.set_xlabel(tex_labels[i], fontsize=13)
            ax.tick_params(labelsize=9)
            if i == 0:
                ax.legend(fontsize=10)
        elif i < j:
            ax.set_facecolor("midnightblue")
            ax.hist2d(out_samples[:, j], out_samples[:, i], bins=50, cmap="viridis")
            ax.plot(truths[j], truths[i], "o", color="crimson", markersize=7)
            ax.tick_params(labelsize=8)
        else:
            ax.axis("off")

fig.suptitle(
    f"Rough Heston posterior  |  "
    rf"true $\rho$={truths[0]:.3f}, $\nu$={truths[1]:.3f}, $\alpha$={truths[2]:.3f}",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig("rheston_posterior.png")
plt.close()
print("  Saved rheston_posterior.png")

workflow.approximator.save("rheston_model.keras")
print("  Saved rheston_model.keras")

print("\nDone.")
"""