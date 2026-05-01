import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import bayesflow as bf
import jax
import matplotlib.pyplot as plt
from simulators.rough_heston import rough_heston_model as rHeston

jax.config.update("jax_default_device", jax.devices("cpu")[0])

N_STEPS  = 100
T        = 1.0

LAMBDA   = 2.0
THETA    = 0.04
V0       = 0.04

def draw_rh_priors():
    rho   = -(np.random.beta(2, 2) / np.sqrt(2))           # in (-1/sqrt(2), 0)
    xi    = float(np.clip(np.random.gamma(4, 0.125), 0.05, 2.0))
    H = float(np.random.beta(2, 5) * 0.30 + 0.55) - 0.5      # in (0.05, 0.35), mode ~0.10
    return dict(rho=float(rho), xi=float(xi), H=float(H))

def sample_rh(rho, xi, H, n=N_STEPS, T=T, N_paths=1):
    sample = rHeston(0.05,
                     np.log(2) / LAMBDA,
                     THETA,
                     xi,
                     rho,
                     H,
                     V0)

    return sample

simulator = bf.make_simulator([draw_rh_priors, sample_rh])

print("Drawing prior samples for diagnostic pairplot...")
prior_samples = simulator.simulators[0].sample(2000)

grid = bf.diagnostics.pairs_samples(
    prior_samples,
    variable_keys=["rho", "xi", "H"],
    variable_names=[r"$\rho$", r"$\xi$", r"$\H$"],
)
grid.figure.suptitle("Rough Heston prior samples")
plt.savefig("rheston_prior_pairplot.png")
plt.close()
print("  Saved rheston_prior_pairplot.png")

output_samples = simulator.sample(1000)['log_returns']

for path in output_samples:
    # prices = 100 * np.exp(np.cumsum(np.concatenate([[0], path])))
    plt.plot(path)

grid.figure.suptitle("Rough Heston Output samples")
plt.savefig("rheston_output_samples.png")
plt.close()

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["rho", "xi", "H"], into="inference_variables")
    .rename("log_returns", "inference_conditions")
)

#summary_net = bf.networks.TimeSeriesNetwork(dropout=0.05)
inference_net = bf.networks.StableConsistencyModel()

workflow = bf.BasicWorkflow(
    simulator        = simulator,
    adapter          = adapter,
    # summary_network  = summary_net,
    inference_network= inference_net,
    checkpoint_filepath ="../rheston_workflow/",
)

print("Simulating training data (10 000 draws)...")
train = workflow.simulate((5_000,))

print("Simulating validation data (500 draws)...")
validation = workflow.simulate((300,))

print("Training the amortised posterior...")
history = workflow.fit_offline(
    data            = train,
    epochs          = 50,
    batch_size      = 32,
    validation_data = validation,
)

fig = bf.diagnostics.loss(history)
plt.savefig("rheston_loss.png")
plt.close()
print("  Saved rheston_loss.png")

NUM_TEST    = 300
NUM_SAMPLES = 300

print(f"Simulating {NUM_TEST} test scenarios...")
test_sims = workflow.simulate((NUM_TEST,))

print(f"Drawing {NUM_SAMPLES} posterior samples per scenario...")
samples = workflow.sample(conditions=test_sims, num_samples=NUM_SAMPLES)

fig = bf.diagnostics.recovery(
    estimates       = samples,
    targets         = test_sims,
    variable_keys   = ["rho", "xi", "H"],
    variable_names  = [r"$\rho$", r"$\xi$", r"$\H$"],
)
plt.savefig("rheston_recovery.png")
plt.close()
print("  Saved rheston_recovery.png")

labels     = ["rho", "xi", "H"]
tex_labels = [r"$\rho$", r"$\xi$", r"$\H$"]

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
    rf"true $\rho$={truths[0]:.3f}, $\xi$={truths[1]:.3f}, $\H$={truths[2]:.3f}",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig("rheston_posterior.png")
plt.close()
print("  Saved rheston_posterior.png")

workflow.approximator.save("rheston_model.keras")
print("  Saved rheston_model.keras")

print("\nDone.")