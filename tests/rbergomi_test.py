import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import bayesflow as bf
import jax
import matplotlib.pyplot as plt
from simulators.rbergomi import rBergomi

jax.config.update('jax_default_device', jax.devices('cpu')[0])

def draw_rb_priors() :
    rho = (np.random.beta(3, 3) * 0.8 + 0.1) * (-1.0)
    eta = np.random.beta(5, 5) + 2
    xi = 0.235**2
    return dict(rho=float(rho), eta=float(eta), xi=float(xi))

def sample_bergomi(rho, eta, xi, n = 252, N = 1, T = 1.0, a = -0.43, seed = 1234) :
    rB = rBergomi(n = n, N = N, T = T, a = a)

    #np.random.seed(seed)

    dW1 = rB.dW1()
    dW2 = rB.dW2()

    Y = rB.Y(dW1)

    dB = rB.dB(dW1, dW2, rho = rho)

    V = rB.V(Y, xi = xi, eta = eta)

    S = rB.S(V, dB, S0 = 1)[0]

    #log_returns = np.log(S[1:] / S[:-1])

    return dict(log_returns=S)

simulator = bf.make_simulator([draw_rb_priors, sample_bergomi])

prior_samples = simulator.simulators[0].sample(1000)

grid = bf.diagnostics.plots.pairs_samples(
    prior_samples, variable_keys=["rho", "eta"]
)

plt.savefig("rbergomi_diagnostic_pairplot.png")

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .as_time_series("log_returns")
    .standardize(include="rho", mean=-0.307, std=0.1435)
    .concatenate(["rho", "eta"], into="inference_variables")
    .rename("log_returns", "summary_variables")
)
#.standardize(include="log_returns", mean=0, std=0.004)

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
plt.savefig("rbergomi_loss.png")

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

plt.savefig("rbergomi_recoveries.png")

labels = ["rho", "eta"]

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

plt.savefig("rbergomi_posterior_plot.png")

workflow.approximator.save("rbergomi_model.keras")

"""
H = 0.1 # Hurst parameter
n = 1000 # Num steps
m = 1000 # Num RL process draws
gamma = np.array([[1/n, 1/((H + 0.5) * n ^ (H + 0.5))],
                  [1/((H + 0.5) * n ^ (H + 0.5)), 1/(2*H*n^(2*H))]]) # Covariance matrix
k = 1 # Integer parameter for order of Hybrid Scheme Riemann-Liouville process



def RL_process(n, k, K, H, gamma):
    W = np.empty((n + 1, 2))
    Y = np.empty(n + 1)

    def b_k(K):
        return (K**(H + 0.5) - (K - 1)**(H + 0.5))/(H + 0.5)**(1 / (H - 0.5))

    # MODIFY THIS TO USE FFT
    for i in range(n + 1):
        W[i] = np.random.multivariate_normal(np.zeros(2), gamma)
        coefficient = W[max(i - 1, 0), 0]
        if (i > k) :
            coefficient += sum( ((b_k(K)/n)**(H - 0.5)) * W[i - K, 1] for K in range(k+1, i) )
        Y[i] = np.sqrt(2 * H) * coefficient

    return (Y, W)
"""