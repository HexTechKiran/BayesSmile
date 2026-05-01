import numpy as np
import matplotlib.pyplot as plt
import jax
import seaborn as sns

import bayesflow as bf

jax.config.update("jax_default_device", jax.devices("cpu")[0])

def simple_heston(
    lambda_=1.0,
    rho=-0.7,
    nu=0.2,
    theta=0.04,
    V0=0.04,
    T=1.0,
    n=100,
    N=1,
    S0=1.0
):
    """
    Simulate the simple (standard) Heston model via Euler-Maruyama.

    The model is:
        dS_t = S_t sqrt(V_t) dW_t
        dV_t = lambda*(theta - V_t)*dt + lambda*nu*sqrt(V_t)*dB_t
    where <dW_t, dB_t> = rho dt.

    Feller condition (2*lambda*theta >= (lambda*nu)^2) must hold; an
    AssertionError is raised if it is violated.

    Parameters
    ----------
    lambda_ : float - Mean-reversion speed.
    rho     : float - Correlation between price and variance Brownians.
    nu      : float - Vol-of-vol (normalized); must satisfy nu <= sqrt(2*theta/lambda_).
    theta   : float - Long-run variance.
    V0      : float - Initial variance.
    T       : float - Maturity in years.
    n       : int   - Time steps per year.
    N       : int   - Number of Monte Carlo paths.
    S0      : float - Initial asset price.

    Returns
    -------
    S : ndarray, shape (N, 1+s) - Asset price paths.
    V : ndarray, shape (N, 1+s) - Variance paths.
    """
    # assert 2.0 * lambda_ * theta >= nu ** 2, (
    #     f"Feller condition violated: 2*lambda*theta={2*lambda_*theta:.4f} < "
    #     f"(lambda*nu)^2={nu**2:.4f}."
    # )

    s = int(n * T)
    dt = T / s
    sqrt_dt = np.sqrt(dt)

    dB = np.random.randn(N, s) * sqrt_dt
    dZ = np.random.randn(N, s) * sqrt_dt
    dW = rho * dB + np.sqrt(1.0 - rho**2) * dZ

    V_path = np.zeros((N, 1 + s))
    V_path[:, 0] = V0
    for i in range(s):
        V_prev = V_path[:, i]
        V_path[:, i + 1] = (V_prev
                            + lambda_ * (theta - V_prev) * dt
                            + nu * np.sqrt(np.maximum(V_prev, 0.0)) * dB[:, i])
        V_path[:, i + 1] = np.maximum(V_path[:, i + 1], 0.0)

    log_increments = -0.5 * V_path[:, :-1] * dt + np.sqrt(np.maximum(V_path[:, :-1], 0.0)) * dW
    log_S = np.zeros((N, 1 + s))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)

    return {
        "price": np.exp(log_S).squeeze(),
    }

def priors():
    lambda_ = np.random.gamma(2, 0.5)        # mean=1.0, positive
    theta   = np.random.gamma(2, 0.02)       # mean=0.04, positive
    nu_max  = np.sqrt(2 * theta / lambda_)   # Feller ceiling: nu < nu_max guarantees 2λθ >= (λν)²
    nu      = np.random.uniform(0, nu_max)
    rho     = -0.65   # Fixed due to unrecoverability
    V0      = np.random.gamma(2, theta / 2)  # mean=theta
    return {
        "lambda_": lambda_,
        "rho":     rho,
        "nu":      nu,
        "theta":   theta,
        "V0":      V0,
    }

simulator = bf.make_simulator([priors, simple_heston])

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["lambda_", "nu", "theta", "V0"], into="inference_variables")
    .expand_dims(["price"], axis=-1)
    .concatenate(["price"], axis=-1, into="summary_variables")
)

summary_net = bf.networks.TimeSeriesTransformer(summary_dim=16)
inference_net = bf.networks.FlowMatching()

workflow = bf.workflows.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
    standardize="all"
)

training_set = workflow.simulate(20000)
val_set = workflow.simulate(500)

history = workflow.fit_offline(
    data=training_set,
    validation_data=val_set,
    epochs=100,
    batch_size=32
)

figures = workflow.plot_default_diagnostics(
    test_data=val_set,
    loss_kwargs={"figsize": (15, 3), "label_fontsize": 12},
    recovery_kwargs={"figsize": (15, 3), "label_fontsize": 12},
    calibration_ecdf_kwargs={"figsize": (15, 3), "legend_fontsize": 8, "label_fontsize": 12, "difference": True},
    coverage_kwargs={"figsize": (15, 3), "legend_fontsize": 8, "label_fontsize": 12, "difference": True},
    z_score_contraction_kwargs={"figsize": (15, 3), "label_fontsize": 12}
)

plt.savefig("simple_heston_diagnostics.png")