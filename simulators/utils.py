import numpy as np


def correlated_normals(rho: float, n: int, rng: np.random.Generator):
    Z1 = rng.standard_normal(n)
    Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    return Z1, Z2


def fgn_davies_harte(H: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Fractional Gaussian Noise via Davies-Harte algorithm.

    Multiply the result by dt**H to get fBm increments with variance dt**(2H).
    """
    if n == 1:
        return rng.standard_normal(1)

    k = np.arange(n, dtype=float)
    gamma = 0.5 * (
        (k + 1) ** (2 * H)
        - 2.0 * k ** (2 * H)
        + np.maximum(k - 1, 0.0) ** (2 * H)
    )

    m = 2 * (n - 1)
    c = np.empty(m)
    c[:n] = gamma
    c[n:] = gamma[n - 2 : 0 : -1]

    lam = np.maximum(np.real(np.fft.fft(c)), 0.0)

    W = rng.standard_normal(m)
    V = rng.standard_normal(m)

    xi = np.zeros(m, dtype=complex)
    xi[0] = np.sqrt(lam[0] / m) * W[0]
    xi[m // 2] = np.sqrt(lam[m // 2] / m) * W[m // 2]
    idx = np.arange(1, m // 2)
    scale = np.sqrt(lam[idx] / (2 * m))
    xi[idx] = scale * (W[idx] + 1j * V[idx])
    xi[m - idx] = np.conj(xi[idx])

    return np.real(np.fft.ifft(xi))[:n] * m
