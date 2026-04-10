import numpy as np
from scipy.special import gamma
from utils import bs, bsinv


class rHeston(object):
    """
    Class for pricing under the rough Heston model of El Euch & Rosenbaum (2016).

    The model is defined by the log-price process:
        dS_t = S_t sqrt(V_t) dW_t
        V_t  = V0 + (1/Gamma(alpha)) * int_0^t (t-s)^(alpha-1) * lambda*(theta - V_s) ds
                   + (lambda*nu/Gamma(alpha)) * int_0^t (t-s)^(alpha-1) * sqrt(V_s) dB_s

    where <dW_t, dB_t> = rho dt and alpha = H + 1/2 in (1/2, 1), H in (0, 1/2).

    The characteristic function of X_t = log(S_t/S_0) is (Theorem 4.1):
        E[exp(ia X_t)] = exp( theta*lambda * I^1(h)(t) + V0 * I^(1-alpha)(h)(t) )

    where h solves the fractional Riccati equation (Section 5):
        D^alpha h = F(a, h),  F(a,x) = (1/2)(-a^2 - ia) + lambda*(ia*rho*nu - 1)*x + (lambda*nu)^2/2 * x^2
        I^(1-alpha) h(0) = 0

    solved via the fractional Adams predictor-corrector scheme (Section 5.1).

    References
    ----------
    El Euch, O. & Rosenbaum, M. (2016). "The characteristic function of rough Heston
    models." arXiv:1609.02108.
    """

    def __init__(self, n=100, N=1000, T=1.0, alpha=0.6,
                 lambda_=1.0, rho=-0.7, nu=0.3, theta=0.04, V0=0.04):
        """
        Constructor for class.

        Parameters
        ----------
        n       : int   - Time steps per year (granularity).
        N       : int   - Number of Monte Carlo paths.
        T       : float - Maturity in years.
        alpha   : float - Roughness parameter in (1/2, 1). Hurst index H = alpha - 1/2.
        lambda_ : float - Mean-reversion speed.
        rho     : float - Correlation between price and vol Brownians.
                          Must satisfy rho in (-1/sqrt(2), 1/sqrt(2)] per Theorem 4.1.
        nu      : float - Vol-of-vol parameter.
        theta   : float - Long-run variance (mean-reversion level).
        V0      : float - Initial variance.
        """
        # Basic assignments
        self.T       = T
        self.n       = n
        self.dt      = 1.0 / n
        self.s       = int(n * T)          # Total number of steps
        self.t       = np.linspace(0, T, 1 + self.s)
        self.N       = N

        # Model parameters
        self.alpha   = alpha               # alpha = H + 1/2
        self.lambda_ = lambda_
        self.rho     = rho
        self.nu      = nu
        self.theta   = theta
        self.V0      = V0

    # ------------------------------------------------------------------
    # Characteristic function via fractional Riccati equation
    # ------------------------------------------------------------------

    def F(self, a, x):
        """
        RHS of the fractional Riccati equation (Section 5):
            F(a, x) = (1/2)(-a^2 - ia) + lambda*(ia*rho*nu - 1)*x + (lambda*nu)^2/2 * x^2
        """
        lam = self.lambda_
        rho = self.rho
        nu  = self.nu
        return 0.5 * (-a**2 - 1j * a) + lam * (1j * a * rho * nu - 1) * x \
               + 0.5 * (lam * nu)**2 * x**2

    def _adams_weights(self, k, alpha):
        """
        Corrector weights a_{j,k+1} for the fractional Adams scheme (Eq. 27).

        For j = 0:
            a_{0,k+1} = Delta^alpha / Gamma(alpha+2) * (k^(alpha+1) - (k-alpha)*(k+1)^alpha)
        For 1 <= j <= k:
            a_{j,k+1} = Delta^alpha / Gamma(alpha+2) * ((k-j+2)^(alpha+1) + (k-j)^(alpha+1)
                                                         - 2*(k-j+1)^(alpha+1))
        For j = k+1 (diagonal):
            a_{k+1,k+1} = Delta^alpha / Gamma(alpha+2)
        """
        dt    = self.dt
        G2    = gamma(alpha + 2)
        scale = dt**alpha / G2

        j = np.arange(0, k + 1, dtype=float)

        a = np.where(
            j == 0,
            scale * (k**(alpha + 1) - (k - alpha) * (k + 1)**alpha),
            scale * ((k - j + 2)**(alpha + 1) + (k - j)**(alpha + 1)
                     - 2 * (k - j + 1)**(alpha + 1))
        )
        a_diag = scale  # a_{k+1, k+1}
        return a, a_diag

    def _predictor_weights(self, k, alpha):
        """
        Predictor weights b_{j,k+1} for the fractional Adams scheme (Section 5.1):
            b_{j,k+1} = Delta^alpha / Gamma(alpha+1) * ((k-j+1)^alpha - (k-j)^alpha)
        """
        dt    = self.dt
        G1    = gamma(alpha + 1)
        scale = dt**alpha / G1
        j     = np.arange(0, k + 1, dtype=float)
        return scale * ((k - j + 1)**alpha - (k - j)**alpha)

    def solve_riccati(self, a_values, t_grid=None):
        """
        Solve the fractional Riccati equation for h(a, t) using the
        fractional Adams predictor-corrector method (Section 5.1).

        h solves the Volterra equation (Eq. 25):
            h(a,t) = (1/Gamma(alpha)) * int_0^t (t-s)^(alpha-1) F(a, h(a,s)) ds

        Parameters
        ----------
        a_values : array-like - Characteristic function arguments (real or complex).
        t_grid   : array-like or None - Time grid; defaults to self.t.

        Returns
        -------
        h : ndarray, shape (len(a_values), len(t_grid))
            Solution h(a, t_j) for each a and time node t_j.
        """
        if t_grid is None:
            t_grid = self.t

        m     = len(t_grid) - 1          # Number of steps
        alpha = self.alpha
        a_arr = np.atleast_1d(np.asarray(a_values, dtype=complex))
        na    = len(a_arr)

        h = np.zeros((na, m + 1), dtype=complex)  # h[:,0] = 0 (initial condition)
        F_vals = np.zeros((na, m + 1), dtype=complex)
        F_vals[:, 0] = self.F(a_arr, h[:, 0])

        for k in range(m):
            # --- Predictor step (Riemann / rectangle rule) ---
            b = self._predictor_weights(k, alpha)             # shape (k+1,)
            h_pred = np.dot(F_vals[:, :k + 1], b[::-1])      # (na,) dot (k+1,) via convolution order

            # --- Corrector step (trapezoidal rule) ---
            a_w, a_diag = self._adams_weights(k, alpha)
            h_corr = np.dot(F_vals[:, :k + 1], a_w[::-1]) \
                     + a_diag * self.F(a_arr, h_pred)

            h[:, k + 1] = h_corr
            F_vals[:, k + 1] = self.F(a_arr, h_corr)

        return h

    def char_func(self, a_values, T=None):
        """
        Characteristic function of the log-price X_T = log(S_T / S_0).

        From Theorem 4.1 (Eq. 23):
            E[exp(ia X_T)] = exp( theta*lambda * I^1(h)(T) + V0 * I^(1-alpha)(h)(T) )

        where:
            I^1(h)(T)       = int_0^T h(a,s) ds               (plain integral)
            I^(1-alpha)(h)(T) = (1/Gamma(1-alpha)) * int_0^T (T-s)^(-alpha) h(a,s) ds
                                                               (fractional integral of order 1-alpha)

        Parameters
        ----------
        a_values : array-like - Real or complex arguments.
        T        : float or None - Maturity; defaults to self.T.

        Returns
        -------
        cf : ndarray - Complex characteristic function values, shape (len(a_values),).
        """
        if T is None:
            T = self.T

        # Build a fine time grid up to T
        m      = self.s
        t_grid = np.linspace(0, T, m + 1)
        dt     = T / m
        alpha  = self.alpha

        h = self.solve_riccati(a_values, t_grid=t_grid)   # (na, m+1)

        # I^1(h)(T) = int_0^T h ds  via trapezoid rule
        I1 = np.trapezoid(h, dx=dt, axis=1)

        # I^(1-alpha)(h)(T) = 1/Gamma(1-alpha) * int_0^T (T-s)^(-alpha) h(s) ds
        # Kernel evaluated at interior points (skip s=T to avoid (T-T)^{-alpha} = inf)
        s_grid  = t_grid[:-1]                              # s in [0, T)
        kernel  = (T - s_grid)**(-alpha)                   # shape (m,)
        # Rectangle rule (left-endpoint) to avoid the singularity at s = T
        I1a = (1.0 / gamma(1.0 - alpha)) * np.sum(h[:, :-1] * kernel, axis=1) * dt

        lam   = self.lambda_
        theta = self.theta
        V0    = self.V0

        exponent = theta * lam * I1 + V0 * I1a
        return np.exp(exponent)

    # ------------------------------------------------------------------
    # Monte Carlo simulation (Euler-Maruyama on the Volterra SDE)
    # ------------------------------------------------------------------

    def dW(self):
        """
        Correlated Brownian increments (dW for price, dB for variance).

        Returns
        -------
        dW1 : ndarray, shape (N, s) - Increments driving the variance process.
        dW2 : ndarray, shape (N, s) - Independent increments for the price.
        """
        sqrt_dt = np.sqrt(self.dt)
        dW1 = np.random.randn(self.N, self.s) * sqrt_dt  # dB (variance BM)
        dW2 = np.random.randn(self.N, self.s) * sqrt_dt  # orthogonal component
        return dW1, dW2

    def dB_price(self, dW1, dW2):
        """
        Correlated price Brownian increments:
            dW = rho * dB + sqrt(1 - rho^2) * dW_perp

        Parameters
        ----------
        dW1 : ndarray, shape (N, s) - Variance BM increments (dB).
        dW2 : ndarray, shape (N, s) - Independent increments.

        Returns
        -------
        dW_price : ndarray, shape (N, s)
        """
        rho = self.rho
        return rho * dW1 + np.sqrt(1.0 - rho**2) * dW2

    def V(self, dW1):
        """
        Simulate the rough variance process V via an Euler-Maruyama discretisation
        of the Volterra SDE (Eq. 3).

        The fractional integral is approximated by a direct Riemann convolution:
            V_{t_{k+1}} = V0
                + (1/Gamma(alpha)) sum_{j=0}^{k} (t_{k+1} - t_j)^{alpha-1}
                                                  * lambda*(theta - V_{t_j}) * dt
                + (lambda*nu/Gamma(alpha)) sum_{j=0}^{k} (t_{k+1} - t_j)^{alpha-1}
                                                          * sqrt(V_{t_j}) * dB_{t_j}

        The kernel weight for the j-th interval uses the midpoint approximation
        (t_{k+1} - t_{j+0.5})^{alpha-1} to reduce bias, consistent with the hybrid
        scheme philosophy of the companion rBergomi code.

        Parameters
        ----------
        dW1 : ndarray, shape (N, s) - Variance BM increments.

        Returns
        -------
        V : ndarray, shape (N, 1+s) - Variance paths on the full time grid.
        """
        alpha  = self.alpha
        lam    = self.lambda_
        nu     = self.nu
        theta  = self.theta
        V0     = self.V0
        dt     = self.dt
        N      = self.N
        s      = self.s
        G_a    = gamma(alpha)

        V_path = np.zeros((N, 1 + s))
        V_path[:, 0] = V0

        # Pre-compute kernel weights: w_k = (k+1)^{alpha-1} * dt for k=0,...,s-1
        # kernel[k] is the weight for the lag of (k+1) steps, i.e. (t - t_j) = (k+1)*dt
        k_idx   = np.arange(1, s + 1, dtype=float)
        kernel  = k_idx**(alpha - 1) * dt / G_a   # shape (s,)

        for i in range(1, 1 + s):
            # Lags: for step i, the j-th term has lag (i - j) steps, j = 0,...,i-1
            lags        = np.arange(i, 0, -1) - 1          # [i-1, i-2, ..., 0]
            w           = kernel[lags]                       # shape (i,)

            V_prev      = V_path[:, :i]                     # (N, i)
            drift       = lam * (theta - V_prev)            # (N, i)
            diffusion   = lam * nu * np.sqrt(np.maximum(V_prev, 0.0))  # (N, i)

            V_path[:, i] = (V0
                            + np.dot(drift, w)
                            + np.einsum('ij,j->i', diffusion * dW1[:, :i], w))

            # Reflect at zero to keep variance non-negative
            V_path[:, i] = np.maximum(V_path[:, i], 0.0)

        return V_path

    def S(self, V, dW_price, S0=1.0):
        """
        Simulate the rough Heston price process via Euler-Maruyama.

        Log-price increments (Eq. 3):
            d log S_t = sqrt(V_t) dW_t - (1/2) V_t dt

        Parameters
        ----------
        V         : ndarray, shape (N, 1+s) - Variance paths from self.V().
        dW_price  : ndarray, shape (N, s)   - Correlated price BM increments.
        S0        : float                   - Initial asset price.

        Returns
        -------
        S : ndarray, shape (N, 1+s) - Asset price paths.
        """
        dt  = self.dt
        log_increments = np.sqrt(V[:, :-1]) * dW_price - 0.5 * V[:, :-1] * dt
        log_S = np.zeros_like(V)
        log_S[:, 0] = np.log(S0)
        log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)
        return np.exp(log_S)

    def call_price_mc(self, K, S0=1.0):
        """
        European call price via Monte Carlo.

        Parameters
        ----------
        K  : float or array-like - Strike(s).
        S0 : float               - Initial spot.

        Returns
        -------
        price : float or ndarray - Discounted (zero rate) call price E[(S_T - K)+].
        """
        dW1, dW2   = self.dW()
        dW_p       = self.dB_price(dW1, dW2)
        V_paths    = self.V(dW1)
        S_paths    = self.S(V_paths, dW_p, S0=S0)
        S_T        = S_paths[:, -1]

        K = np.atleast_1d(K)
        payoffs = np.maximum(S_T[:, np.newaxis] - K[np.newaxis, :], 0.0)
        return payoffs.mean(axis=0).squeeze()

    def impl_vol_mc(self, K, S0=1.0):
        """
        Implied Black-Scholes volatility from Monte Carlo call prices.

        Parameters
        ----------
        K  : float or array-like - Strike(s).
        S0 : float               - Initial spot price.

        Returns
        -------
        ivol : float or ndarray - Implied vol(s).
        """
        K      = np.atleast_1d(np.asarray(K, dtype=float))
        prices = np.atleast_1d(self.call_price_mc(K, S0=S0))
        T      = self.T

        ivols = np.array([bsinv(p, S0, k, T) for p, k in zip(prices, K)])
        return ivols.squeeze()
