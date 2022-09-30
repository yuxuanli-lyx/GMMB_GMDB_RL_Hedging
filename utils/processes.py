# REFERENCE: https://jtsulliv.github.io/stock-movement/
import numpy as np


def pure_brownian_motion(dt: float, n_steps: int) -> np.array:
    """Generate one sample path of pure Brownian motion.

    .. math::
        W_t

    Parameters
    ----------
    dt: `float`
        step size of each time increment
    n_steps: `int`
        number of steps to simulate into the future

    Returns
    -------
    `np.array` of size (`n_steps` + 1,)
        a simulated path of pure Brownian motion with initial value 0
    """
    # increments
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=n_steps)
    # sample path
    Wt = np.cumsum(dW)
    Wt = np.insert(Wt, 0, 0)  # Wt starts with 0
    return Wt


def geometric_brownian_motion(
    S: float,
    mu: float,
    sigma: float,
    tau: float,
    n_steps: int,
) -> np.array:
    """Generate one sample path of geometric Brownian motion.

    .. math::
        dS_t &= \mu S_t dt + \sigma S_t dW_t \\\\
         S_t &= S_0 \exp( (\mu - \\frac{1}{2} \sigma^2) t + \sigma W_t )

    Parameters
    ----------
    S: `float`
        initial asset price, :math:`S_0`
    mu: `float`
        instantaneous return, in percent per year, of asset price, :math:`\mu`
    sigma: `float`
        instantaneous volatility, in percent per year, of asset price, :math:`\sigma`
    tau: `float`
        time to maturity, in years, :math:`\\tau = T - t`
    n_steps: `int`
        number of steps to simulate into the future

    Returns
    -------
    `np.array` of size (`n_steps` + 1,)
        a simulated path of geometric Brownian motion with initial value `S` (i.e. :math:`S_0`)
    """
    dt = tau / n_steps
    timeline = np.arange(n_steps + 1) * dt
    Wt = pure_brownian_motion(dt=dt, n_steps=n_steps)
    return S * np.exp((mu - 0.5 * (sigma**2)) * timeline + sigma * Wt)
