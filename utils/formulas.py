# this file consists of commonly used formulas for computing net liabilities and Deltas
from typing import Callable
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy import optimize


def integral_evaluator(
    func: Callable[[np.ndarray], np.ndarray],
    lb: float,
    ub: float,
    precision: int = 1000,
) -> np.ndarray:
    """
    Function evaluating the integration for theoretical GMDB rider net liability and Delta
    :param func: function to be evaluated (Note that it is a vector-valued function)
    :param lb: lower bound of integration
    :param ub: upper bound of integration
    :param precision: number of partitions to use (default = 1000)
    :return: the value of the component-wise integral
    """
    ds = 1 / precision
    grid_pts = np.linspace(lb, ub, precision + 1).reshape([precision + 1, 1])
    val_at_grid = func(grid_pts)
    val_integral = np.sum(ds * val_at_grid, axis=0)
    return val_integral


def fair_rate_bs_cfm(
    F0: float, G_m: float, G_d: float, mu: float, T: float, r: float, sigma: float
) -> float:
    """
    Function that evaluates the fair annual deduction rate m, with a underlying market model to be BS + CFM
    Note that by default, the rider charge rate is of 95% of the annual deduction rate, i.e. me = 0.95 * m

    :param F0: initial segregate account value
    :param G_m: guaranteed minimum maturity benefit
    :param G_d: guaranteed minimum death benefit
    :param mu: force of mortality
    :param T: maturity
    :param r: interest rate
    :param sigma: volatility
    :return: the fair annual deduction rate
    """
    func = lambda m: np.abs(
        net_liability_bs_cfm(F0, 0, G_m, G_d, mu, m, m * 0.95, T, r, sigma)
    )
    sol = optimize.minimize(func, 0.5, method="Nelder-Mead", tol=1e-6)
    return float(sol.x)


def net_liability_bs_cfm(
    Ft: np.ndarray,
    t: np.ndarray,
    G_m: float,
    G_d: float,
    mu: float,
    m: float,
    me: float,
    T: float,
    r: float,
    sigma: float,
) -> np.ndarray:
    """
    The function computes the net liability of the insurer for a generic policyholder, provided
    that this policyholder is alive at time t. The underlying market model is BS + CFM

    :param Ft: the segregate account value at time t (should have a shape  of (X, ))
    :param t: current time point (should have a shape of (X, ))
    :param G_m: guaranteed minimum maturity benefit
    :param G_d: guaranteed minimum death benefit
    :param mu: force of mortality
    :param m: account deduction fee
    :param me: rider charge rate
    :param T: maturity
    :param r: interest rate
    :param sigma: volatility
    :return: the net liability at time t
    """
    # margin to prevent numerical error
    _MARGIN = 1e-15

    # gross liability due to the GMDB rider
    d_1_d: Callable[[np.ndarray], np.ndarray] = lambda s: (
        np.log(Ft / G_d) + (r - m + (sigma**2 / 2) * (np.maximum(s, t) - t))
    ) / (sigma * np.sqrt(np.maximum(s, t) - t) + _MARGIN)
    d_2_d: Callable[[np.ndarray], np.ndarray] = lambda s: d_1_d(s) - sigma * np.sqrt(
        np.maximum(s, t) - t
    )
    V_d: Callable[[np.ndarray], np.ndarray] = lambda s: G_d * np.exp(
        -r * (np.maximum(s, t) - t)
    ) * norm.cdf(-d_2_d(s), 0, 1) - Ft * np.exp(-m * (np.maximum(s, t) - t)) * norm.cdf(
        -d_1_d(s), 0, 1
    )
    cfm_density: Callable[[np.ndarray], np.ndarray] = (
        lambda s: mu * np.exp(-mu * (s - t)) * (s > t)
    )
    integrand_d: Callable[[np.ndarray], np.ndarray] = lambda s: V_d(s) * cfm_density(s)
    GL_GMDB = integral_evaluator(integrand_d, 0, T)

    # gross liability due to the GMMB rider
    d_1_m = (np.log(Ft / G_m) + (r - m + (sigma**2 / 2) * (T - t))) / (
        sigma * np.sqrt(T - t) + _MARGIN
    )
    d_2_m = d_1_m - sigma * np.sqrt(T - t)
    GL_GMMB = np.exp(-mu * (T - t)) * (
        G_m * np.exp(-r * (T - t)) * norm.cdf(-d_2_m, 0, 1)
        - Ft * np.exp(-m * (T - t)) * norm.cdf(-d_1_m, 0, 1)
    )

    # rider charge
    RC = ((1 - np.exp(-(m + mu) * (T - t))) / (m + mu + _MARGIN)) * me * Ft

    # net liability
    NL = GL_GMDB + GL_GMMB - RC
    return NL


def delta_bs_cfm(
    Ft: np.ndarray,
    t: np.ndarray,
    G_m: float,
    G_d: float,
    mu: float,
    m: float,
    me: float,
    T: float,
    r: float,
    sigma: float,
    rho: float,
) -> np.ndarray:
    """
    The function computes the Delta per contract, provided
    that this policyholder is alive at time t. The underlying market model is BS + CFM
    :param Ft: the segregate account value at time t
    :param t: current time point
    :param G_m: guaranteed minimum maturity benefit
    :param G_d: guaranteed minimum death benefit
    :param mu: force of mortality
    :param m: account deduction fee
    :param me: rider charge rate
    :param T: maturity
    :param r: interest rate
    :param sigma: volatility
    :param rho: initial investment strategy
    :return: the Delta at time t
    """
    # margin to prevent numerical error
    _MARGIN = 1e-15

    # Delta from the GMDB rider (w.r.t Ft)
    d_1_d: Callable[[np.ndarray], np.ndarray] = lambda s: (
        np.log(Ft / G_d) + (r - m + (sigma**2 / 2) * (np.maximum(s, t) - t))
    ) / (sigma * np.sqrt(np.maximum(s, t) - t) + _MARGIN)
    V_d: Callable[[np.ndarray], np.ndarray] = lambda s: np.exp(-m * (s - t)) * (
        norm.cdf(d_1_d(s), 0, 1) - 1
    )
    cfm_density: Callable[[np.ndarray], np.ndarray] = (
        lambda s: mu * np.exp(-mu * (s - t)) * (s > t)
    )
    integrand_d: Callable[[np.ndarray], np.ndarray] = lambda s: V_d(s) * cfm_density(s)
    delta_GMDB = integral_evaluator(integrand_d, 0, T)

    # Delta from the GMMB rider (w.r.t Ft)
    d_1_m = (np.log(Ft / G_m) + (r - m + (sigma**2 / 2) * (T - t))) / (
        sigma * np.sqrt(T - t) + _MARGIN
    )
    delta_GMMB = np.exp(-(mu + m) * (T - t)) * (norm.cdf(d_1_m, 0, 1) - 1)

    # Delta from the rider charge (w.r.t Ft)
    delta_RC = ((1 - np.exp(-(m + mu) * (T - t))) / (m + mu + _MARGIN)) * me

    # compute the true Delta (w.r.t St)
    delta_true = (
        (delta_GMDB + delta_GMMB - delta_RC) * rho * np.exp(-m * t)
    )  # chain rule
    return delta_true


def delta_bs_ifm(
    Ft: np.ndarray,
    t: np.ndarray,
    G_m: float,
    G_d: float,
    ub: float,
    m: float,
    me: float,
    T: float,
    r: float,
    sigma: float,
    rho: float,
) -> np.ndarray:
    """
    The function computes the Delta per contract, provided
    that this policyholder is alive at time t. The underlying market model is BS + IFM
    :param Ft: the segregate account value at time t
    :param t: current time point
    :param G_m: guaranteed minimum maturity benefit
    :param G_d: guaranteed minimum death benefit
    :param ub: upper bound of the uniform lifetime distribution
    :param m: account deduction fee
    :param me: rider charge rate
    :param T: maturity
    :param r: interest rate
    :param sigma: volatility
    :param rho: initial investment strategy
    :return: the Delta at time t
    """
    # margin to prevent numerical error
    _MARGIN = 1e-15

    # Delta from the GMDB rider (w.r.t Ft)
    d_1_d: Callable[[np.ndarray], np.ndarray] = lambda s: (
        np.log(Ft / G_d) + (r - m + (sigma**2 / 2) * (np.maximum(s, t) - t))
    ) / (sigma * np.sqrt(np.maximum(s, t) - t) + _MARGIN)
    V_d: Callable[[np.ndarray], np.ndarray] = lambda s: np.exp(-m * (s - t)) * (
        norm.cdf(d_1_d(s), 0, 1) - 1
    )
    ifm_density: Callable[[np.ndarray], np.ndarray] = lambda s: (1 / (ub - t)) * (s > t)
    integrand_d: Callable[[np.ndarray], np.ndarray] = lambda s: V_d(s) * ifm_density(s)
    delta_GMDB = integral_evaluator(integrand_d, 0, T)

    # Delta from the GMMB rider (w.r.t Ft)
    d_1_m = (np.log(Ft / G_m) + (r - m + (sigma**2 / 2) * (T - t))) / (
        sigma * np.sqrt(T - t) + _MARGIN
    )
    p_survival = (ub - T) / (ub - t)
    delta_GMMB = p_survival * np.exp(-m * (T - t)) * (norm.cdf(d_1_m, 0, 1) - 1)

    # Delta from the rider charge (w.r.t Ft)
    integrand_rc: Callable[[np.ndarray], np.ndarray] = (
        lambda s: np.exp(m * s) * ((ub - s) / (ub - t)) * (s > t)
    )
    actuarial_value = integral_evaluator(integrand_rc, 0, T)
    delta_RC = me * np.exp(-r * (T - t) - m * t) * actuarial_value

    # compute the true Delta (w.r.t St)
    delta_true = (
        (delta_GMDB + delta_GMMB - delta_RC) * rho * np.exp(-m * t)
    )  # chain rule
    return delta_true


def delta_heston_cfm(
    Ft: float,
    t: float,
    G_m: float,
    G_d: float,
    mu: float,
    m: float,
    me: float,
    T: float,
    r: float,
    sigma: float,
    rho: float,
    kappa: float,
    v_bar: float,
    eta: float,
    corr: float,
    F0: float,
) -> float:
    """
    function computes the Delta per contract, provided
    that this policyholder is alive at time t. The underlying market model is Heston + CFM
    Reference: Feng et al. (2020)
    :param Ft: the segregate account value at time t
    :param t: current time point
    :param G_m: guaranteed minimum maturity benefit
    :param G_d: guaranteed minimum death benefit
    :param mu: force of mortality
    :param m: account deduction fee
    :param me: rider charge fee
    :param T: maturity
    :param r: interest rate
    :param sigma: volatility
    :param rho: initial investment strategy
    :param kappa: mean-reverting rate of the volatility
    :param v_bar: long term average volatility
    :param eta: volatility of volatility
    :param corr: correlation between price and volatility
    :param F0: initial value of segregate account
    :return: Delta at time t

    Note: unlike the delta calculators above, here this function only returns a float instead of an array
    """
    # margin to prevent numerical error
    _MARGIN = 1e-15

    d: Callable[[float], float] = lambda x: np.sqrt(
        (kappa - complex(0, corr * eta * x)) ** 2
        + (eta**2) * (complex(0, x) + x**2)
    )
    q: Callable[[float], float] = lambda x: kappa - d(x) - complex(0, corr * eta * x)
    g: Callable[[float], float] = lambda x: q(x) / (q(x) + 2 * d(x))

    # Delta from GMDB rider
    char: Callable[[float, float], float] = lambda x, s: np.exp(
        complex(0, x * np.log(Ft / F0))
        + complex(0, x * (r - m) * (s - t))
        + kappa * v_bar * (s - t) * q(x) / (eta**2)
        + (2 * kappa * v_bar / (eta**2))
        * np.log((1 - g(x)) / (1 - g(x) * np.exp(-d(x) * (s - t))))
        + (sigma * q(x) / (eta**2))
        * (1 - np.exp(-d(x) * (s - t)))
        / (1 - g(x) * np.exp(-d(x) * (s - t)))
    )
    integrand_1_gmdb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_d / Ft))) * char(x, s)
    ).real
    integrand_2_gmdb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_d / Ft))) * char((x - complex(0, 1)), s)
    ).real
    integrand_pi_2_gmdb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_d / Ft)))
        * char((x - complex(0, 1)), s)
        / (complex(0, x * char(complex(0, -1), s)))
    ).real
    dpi_1_df: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_1_gmdb, 0, 100, args=(s,))[0]
    )
    dpi_2_df: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_2_gmdb, 0, 100, args=(s,))[0]
    )
    pi_2: Callable[[float], float] = lambda s: char(complex(0, -1), s) * (
        0.5 - (1 / np.pi) * quad(integrand_pi_2_gmdb, 0, 100, args=(s,))[0]
    )
    delta_integrand: Callable[[float], float] = (
        lambda s: (
            rho
            * np.exp((r - m) * t)
            * np.exp(-r * s)
            * (G_d * dpi_1_df(s) - pi_2(s) - Ft * dpi_2_df(s))
        ).real
        * mu
        * np.exp(-(s - t) * mu)
        * (s > t)
    )

    # numerical evaluation (which is faster than quad, depending on the level of precision)
    PRECISION = 5
    ds = 1 / PRECISION
    _grid_pts = np.linspace(t, T, PRECISION + 1)
    delta_GMDB = np.sum(np.array([delta_integrand(s) for s in _grid_pts]) * ds)

    # Delta from GMMB rider
    integrand_1_gmmb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_m / Ft))) * char(x, s)
    ).real
    integrand_2_gmmb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_m / Ft))) * char((x - complex(0, 1)), s)
    ).real
    integrand_pi_2_gmmb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_m / Ft)))
        * char((x - complex(0, 1)), s)
        / (complex(0, x * char(complex(0, -1), s)))
    ).real
    dpi_1_df_gmmb: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_1_gmmb, 0, 100, args=(s,))[0]
    )
    dpi_2_df_gmmb: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_2_gmmb, 0, 100, args=(s,))[0]
    )
    pi_2_gmmb: Callable[[float], float] = lambda s: char(complex(0, -1), s) * (
        0.5 - (1 / np.pi) * quad(integrand_pi_2_gmmb, 0, 100, args=(s,))[0]
    )
    p_survival = np.exp(-mu * (T - t))
    delta_temp = (
        rho
        * np.exp(-r * (T - t) - m * t)
        * (G_m * dpi_1_df_gmmb(t) - pi_2_gmmb(t) - Ft * dpi_2_df_gmmb(t))
    )
    delta_GMMB = p_survival * delta_temp

    # Delta from Rider Charge
    integrand_rc: Callable[[float], float] = (
        lambda s: np.exp(-r * (s - t))
        * np.exp(-mu * (s - t))
        * char(complex(0, -1), s).real
    )
    delta_RC = me * np.sum(np.array([integrand_rc(s) for s in _grid_pts]) * ds)

    # compute the true Delta (w.r.t St)
    delta_true = (
        (delta_GMDB + delta_GMMB - delta_RC) * rho * np.exp(-m * t)
    ).real  # chain rule
    return delta_true


def delta_heston_ifm(
    Ft: float,
    t: float,
    G_m: float,
    G_d: float,
    ub: float,
    m: float,
    me: float,
    T: float,
    r: float,
    sigma: float,
    rho: float,
    kappa: float,
    v_bar: float,
    eta: float,
    corr: float,
    F0: float,
) -> float:
    """
    function computes the Delta per contract, provided
    that this policyholder is alive at time t. The underlying market model is Heston + CFM
    Reference: Feng et al. (2020)
    :param Ft: the segregate account value at time t
    :param t: current time point
    :param G_m: guaranteed minimum maturity benefit
    :param G_d: guaranteed minimum death benefit
    :param ub: upper bound of the uniform lifetime distribution
    :param m: account deduction fee
    :param me: rider charge fee
    :param T: maturity
    :param r: interest rate
    :param sigma: volatility
    :param rho: initial investment strategy
    :param kappa: mean-reverting rate of the volatility
    :param v_bar: long term average volatility
    :param eta: volatility of volatility
    :param corr: correlation between price and volatility
    :param F0: initial value of segregate account
    :return: Delta at time t
    """
    # margin to prevent numerical error
    _MARGIN = 1e-15

    d: Callable[[float], float] = lambda x: np.sqrt(
        (kappa - complex(0, corr * eta * x)) ** 2
        + (eta**2) * (complex(0, x) + x**2)
    )
    q: Callable[[float], float] = lambda x: kappa - d(x) - complex(0, corr * eta * x)
    g: Callable[[float], float] = lambda x: q(x) / (q(x) + 2 * d(x))

    # Delta from GMDB rider
    char: Callable[[float, float], float] = lambda x, s: np.exp(
        complex(0, x * np.log(Ft / F0))
        + complex(0, x * (r - m) * (s - t))
        + kappa * v_bar * (s - t) * q(x) / (eta**2)
        + (2 * kappa * v_bar / (eta**2))
        * np.log((1 - g(x)) / (1 - g(x) * np.exp(-d(x) * (s - t))))
        + (sigma * q(x) / (eta**2))
        * (1 - np.exp(-d(x) * (s - t)))
        / (1 - g(x) * np.exp(-d(x) * (s - t)))
    )
    integrand_1_gmdb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_d / Ft))) * char(x, s)
    ).real
    integrand_2_gmdb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_d / Ft))) * char((x - complex(0, 1)), s)
    ).real
    integrand_pi_2_gmdb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_d / Ft)))
        * char((x - complex(0, 1)), s)
        / (complex(0, x * char(complex(0, -1), s)))
    ).real
    dpi_1_df: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_1_gmdb, 0, 100, args=(s,))[0]
    )
    dpi_2_df: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_2_gmdb, 0, 100, args=(s,))[0]
    )
    pi_2: Callable[[float], float] = lambda s: char(complex(0, -1), s) * (
        0.5 - (1 / np.pi) * quad(integrand_pi_2_gmdb, 0, 100, args=(s,))[0]
    )
    delta_integrand: Callable[[float], float] = (
        lambda s: (
            rho
            * np.exp((r - m) * t)
            * np.exp(-r * s)
            * (G_d * dpi_1_df(s) - pi_2(s) - Ft * dpi_2_df(s))
        ).real
        * (1 / (ub - t))
        * (s > t)
    )

    # numerical evaluation (which is faster than quad, depending on the level of precision)
    PRECISION = 5
    ds = 1 / PRECISION
    _grid_pts = np.linspace(t, T, PRECISION + 1)
    delta_GMDB = np.sum(np.array([delta_integrand(s) for s in _grid_pts]) * ds)

    # Delta from GMMB rider
    integrand_1_gmmb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_m / Ft))) * char(x, s)
    ).real
    integrand_2_gmmb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_m / Ft))) * char((x - complex(0, 1)), s)
    ).real
    integrand_pi_2_gmmb: Callable[[float, float], float] = lambda x, s: (
        np.exp(complex(0, -x * np.log(G_m / Ft)))
        * char((x - complex(0, 1)), s)
        / (complex(0, x * char(complex(0, -1), s)))
    ).real
    dpi_1_df_gmmb: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_1_gmmb, 0, 100, args=(s,))[0]
    )
    dpi_2_df_gmmb: Callable[[float], float] = (
        lambda s: (1 / (-np.pi * Ft)) * quad(integrand_2_gmmb, 0, 100, args=(s,))[0]
    )
    pi_2_gmmb: Callable[[float], float] = lambda s: char(complex(0, -1), s) * (
        0.5 - (1 / np.pi) * quad(integrand_pi_2_gmmb, 0, 100, args=(s,))[0]
    )
    p_survival = (ub - T) / (ub - t)
    delta_temp = (
        rho
        * np.exp(-r * (T - t) - m * t)
        * (G_m * dpi_1_df_gmmb(t) - pi_2_gmmb(t) - Ft * dpi_2_df_gmmb(t))
    )
    delta_GMMB = p_survival * delta_temp

    # Delta from Rider Charge
    integrand_rc: Callable[[float], float] = (
        lambda s: np.exp(-r * (s - t))
        * ((ub - s) / (ub - t))
        * (s > t)
        * char(complex(0, -1), s).real
    )
    delta_RC = me * np.sum(np.array([integrand_rc(s) for s in _grid_pts]) * ds)

    # compute the true Delta (w.r.t St)
    delta_true = (
        (delta_GMDB + delta_GMMB - delta_RC) * rho * np.exp(-m * t)
    ).real  # chain rule
    return delta_true
