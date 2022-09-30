import numpy as np
from utils.formulas import (
    delta_bs_cfm,
    delta_bs_ifm,
    delta_heston_cfm,
    delta_heston_ifm,
)


class DeltaAgentBSCFM:
    """
    Object for Delta agent under BS financial + CFM actuarial markets
    """

    def __init__(
        self,
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
    ) -> None:
        """
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
        """
        super(DeltaAgentBSCFM, self).__init__()
        self.delta_traj = delta_bs_cfm(Ft, t, G_m, G_d, mu, m, me, T, r, sigma, rho)

    def predict(self, time_step: int) -> float:
        """
        Method returns the action taken by the Delta agent, given the current time point
        :param time_step: the current time step at which the action is taken
        :return: action taken at `time_step'
        """
        try:
            action = self.delta_traj[time_step]
        except IndexError:
            raise IndexError("time step is not valid")
        else:
            return action


class DeltaAgentBSIFM:
    """
    Object for Delta agent under BS financial + IFM actuarial markets
    """

    def __init__(
        self,
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
    ) -> None:
        """
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
        """
        super(DeltaAgentBSIFM, self).__init__()
        self.delta_traj = delta_bs_ifm(Ft, t, G_m, G_d, ub, m, me, T, r, sigma, rho)

    def predict(self, time_step: int) -> float:
        """
        Method returns the action taken by the Delta agent, given the current time point
        :param time_step: the current time step at which the action is taken
        :return: action taken at `time_step'
        """
        try:
            action = self.delta_traj[time_step]
        except IndexError:
            raise IndexError("time step is not valid")
        else:
            return action


class DeltaAgentHestonCFM:
    """
    Object for Delta agent under Heston financial + CFM actuarial markets
    """

    def __init__(
        self,
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
    ) -> None:
        """
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
        """
        # initialize with static parameters
        self.G_m = G_m
        self.G_d = G_d
        self.fom = mu
        self.m = m
        self.me = me
        self.T = T
        self.r = r
        self.sigma = sigma
        self.rho = rho
        self.kappa = kappa
        self.v_bar = v_bar
        self.eta = eta
        self.corr = corr
        self.F0 = F0

    def predict(self, Ft: float, t: float) -> float:
        """
        Method returns the action taken by the Delta agent, given the current observed Ft and time
        :param Ft: current segregate account value
        :param t: current time point
        :return: action taken by the Delta Agent
        """
        action = delta_heston_cfm(
            Ft,
            t,
            self.G_m,
            self.G_d,
            self.fom,
            self.m,
            self.me,
            self.T,
            self.r,
            self.sigma,
            self.rho,
            self.kappa,
            self.v_bar,
            self.eta,
            self.corr,
            self.F0,
        )
        return action


class DeltaAgentHestonIFM:
    """
    Object for Delta agent under Heston financial + IFM actuarial markets
    """

    def __init__(
        self,
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
    ) -> None:
        """
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
        """
        # initialize with static parameters
        self.G_m = G_m
        self.G_d = G_d
        self.ub = ub
        self.m = m
        self.me = me
        self.T = T
        self.r = r
        self.sigma = sigma
        self.rho = rho
        self.kappa = kappa
        self.v_bar = v_bar
        self.eta = eta
        self.corr = corr
        self.F0 = F0

    def predict(self, Ft: float, t: float) -> float:
        """
        Method returns the action taken by the Delta agent, given the current observed Ft and time
        :param Ft: current segregate account value
        :param t: current time point
        :return: action taken by the Delta Agent
        """
        action = delta_heston_ifm(
            Ft,
            t,
            self.G_m,
            self.G_d,
            self.ub,
            self.m,
            self.me,
            self.T,
            self.r,
            self.sigma,
            self.rho,
            self.kappa,
            self.v_bar,
            self.eta,
            self.corr,
            self.F0,
        )
        return action
