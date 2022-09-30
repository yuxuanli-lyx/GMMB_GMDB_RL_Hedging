import numpy as np
from gym import spaces, Env
from typing import Callable, Dict, Tuple
from utils.formulas import fair_rate_bs_cfm
from utils.formulas import net_liability_bs_cfm
from utils.processes import geometric_brownian_motion

# const
PRECISION = np.float64


class TradingEnvUnderBSCFM(Env):
    """
    Build a trading environment to hedge contracts with GMMB + GMDB riders.
    Construct a replicating portfolio P_t to mimic the value dynamic of
    the net liability of contracts.

    This replicating portfolio consists of a stock (risky asset)
    and a bond (risk-free asset). The market under consideration is BS + CFM.
    """

    def __init__(
        self,
        financial_market_params: Dict,
        actuarial_market_params: Dict,
        fair_rate_calculator: Callable = fair_rate_bs_cfm,
        liability_calculator: Callable = net_liability_bs_cfm,
        market_simulator: Callable = geometric_brownian_motion,
        reward_type: "str" = "terminal",
    ) -> None:
        """
        :param financial_market_params: dictionary of financial market parameters
        :param actuarial_market_params: dictionary of actuarial market parameters
        :param fair_rate_calculator: fair rate calculator
        :param liability_calculator: net liability calculator
        :param market_simulator: stock price simulator
        :param reward_type: reward function
        """
        super(TradingEnvUnderBSCFM, self).__init__()

        # define action space: position to hold in the stock (per survival policyholder)
        self.action_space = spaces.Box(low=-5, high=5, shape=(1,), dtype=PRECISION)

        # define observation space: (current version)
        # (
        #  ln(Ft), 'the ln of segregate account value'
        #  Pt / total_pc, 'the hedging portfolio value per pc'
        #  num_survival_pc / total_pc, 'the survival rate'
        #  tau, 'term to maturity'
        # )
        try:
            _tau = actuarial_market_params["T"]
        except KeyError:
            raise KeyError("T is not included " "in the actuarial market parameters")
        else:
            self.observation_space = spaces.Box(
                low=np.array([float("-inf"), float("-inf"), 0, 0]),
                high=np.array([float("inf"), float("inf"), 1, _tau]),
                dtype=PRECISION,
            )

        self._financial_market_params = financial_market_params
        self._actuarial_market_params = actuarial_market_params
        self._fair_rate_calculator = fair_rate_calculator
        self._liability_calculator = liability_calculator
        self._market_simulator = market_simulator
        self._reward_type = reward_type
        self._online_mode = False
        self._init_Pt = False
        self._online_S0 = self._financial_market_params["S0"]
        self._init_Pt_val = 0

    def _initialize(self) -> None:
        """
        Initialize the trading environment
        """
        # extract the static market parameters
        try:
            self._r = self._financial_market_params["r"]
            self._S0 = self._financial_market_params["S0"]
            self._mu = self._financial_market_params["mu"]
            self._sigma = self._financial_market_params["sigma"]
            self._Gm = self._actuarial_market_params["Gm"]
            self._Gd = self._actuarial_market_params["Gd"]
            self._rho = self._actuarial_market_params["rho"]
            self._T = self._actuarial_market_params["T"]
            self._N0 = self._actuarial_market_params["N0"]
            self._fom = self._actuarial_market_params["fom"]
            self._m = self._fair_rate_calculator(
                self._S0 * self._rho,
                self._Gm,
                self._Gd,
                self._fom,
                self._T,
                self._r,
                self._sigma,
            )
            self._me = self._m * 0.95  # by default, m_e/m = 0.95
            self._total_steps = int(
                self._T * self._financial_market_params["num_steps"]
            )
            self._dt = 1 / self._financial_market_params["num_steps"]
        except KeyError:
            raise KeyError("financial/actuarial market parameters are not complete")

        # generate the stock price/segregate account value trajectory, policyholders' lifetime,
        # number of survival policyholders trajectory, and net liability trajectory
        if self._online_mode:
            self._S0 = self._online_S0
        _timeline = np.arange(self._total_steps + 1) * self._dt
        _lifetime = np.random.exponential(1 / self._fom, self._N0)
        self._St_traj = self._market_simulator(
            self._S0, self._mu, self._sigma, self._T, self._total_steps
        )
        self._Ft_traj = self._rho * self._St_traj * np.exp(-self._m * _timeline)
        self._Nt_traj = np.array([np.sum(_lifetime >= _pts) for _pts in _timeline])
        self._Lt_traj = self._liability_calculator(
            self._Ft_traj,
            _timeline,
            self._Gm,
            self._Gd,
            self._fom,
            self._m,
            self._me,
            self._T,
            self._r,
            self._sigma,
        )

        # initialize the temporal info
        self._step = 0
        self._Lt = self._Lt_traj[self._step] * self._Nt_traj[self._step]
        self._Pt = 0  # initial hedging portfolio value (can be set as self._Lt since self._Lt = 0 at time 0)
        if self._init_Pt:
            self._Pt = self._init_Pt_val
        self.state = np.array(
            [
                np.log(self._Ft_traj[self._step]),
                self._Pt / self._N0,
                self._Nt_traj[self._step] / self._N0,
                self._T,
            ]
        )  # initial observations

    def _transition(self, action: float) -> None:
        """
        The transition of the trading environment as well as the
        observed states by the RL agent

        :param action: the action taken by the RL agent,
        which represents the number of shares holding per survival policyholder
        """
        if self._step >= self._total_steps or self._Nt_traj[self._step] == 0:
            raise ValueError(
                "End of sample path or no policyholder alive. "
                "No further transitions allowed."
            )
        else:
            _next_step = self._step + 1
            _St, _St_next = self._St_traj[self._step], self._St_traj[_next_step]
            _Nt, _Nt_next = self._Nt_traj[self._step], self._Nt_traj[_next_step]
            _Ft, _Ft_next = self._Ft_traj[self._step], self._Ft_traj[_next_step]
            self._Lt = (
                self._Lt_traj[_next_step] * _Nt_next
            )  # the total net liability at time (t + 1)

            # update the hedging portfolio value at time (t + 1)
            self._Pt = (
                (self._Pt - action * _Nt * _St + self._me * _Ft * self._dt * _Nt)
                * np.exp(self._r * self._dt)
                + action * _Nt * _St_next
                - (_Nt - _Nt_next) * np.maximum(self._Gd - _Ft_next, 0)
            )

            # update the state vector at next time step
            _tau_next = self._T - _next_step * self._dt
            _next_state = np.array(
                [np.log(_Ft_next), self._Pt / self._N0, _Nt_next / self._N0, _tau_next]
            )
            self.state = _next_state
            self._step = _next_step

    def _reward(
        self,
        Pt: float,
        Pt_next: float,
        Lt: float,
        Lt_next: float,
        step_next: int,
        reward_type: str,
    ) -> float:
        """
        Reward function of the environment

        :param Pt: the current step hedging portfolio value
        :param Pt_next: the next step hedging portfolio value
        :param Lt: the current step total net liability
        :param Lt_next: the next step total net liability
        :param step_next: the next step index
        :param reward_type: reward type
        :return: numerical value of the reward
        """
        _supported_reward_type = [
            "terminal",
            "anchor-hedging",
            "stepwise",
            "evaluation",
        ]
        if reward_type not in _supported_reward_type:
            raise ValueError("Reward type: {} is not supported".format(reward_type))
        elif reward_type == "terminal":  # here we only collect the terminal reward
            if step_next != self._total_steps:
                reward = 0
            else:
                reward = -1 * ((Pt_next - Lt_next) / self._N0) ** 2
        elif reward_type == "anchor-hedging":
            reward = ((Pt - Lt) / self._N0) ** 2 - ((Pt_next - Lt_next) / self._N0) ** 2
        elif reward_type == "evaluation":  # evaluation mode
            reward = (Pt_next - Lt_next) / self._N0
        else:
            reward = -1 * ((Pt_next - Lt_next) / self._N0) ** 2

        return reward

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        interaction between the env and the RL agent, which
        1. moves the env forward
        2. feedbacks RL agent with new state vector and reward

        :param action: the action taken by the RL agent,
        which represents the number of shares holding per survival policyholder
        :return: the next state vector, the reward, done, info for useful additional information
        """
        action = float(action)

        # store the current step hedging portfolio value and net liability
        Pt = self._Pt
        Lt = self._Lt
        state = self.state

        # interact with env, update, and cal the reward
        self._transition(action)
        Pt_next = self._Pt
        Lt_next = self._Lt
        step_next = self._step
        Nt_next = self._Nt_traj[step_next]
        state_next = self.state
        reward = self._reward(Pt, Pt_next, Lt, Lt_next, step_next, self._reward_type)
        self._online_S0 = self._St_traj[step_next]
        done = step_next >= self._total_steps or Nt_next <= 0
        info = {"pre_state": state, "current_step": step_next}
        return state_next, reward, done, info

    def reset(self) -> np.array:
        """
        Reset the environment

        :return: initial state vector
        """
        self._initialize()
        _init_state = self.state
        return _init_state

    def online_mode(self) -> None:
        """
        Turn on the online mode
        """
        self._online_mode = True

    def init_pt(self, val: float) -> None:
        """
        Set the initial value of hedging portfolio (only useful for the online learning phase)
        :param val: the initial value of hedging portfolio
        """
        self._init_Pt = True
        self._init_Pt_val = val

    def render(self, mode="human"):
        pass

    def close(self):
        del self
