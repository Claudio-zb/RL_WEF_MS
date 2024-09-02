from typing import List
import gymnasium as gym

from scipy.special import exp1
import torch
from environments.utils.funcionesEMS import *
from gymnasium import spaces
from typing import Tuple


class MicroGridEnv:
    def __init__(self, n_crops: int = 1):

        # setting up the environment
        self.n_crops: int = n_crops
        self.v_tanks_min: List[float] = [Vt_min] * n_crops
        self.v_tanks_max: List[float] = [Vt_max] * n_crops

        # ss variables
        self.v_tanks: List[float] = [(Vt_max + Vt_min) / 2] * n_crops
        self.v_irrs: List[float] = [0] * n_crops
        self.drawdowns: List[float] = [0] * n_crops
        self.soe: float = SoE_max

        # drawdown relevant variables
        self.prev_Qps: List[float] = [0] * n_crops
        self.dQs: List[np.ndarray] = [np.array([0])] * n_crops

        # daily time counter
        self.k: int = 0
        # still don't know why to include it
        self.v_refs = [0.0 for _ in range(n_crops)]

    def next_step(self, actions: Tuple[float, list], disturbances) -> Tuple[list, list, list, float, float]:
        """The pbat battery is computed from an external policy"""

        assert len(actions[1]) == self.n_crops, "The number of actions should match the number of crops"

        # unpacking the actions
        p_bat = actions[0]
        q_ps = [a_pair[0] for a_pair in actions]
        q_irrs = [a_pair[1] for a_pair in actions]

        q_ps = np.clip(np.array(q_ps), 0, Q_p_max)
        q_irrs = np.clip(np.array(q_irrs), 0, I_max)

        p_fv = disturbances[0]
        p_load = disturbances[1]

        # loop over the crops
        for idx, vtank in self.v_tanks:
            self.v_tanks[idx] = np.clip(vtank + q_ps[idx], self.v_tanks_min[idx], self.v_tanks_max[idx])
            self.v_irrs[idx] = np.clip(self.v_irrs[idx] + q_irrs[idx], 0, np.inf)
            self.soe = np.clip(self.soe + p_bat * 600, SoE_min, SoE_max)

            self.drawdowns[idx] = drawdown(self.k, self.dQs[idx])

            self.dQs[idx].append(q_ps[idx] - self.prev_Qps[idx])
            self.prev_Qps[idx] = q_ps[idx]

        self.k += 1

        if (self.k % 144) == 0:  # The time at s' is 00:00 i.e. the final day is over|
            for idx in range(self.n_crops):
                self.v_irrs[idx] = 0.0

        return self.v_tanks, self.v_irrs, self.drawdowns, self.soe, self.k

    def set_state(self, vtanks: list, virrs: list, dqs: list, soe: float, k: int):
        """ Set the state of the environment """
        self.v_tanks = vtanks
        self.v_irrs = virrs
        self.dQs = dqs
        self.soe = soe
        self.k = 0

    def get_state(self) -> Tuple[list, list, list, float, float]:
        return self.v_tanks, self.v_irrs, self.drawdowns, self.soe, self.k


class ContinousEMSEnv(gym.Env):
    """
    Continous environment for the Energy Management System
    """

    def __init__(self, rwd_function=None, render: bool = True):
        """
        Initialize the environment
        :param render:
        """
        self.render = render

        self.micro_grid = MicroGridEnv(2)

        # Hyper params
        self.max_steps: int = 288
        self.day_steps: int = 144
        self.start_index: int = 0
        self.k: int = 0
        self.time: int = 0

        # Load meteorological data and demand

        self.radiation_data: np.ndarray = get_rad('ver')
        self.temperature_data: np.ndarray = get_temperatura('ver')
        self.demand_data: np.ndarray = get_demand()
        self.L = len(self.temperature_data)  # length(temperatura)
        self.N_dias: int = 70
        self.transform = T_matrix
        self.transform_dim = self.transform.shape[1]

        # Data variables 

        self.p_fv: np.ndarray[float] = np.array([0.0])
        self.temperatura: np.ndarray[float] = np.array([0.0])
        self.demanda: np.ndarray[float] = np.array([0.0])
        self.radiacion: np.ndarray[float] = np.array([0.0])
        self.V_refs = get_ref()  # V_refs 

        # State variables en inputs

        self.Irr = np.array([0.0])
        self.Q_p = np.array([0.0])
        self.Pbat = np.array([0.0])
        self.V_ref = np.array([0.0])
        self.Vt = np.array([0.0])
        self.day_picked = 0

        self.reward_fun = continous_rwd_fun

        # Bounds for observations
        obs_low = np.array([0.0,
                            Vt_min, I_min,
                            SoE_min,
                            0.0,
                            0.0, 0,
                            0.0, 0.0,
                            -1, -1], dtype=np.float32)

        obs_high = np.array([Vt_max,
                             Vt_max, I_max,
                             SoE_max,
                             Vt_max,
                             Q_p_max, 4,
                             1000, 1000,
                             1, 1], dtype=np.float32)

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0],
                                   dtype=np.float32)

        self.action_high = np.array([Q_p_max, I_max],
                                    dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(11,),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=self.action_low,
                                       high=self.action_high,
                                       shape=(2,),
                                       dtype=np.float32)

    def step(self, action: np.ndarray, mode: str = "train") -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :param mode: mode of the environment. Can be "train" or "eval"
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Store the previous values of the variables to compute the reward
        obs = self._get_obs()

        action = self.map_action(action)
        action = action.flatten()

        disturbances = [self.p_fv[self.k % 144], self.demanda[self.k % 144]]
        next_obs = self.micro_grid.next_step((self.Pbat, [self.Q_p, self.Irr]), disturbances)

        truncated = False
        terminated = False
        Info = {}

        self.k = self.k + 1

        # self.drawdown = drawdown(self.k + 144, self.dQ)

        if (self.k % 144) == 0:  # The time at s' is 00:00 i.e. the final day is over

            self.V_ref = 4.0 * np.random.rand()

            if mode == "eval":
                self.day_picked = (self.day_picked + 1) % 70
            else:
                self.day_picked = np.random.randint(0, 70)

            self.p_fv, self.demanda = self._pick_metereological_data(self.day_picked)

        if self.k % 288 == 0:
            self.V_ref = 4.0 * np.random.rand()
            terminated = True

        observation_next = self._get_obs()

        reward = self.reward_fun(obs, action, next_obs)

        return observation_next, reward, terminated, truncated, Info

    def map_action(self, policy_output: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if torch.is_tensor(policy_output):
            return policy_output.detach().cpu().numpy()
        else:
            return policy_output

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state
        :param seed: random seed
        :param options: options for the environment
        :return: tuple of (initial_observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        if options is not None:
            if options["mode"] == "eval":
                np.random.seed(0)
                self.day_picked = 0
        else:
            self.day_picked = np.random.randint(0, 70)

        V_tank = [np.minimum((Vt_max - Vt_min) * np.random.random_sample() + Vt_min,
                             (Vt_max - Vt_min) * np.random.random_sample() + Vt_min) for _ in
                  range(self.micro_grid.n_crops)]

        V_refs = [3.0 * np.random.rand() + 1.0 for _ in range(self.micro_grid.n_crops)]

        SoE = (SoE_max - SoE_min) * np.random.random_sample() + SoE_min

        self.micro_grid.set_state(V_tank, [0.0 for _ in range(self.micro_grid.n_crops)], SoE)

        InitialObservation = self.set_initial_conditions(self.day_picked, V_refs, 0)
        info = {}

        return InitialObservation, info

    def _pick_metereological_data(self, day_picked: int):
        """returns p_fv, demanda for a given day"""

        n_steps = self.day_steps  # self.k + self.day_steps
        start_index = day_picked * self.day_steps  # self.max_steps
        radiacion = self.radiation_data[start_index:start_index + n_steps + 1] + 1e-4 * np.random.randn(n_steps + 1)
        temperatura = self.temperature_data[start_index:start_index + n_steps + 1] + 1e-2 * np.random.randn(n_steps + 1)
        p_fv = solar_power(radiacion, temperatura) + 1e-4 * np.random.randn(n_steps + 1)
        demanda = self.demand_data[start_index:start_index + n_steps + 1]

        return p_fv, demanda

    def set_initial_conditions(self, day_picked: int, v_refs: List[float], instant_k: int):
        """
        Set the initial conditions of the environment
        :param instant_k:
        :param day_picked:
        :param v_refs:
        :return: Initial observation
        """
        self.k = instant_k
        self.v_refs = v_refs
        self.p_fv, self.demanda = self._pick_metereological_data(day_picked)

        v_tanks, v_irrs, drawdowns, soe = self.micro_grid.get_state()
        p_fv, demanda = self.p_fv[self.k], self.demanda[self.k]
        disturbances = [p_fv, demanda]
        _, _, e_residual = manage_batteries(soe, p_fv, demanda, 0)

        init_obs = np.array(v_refs + v_tanks + v_irrs + drawdowns + disturbances + [self.k % 144] + [e_residual])
        return init_obs

    def _create_dQ(self, V_req) -> Tuple[list, float]:
        """Returns the dQ sequence from a previous day and the last value for the pump action Q_p"""
        L = self.day_steps + self.max_steps
        prev_Q = np.zeros(self.day_steps)
        sum_ = 0
        K = Q_p_max * dt / 1000
        for i in range(L):
            if sum_ < V_req:
                x = np.random.rand() * K
                if sum_ + x < V_req:
                    prev_Q[i] = x
                    sum_ += x
                else:
                    prev_Q[i] = V_req - sum_
                    break
            else:
                break
        prev_Q = prev_Q / K
        np.random.shuffle(prev_Q)
        last_Q = prev_Q[-1]
        prev_d_Q = np.diff(prev_Q, prepend=0) / 1000
        # d_Q[0:self.day_steps] = prev_d_Q
        d_Q = [dq for dq in prev_d_Q]
        return d_Q, last_Q

    def _get_obs(self) -> np.ndarray:
        """Return the observation of the environment"""
        v_tanks, v_irrs, drawdowns, soe = self.micro_grid.get_state()
        disturbances = [self.p_fv[self.k % 144], self.demanda[self.k % 144]]
        observation = np.array([soe] + v_tanks + v_irrs + disturbances)

        observation = np.array([self.V_refs] + v_tanks + v_irrs + disturbances + [self.k % 144] + [soe])
        return observation


class normalizationWrapper(gym.Wrapper):
    def __init__(self, env: ContinousEMSEnv):
        super().__init__(env)
        self.env = env
        self.transform = T_matrix
        self.prev_action: np.ndarray = np.zeros(self.env.action_space.shape[-1])
        low = np.matmul(self.transform, env.observation_space.low)
        high = np.matmul(self.transform, env.observation_space.high)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_high,
                                       high=self.action_high,
                                       shape=(2,),
                                       dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.prev_action = np.zeros_like(self.env.action_space.shape[-1])
        state, info = self.env.reset()
        t_state = np.matmul(self.transform, state)
        info = state
        return t_state, state

    def step(self, action):
        action_ = self.prev_action + action
        action_ = np.clip(action_, self.env.action_low, self.env.action_high)
        state, reward, terminated, truncated, _ = self.env.step(action_)
        t_state = np.matmul(self.transform, state)
        info = state
        return t_state, reward, terminated, truncated, info


def default_rwd_fun(s, a, s_next):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state
    :param e_penal: penalty for energy deficit"""
    reward = 0.0
    norm_next_error = (s[0] - s_next[1]) / s[0]  # Normalize the error to be a fraction of the daily demand

    reward = - norm_next_error if norm_next_error > 0 else norm_next_error

    reward += -a[1] if s[3] <= Vt_min and a[
        1] > 0 else 0.0  #penalize unfeasible action (irrigation is on and tank is empty)

    reward += -a[0] if s[3] >= Vt_max and a[0] > 0 else 0.0  #penalize unfeasible action (pump is on and tank is full)

    e_balance = s_next[-1]

    reward += e_balance if e_balance < 0 else 0

    if a[0] < 0.0 or 1.0 < a[0]:
        reward -= abs(a[0]) * 2

    if a[1] < 0.0 or 1.0 < a[1]:
        reward -= abs(a[1]) * 2

    return np.array([reward], dtype=np.float32)


def continous_rwd_fun(s, a, s_next):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state"""
    reward = 0.0
    p_reward = 0.0
    K = s[-2] / 143
    if s_next[-2] != 0:
        norm_next_error = (s[0] - s_next[1]) / s[0]
        if s_next[0] <= s_next[1] and a[1] > 0:
            p_reward = -a[1]
    else:
        norm_next_error = (s[0] - s[1]) / s[0]

    reward = (1 - norm_next_error) ** 2 if norm_next_error > 0 else (1 + norm_next_error) ** 2  # ta gucci
    if norm_next_error < -1:
        reward = -1.0
    reward = K * np.maximum(reward, -1.0)  # ta gucci
    reward = reward  #+ p_reward

    reward += -10 * a[1] if s[3] <= Vt_min and a[
        1] > 0 else 0.0  # penalize unfeasible action (irrigation is on and tank is empty)

    reward += -10 * a[0] if s[3] >= Vt_max and a[
        0] > 0 else 0.0  # penalize unfeasible action (pump is on and tank is full)

    reward += -10 * a[0] if s[5] > 1 else 0.0  # penalize drawdown

    e_balance = s_next[-1]

    reward += e_balance if e_balance < 0 else 0.0

    if a[0] < 0.0 or 1.0 < a[0]:
        reward -= 10 * abs(a[0])

    if a[1] < 0.0 or 1.0 < a[1]:
        reward -= 10 * abs(a[1])

    return reward


def EMS_ode(x, d, u) -> Tuple[np.ndarray, float, float]:
    """The ODE of the microgrid system
    :param x: state vector V_irr, V_tank, SoE
    :param d: disturbance vector V_ref, P_sun, P_demand
    :param u: control vector Q_pump, Irrigation, P_bat
    :return: next state, battery power, energy residual"""

    V_irr = x[0]  # Irrigatated volume [m3]
    V_tank = x[1]  # Tank volume [m3]
    SoE = x[2]  # State of Energy [kWh]
    # s = x[3] # Descenso del pozo

    V_ref = d[0]  # Reference volume [m3]
    P_sun = d[1]  # Solar power [kW]
    P_demand = d[2]  # Demand power [kW]

    u = np.clip(u, [0.0, 0.0], [1.0, 1.0])  # Clip the action to the feasible range

    Q_pump = u[0]  # Pump flow rate [l/s]
    Irrigation = u[1]  # Irrigation flow rate [l/s]
    # P_bat = u[2] # Battery power [kW]

    if V_tank <= Vt_min:  # If the tank is empty, there is no irrigation
        if Irrigation > 0:
            Irrigation = 0.0

    amount_to_irrigate = dt * (Irrigation * 1e-3)

    # Vt_to_fill = Vt_max - V_tank - amount_to_irrigate  # Amount of water that can be filled

    # if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
    #     self.Q_p = 0

    amount_to_pump = dt * (Q_pump * 1e-3)  # Volume [m3]

    V_tank_next = np.clip(V_tank + amount_to_pump - amount_to_irrigate, Vt_min, Vt_max)
    V_Irr_next = V_irr + amount_to_irrigate

    P_Q_p_ = P_Q_p(Q_pump, h_p_const)  # water pump power [kW]

    P_bat, SoE_next, E_residual = manage_batteries(SoE,
                                                   P_sun,
                                                   P_demand,
                                                   P_Q_p_)
    x_next = np.array([V_Irr_next,
                       V_tank_next,
                       SoE_next])

    return x_next, P_bat, E_residual


def drawdown(k: int, dQ: Union[np.ndarray, List]):  # drawdown of the well
    assert k == len(dQ), "The length of dQ should match the number of temporal k points"
    if type(dQ) == list:
        dQ = np.array(dQ)
    l = np.arange(1, k + 1)
    arg = (r_pozos ** 2 * S) / (4 * T * (k - l + 1) * 600)
    sum_ = np.dot(dQ, exp1(arg))
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val


def P_Q_p(q_p: float, h_p: float):
    """Water pump power [kW]
    :param q_p: flow rate [l/s]
    :param h_p: height [m]
    :return: power [kW]"""
    P_Q_p_ = B_p * (q_p * 1e-3) * h_p / 1e3
    return P_Q_p_


class EMS_Wrapper(gym.Wrapper):
    """This class normalize the observations of the EMS environment"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.transform = T_matrix

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info
