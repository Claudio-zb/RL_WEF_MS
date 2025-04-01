# This file contains the definitions of the Energy water microgrid environment, as well  as the energy management system
# and the gymnasium extension of the environment

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from scipy.special import exp1
import torch
from environments.utils.funcionesEMS import *
from gymnasium import spaces
from typing import Tuple, List, SupportsFloat
from abc import ABC, abstractmethod
from stable_baselines3.common.base_class import BaseAlgorithm


class EnergyWaterMG:
    def __init__(self, n_crops: int = 1):

        # setting up the environment
        self.n_crops: int = n_crops
        self.v_tanks_min: List[float] = [Vt_min] * n_crops
        self.v_tanks_max: List[float] = [Vt_max] * n_crops

        # ss variables
        self.v_tanks: npt.NDArray[np.float32] = np.ones(n_crops) * (Vt_max + Vt_min) / 2
        self.v_irrs: npt.NDArray[np.float32] = np.zeros(n_crops)
        self.drawdowns: npt.NDArray[np.float32] = np.zeros(n_crops)
        self.soe: float = SoE_max

        # drawdown relevant variables
        self.prev_Qps: npt.NDArray[np.float32] = np.zeros(n_crops)
        self.dQs: List[np.ndarray] = [np.array([0])] * n_crops

        # daily time counter
        self.k: int = 0
        self.doy = 1

    def next_step(self, actions: Tuple[float, List[list]]) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], float, float]:
        """
        note: the pbat action is computed from an external policy
        :param actions: Tuple of actions (p_bat, [[q_p, q_irr]])
        :return: Tuple of (v_tanks, v_irrs, drawdowns, soe, k)
        """
        if (self.k % 144) == 0:  # The time at s is 00:00 i.e. a new day is starting
            for idx in range(self.n_crops):
                self.v_irrs[idx] = 0.0

        p_bat, pumps = actions
        assert len(pumps) == self.n_crops, "The number of actions should match the number of crops"

        # unpacking the actions
        q_ps = [pair[0] for pair in pumps]
        q_irrs = [pair[1] for pair in pumps]

        q_ps = np.clip(np.array(q_ps), 0, Q_p_max)
        q_irrs = np.clip(np.array(q_irrs), 0, I_max)

        # p_fv = disturbances[0]
        # p_load = disturbances[1]

        # loop over the crops
        for idx, v_tank in enumerate(self.v_tanks):

            if self.v_tanks[idx] <= Vt_min:  # If the tank is empty, there is no irrigation
                q_irrs[idx] = 0.0

            self.v_tanks[idx] = np.clip(v_tank + (q_ps[idx] - q_irrs[idx]) * 600 / 1000, self.v_tanks_min[idx],
                                        self.v_tanks_max[idx])
            
            volume_to_extract = q_irrs[idx]*600/1000
            volume_available = np.max([self.v_tanks[idx] - self.v_tanks_min[idx], 0])

            if volume_to_extract > volume_available:
                q_irrs[idx] = volume_available*1000/600
            
            self.v_irrs[idx] = np.clip(self.v_irrs[idx] + q_irrs[idx] * 600 / 1000, 0, np.inf)

            delta_SoE = np.max([p_bat, 0]) * n_c * (dt / 3600) + np.min([p_bat, 0]) / n_d * (dt / 3600)  # [kWh]
            self.soe = np.clip(self.soe + delta_SoE, SoE_min, SoE_max)

            self.drawdowns[idx] = drawdown(self.k + 1, self.dQs[idx] / 1e3)

            self.dQs[idx] = np.append(self.dQs[idx], q_ps[idx] - self.prev_Qps[idx])
            self.prev_Qps[idx] = q_ps[idx]

        self.k += 1

        return self.v_tanks, self.v_irrs, self.drawdowns, self.soe, self.k % 144

    def set_state(self, v_tanks: list, v_irrs: list, dqs: list, soe: float, k: int):
        """ Set the state of the environment """
        self.v_tanks = v_tanks
        self.v_irrs = v_irrs
        self.dQs = dqs
        self.soe = soe
        self.k = k

    def start(self, doy: int) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], float, float]:
        """ Start the model in a certain day of year
        Returns: Tuple of (v_tanks, v_irrs, drawdowns, soe, k)"""
        self.doy = doy
        self.k = 0
        self.v_irrs = [0.0]*self.n_crops
        self.prev_Qps = np.zeros(self.n_crops)
        self.dQs = [np.array([0])] * self.n_crops
        return self.get_state()

    def get_state(self) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], float, float]:
        """ Get the state of the environment 
        Returns: Tuple of (v_tanks, v_irrs, drawdowns, soe, k)"""

        return self.v_tanks, self.v_irrs, self.drawdowns, self.soe, self.k


class AbstractEMS(ABC):
    def __init__(self, n_crops: int):
        self.n_crops = n_crops

    @abstractmethod
    def get_action(self,
                   state: Tuple[List[float], List[float], List[float], float, int],
                   v_reqs: Union[List[float], np.ndarray],
                   disturbances) -> Tuple[float, List[list]]:
        pass


class RuleBasedEMS(AbstractEMS):
    """Class for the Q_p, Q_irr action space agent"""
    def __init__(self, n_crops: int, rl_policy: BaseAlgorithm, isNormalized: bool = False):
        super().__init__(n_crops)
        self.policy = rl_policy
        self.residual: float = 0.0
        if isNormalized:
            self.transform = generate_t_matrix(n_crops)
        else:
            self.transform = np.eye(4 * n_crops + 5)

    def get_action(self,
                   state: Tuple[List[float], List[float], List[float], float, int],
                   v_reqs: Union[List[float], np.ndarray],
                   disturbances) -> Tuple[float, List[list]]:
        """Computes the action since the current observation
        disturbances: Tuple of (p_fv, p_load)
        return pbat, [[Q_p, Qirr]]"""
        p_fv = disturbances[0]
        p_load = disturbances[1]
        n_crops = self.n_crops
        flattened_state = np.concatenate((v_reqs, state[0], state[1], state[2],
                                          np.array([p_fv, p_load, state[3], self.residual, state[4]])))
        transformed_state = np.matmul(self.transform, flattened_state)
        actions = self.policy.predict(transformed_state, deterministic=True)[0]
        Q_p = actions[0:n_crops]
        P_q_ps = get_p_q_p(Q_p, h_p_const)
        Q_irr = actions[n_crops:]
        pumps = [[Q_p[i], Q_irr[i]] for i in range(n_crops)]

        SoE = state[3]
        Pbat, _, self.residual = manage_batteries(SoE, p_fv, p_load, P_q_ps)
        return Pbat, pumps


def obs_to_array(obs: Tuple[List[float], List[float], List[float], float, int]) -> Tuple[np.ndarray, int]:
    v_tanks, v_irrs, drawdowns, soe, k = obs
    n = len(v_tanks)  # number of crops
    return np.array(v_tanks + v_irrs + drawdowns + [soe, k]), n

def mg_tuple2array(mg_tuple):
    v_tanks, v_irrs, drawdowns, soe, k = mg_tuple
    mg_array = np.concatenate((v_tanks, v_irrs, drawdowns, [soe, k]))
    return mg_array


def map_action(policy_output: Union[torch.Tensor, np.ndarray]) -> npt.NDArray[np.float32]:
    if torch.is_tensor(policy_output):
        return policy_output.detach().cpu().numpy()
    else:
        return policy_output


class MicrogridEnv(gym.Env):
    """
    Gymnasium environment for the Energy Water Microgrid
    """

    def __init__(self, n_crops: int = 1, render: bool = True):
        """
        Initialize the environment
        :param render:
        """
        self.render = render
        self.days_per_episode = 3

        self.micro_grid = EnergyWaterMG(n_crops)
        self.n_crops = n_crops
        # Hyper params
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

        # Data variables 

        self.p_fv: float = None
        self.temperatura: npt.NDArray[np.float32] = None
        self.p_load:float = None
        self.radiation: npt.NDArray[np.float32] = None
        self.V_refs: npt.NDArray[np.float32] = get_ref()  # V_refs

        # State variables en inputs

        self.Q_irr: List[float] = [0.0] * n_crops
        self.Q_p: List[float] = [0.0] * n_crops
        self.V_ref: List[float] = [0.0] * n_crops
        self.Vt: List[float] = [0.0] * n_crops

        self.Pbat: float = 0.0
        self.res_energy: float = 0.
        self.day_picked: int = 0

        self.reward_fun = lambda s, a, s_next: default_rwd_fun(s, a, s_next, n_crops=n_crops)

        # Bounds for observations
        obs_low = np.array(n_crops * [0.0] + n_crops * [Vt_min] + 2 * n_crops * [0.0] + [0.0, 0.0, SoE_min, -10., 0],
                           dtype=np.float32)

        obs_high = np.array(
            n_crops * [4.0] + n_crops * [Vt_max] + 2 * n_crops * [0.0] + [max_power_sun, max_power_d, SoE_max, 50.,
                                                                          143],
            dtype=np.float32)

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0] * n_crops,
                                   dtype=np.float32)

        self.action_high = np.array([Q_p_max, I_max] * n_crops,
                                    dtype=np.float32)

        self.observation_space: spaces.Box = spaces.Box(low=obs_low,
                                                        high=obs_high,
                                                        shape=(4 * n_crops + 5,),
                                                        dtype=np.float32)

        self.action_space: spaces.Box = spaces.Box(low=self.action_low,
                                                   high=self.action_high,
                                                   shape=(2 * n_crops,),
                                                   dtype=np.float32)
        
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

        V_refs = [5.0 * np.random.rand() + 1.0 for _ in range(self.micro_grid.n_crops)]

        SoE = (SoE_max - SoE_min) * np.random.random_sample() + SoE_min

        self.micro_grid.set_state(v_tanks=V_tank,
                                  v_irrs=[0.0 for _ in range(self.micro_grid.n_crops)],
                                  dqs=[np.array([0]) for _ in range(self.micro_grid.n_crops)],
                                  soe=SoE,
                                  k=0)

        InitialObservation = self.set_initial_conditions(V_refs, 0)
        info = {}

        return InitialObservation, info

    def step(self, action: np.ndarray, mode: str = "train") -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :param mode: mode of the environment. Can be "train" or "eval"
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Store the previous values of the variables to compute the reward
        obs = self._get_obs()

        action = map_action(action)
        action = action.flatten()
        Q_pumps = action[0:self.n_crops]
        Q_irrs = action[self.n_crops:]
        P_pumps = get_p_q_p(Q_pumps, h_p_const)
        p_fv, p_load = self._get_disturbances() # update the disturbances
        self.Pbat, _, self.res_energy = manage_batteries(float(obs[-3]), float(p_fv), float(p_load),
                                                         P_pumps)
        pumps = [[Q_pumps[i], Q_irrs[i]] for i in range(self.n_crops)]
        _ = self.micro_grid.next_step((self.Pbat, pumps))

        truncated = False
        terminated = False
        Info = {}

        self.k = self.k + 1

        # self.drawdown = drawdown(self.k + 144, self.dQ)

        if (self.k % 144) == 0:  # The time at s' is 00:00 i.e. the final day is over

            self.V_refs = [6.0 * np.random.rand() for _ in range(self.n_crops)]

            if mode == "eval":
                self.day_picked = (self.day_picked + 1) % 70
            else:
                self.day_picked = np.random.randint(0, 70)

        if self.k % (self.day_steps*self.days_per_episode) == 0:
            terminated = True

        observation_next = self._get_obs()

        reward = self.reward_fun(obs, action, observation_next)

        return observation_next, reward, terminated, truncated, Info

    

    def _pick_meteorological_data(self, day_picked: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns p_fv and p_demanded for a given day"""

        n_steps = self.day_steps  # self.k + self.day_steps
        start_index = day_picked * self.day_steps  # self.max_steps
        radiation = self.radiation_data[start_index:start_index + n_steps + 1] + 1e-4 * np.random.randn(n_steps + 1)
        temperatura = self.temperature_data[start_index:start_index + n_steps + 1] + 1e-2 * np.random.randn(n_steps + 1)
        p_fv = solar_power(radiation, temperatura) + 1e-4 * np.random.randn(n_steps + 1)
        p_demanded = self.demand_data[start_index:start_index + n_steps + 1]

        return p_fv, p_demanded

    def _get_disturbances(self) -> Tuple[float, float]:
        """Returns the disturbances for the current time step
        :return: Tuple of (p_fv, p_demanded)"""
        n_steps = self.day_steps  # (144) # self.k + self.day_steps
        idx = self.day_picked * n_steps + self.k  # self.max_steps
        radiation = self.radiation_data[idx] + 1e-4 * np.random.randn()
        temperatura = self.temperature_data[idx] + 1e-2 * np.random.randn()
        p_fv = solar_power(radiation, temperatura) + 1e-4 * np.random.randn()
        p_demanded = self.demand_data[idx]

        return p_fv, p_demanded


    def set_initial_conditions(self, v_refs: npt.NDArray[np.float32], instant_k: int):
        """
        Set the initial conditions of the environment
        :param instant_k:
        :param day_picked:
        :param v_refs:
        :return: Initial observation
        """
        self.k = instant_k
        self.V_refs = v_refs
        self.p_fv, self.p_load = self._get_disturbances()

        v_tanks, v_irrs, drawdowns, soe, k = self.micro_grid.get_state()

        disturbances = np.array([self.p_fv, self.p_load])

        init_obs = self._get_obs()
        return init_obs

    def _create_dQ(self, V_req) -> Tuple[list, SupportsFloat]:
        """Returns the dQ sequence from a previous day and the last value for the pump action Q_p"""
        L = self.day_steps*(self.days_per_episode + 1)
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

    def _get_obs(self) -> npt.NDArray[np.float32]:
        """Return the observation of the environment"""
        v_tanks, v_irrs, drawdowns, soe, k = self.micro_grid.get_state()
        p_pv, p_load = self._get_disturbances()

        observation = np.concatenate((self.V_refs, v_tanks, v_irrs, drawdowns, [p_pv, p_load, soe, self.res_energy, self.k % 144]))
        return observation


class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env: MicrogridEnv):
        super().__init__(env)
        self.env = env
        self.transform = generate_t_matrix(env.n_crops)
        self.prev_action: np.ndarray = np.zeros(self.env.action_space.shape[-1])
        low = np.matmul(self.transform, env.observation_space.low)
        high = np.matmul(self.transform, env.observation_space.high)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        #self.action_space = spaces.Box(low=-self.action_high,
        #                               high=self.action_high,
        #                               shape=(2,),
        #                               dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.prev_action = np.zeros_like(self.env.action_space.shape[-1])
        state, info = self.env.reset()
        t_state = np.matmul(self.transform, state)
        return t_state, {"state": state}

    def step(self, action):
        action_ = action  # self.prev_action + action
        action_ = np.clip(action_, self.env.action_low, self.env.action_high)
        state, reward, terminated, truncated, _ = self.env.step(action_)
        t_state = np.matmul(self.transform, state)
        info = {"state": state}
        return t_state, reward, terminated, truncated, info


def default_rwd_fun(s, a, s_next, n_crops=1):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state
    :param n_crops: number of crops
    :return: reward"""
    reward = 0.0
    for i in range(n_crops):
        norm_next_error = (s[i] - s_next[i + 2 * n_crops]) / s[i]  # Normalize the error
        reward = np.clip(1 - abs(norm_next_error), -1.0, 1.0)

        reward += -4 * a[i + n_crops] if s[i + n_crops] <= Vt_min and a[
            i + n_crops] > 0 else 0.0  # penalize unfeasible action (irrigation is on and tank is empty)

        reward += -4 * a[i] if s[i + n_crops] >= Vt_max and a[
            i] > 0 else 0.0  # penalize unfeasible action (pump is on and tank is full)

        reward += -5 * a[i] if s[i + n_crops] > 1 else 0.0  # penalize drawdown

    e_balance = s_next[-2]

    reward += e_balance if e_balance < 0 else 0

    return reward



def ems_ode(x, d, u) -> Tuple[np.ndarray, float, float]:
    """The ODE of the microgrid system
    :param x: state vector V_irr, V_tank, SoE
    :param d: disturbance vector V_ref, P_sun, P_demand
    :param u: control vector Q_pump, Irrigation, P_bat
    :return: next state, battery power, energy residual"""

    V_irr = x[0]  # Irrigated volume [m3]
    V_tank = x[1]  # Tank volume [m3]
    SoE = x[2]  # State of Energy [kWh]
    # s = x[3] # Well drawdown [m]

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

    amount_to_pump = dt * (Q_pump * 1e-3)  # Volume [m3]

    V_tank_next = np.clip(V_tank + amount_to_pump - amount_to_irrigate, Vt_min, Vt_max)
    V_Irr_next = V_irr + amount_to_irrigate

    P_Q_p_ = get_p_q_p(Q_pump, h_p_const)  # water pump power [kW]

    P_bat, SoE_next, E_residual = manage_batteries(SoE,
                                                   P_sun,
                                                   P_demand,
                                                   P_Q_p_)
    x_next = np.array([V_Irr_next,
                       V_tank_next,
                       SoE_next])

    return x_next, P_bat, E_residual


def drawdown(k: int, dQ: Union[np.ndarray, List]):  # drawdown of the well
    """
    Computes the drawdown of the well according the theis equation
    :param k: time instant
    :param dQ: delta flow rate [m3/s]
    :return: drawdown
    """
    assert k == len(dQ), "The length of dQ should match the number of temporal k points"
    if type(dQ) == list:
        dQ = np.array(dQ)
    l = np.arange(1, k + 1)
    arg = (r_wells ** 2 * S) / (4 * T * (k - l + 1) * 600)
    sum_ = np.dot(dQ, exp1(arg))
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val


def get_p_q_p(q_p: Union[float, np.ndarray], h_p: float) -> np.ndarray:
    """Water pump power [kW]
    :param q_p: flow rate [l/s]
    :param h_p: height [m]
    :return: power [kW]"""
    if type(q_p) == float:
        q_p = np.array([q_p])
    P_Q_p_ = B_p * (q_p * 1e-3) * h_p / 1e3
    return P_Q_p_

class EnergyWaterMG2:
    """Wrapper for the Energy Water Microgrid model but incorporating the 
    betteries policy within the model"""
    def __init__(self, n_crops: int = 1):
        self.energy_water_mg = EnergyWaterMG(n_crops)
    def next_step(self, actions: List[List[float]]) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], float, float]:
        SoE = self.energy_water_mg.soe
        p_fv = self.energy_water_mg.p_fv
        p_load = self.energy_water_mg.p_demanded
        P_q_ps = get_p_q_p(actions, h_p_const)
        manage_batteries(SoE, p_fv, p_load, P_q_ps)
        self.energy_water_mg.next_step(actions)
