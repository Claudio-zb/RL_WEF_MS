# This file contains the definitions of the Energy water microgrid environment, as well  as the energy management system
# and the gymnasium extension of the environment

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np

from scipy.special import exp1
from environments.utils.funcionesEMS import *
from gymnasium import spaces
from typing import Tuple, List
from environments.EnergyWaterMG import EnergyWaterMG

class MicrogridEnv(gym.Env):
    """
    Gymnasium environment for the Energy Water Microgrid
    """

    def __init__(self, n_crops: int = 1, 
                 weights:np.ndarray[float, int] = np.array([1.0, 4.0, 1.0]),
                 days_per_episode:int = 3,
                 mode:str = "train"):
        """
        Initialize the environment
        :param render:
        """
        self.mode = mode
        self.days_per_episode = days_per_episode
        self.spec:EnvSpec = EnvSpec(id="MGEnv", entry_point="MGEnv")

        self.micro_grid = EnergyWaterMG(n_crops)
        self.n_crops = n_crops
        
        # Hyper params
        self.day_steps: int = 144
        self.start_index: int = 0
        self.k: int = 0

        # Load photovoltaic power and residential consumption data
        self.p_res_data: np.ndarray[np.floating, int] = get_demand()
        self.p_pv_data: np.ndarray[np.floating, int] = solar_power(get_rad('ver'), get_temperatura('ver'))

        # daily disturbances profile
        self.p_pv_daily: np.ndarray[np.floating, int] = np.zeros(self.day_steps)
        self.p_res_daily: np.ndarray[np.floating, int] = np.zeros(self.day_steps)

        self.N_dias: int = 70

        # State variables en inputs

        self.Q_irr: list[float] = [0.0] * n_crops
        self.Q_p: list[float] = [0.0] * n_crops
        self.V_ref: list[float] = [0.0] * n_crops
        self.Vt: list[float] = [0.0] * n_crops

        self.Pbat: float = 0.0
        self.res_energy: float = 0.
        # Disturbances 

        self.p_pv: float = 0.0
        self.p_res:float = 0.0

        self.reward_fun = lambda s, a, s_next: default_rwd_fun(s, a, s_next, n_crops=n_crops, weights=weights)

        # Bounds for observations
        obs_low = np.array(n_crops * [0.0] + n_crops * [Vt_min] + 3 * n_crops * [0.0] + [SoE_min, -10., 0, 0., 0.],
                           dtype=np.float32)

        obs_high = np.array(
            n_crops * [20.0] + n_crops * [Vt_max] + 3 * n_crops * [0.0] + [SoE_max, 50., 143, max_power_sun, max_power_d,],
            dtype=np.float32)

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0] * n_crops,
                                   dtype=np.float32)

        self.action_high = np.array([Q_p_max, I_max] * n_crops,
                                    dtype=np.float32)

        self.observation_space: spaces.Box = spaces.Box(low=obs_low,
                                                        high=obs_high,
                                                        shape=(5 * n_crops + 5,),
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

        V_tank = [np.minimum((Vt_max - Vt_min) * np.random.random_sample() + Vt_min,
                             (Vt_max - Vt_min) * np.random.random_sample() + Vt_min) for _ in
                  range(self.micro_grid.n_crops)]
        
        v_irrs = [0.0 for _ in range(self.micro_grid.n_crops)]
        drawdowns = [0.0 for _ in range(self.micro_grid.n_crops)]

        pbats = [0.0 for _ in range(self.micro_grid.n_crops)]
        dqs=[np.array([0]) for _ in range(self.micro_grid.n_crops)]
        e_residual = 0.0

        self.V_refs = [20.0 * np.random.rand() for _ in range(self.micro_grid.n_crops)]

        SoE = (SoE_max - SoE_min) * np.random.random_sample() + SoE_min
        observation = np.array((V_tank + v_irrs + drawdowns + pbats + [SoE, e_residual, 0]))
        self.micro_grid.set_state(observation, dqs)

        info = {}

        self.k = 0
        
        self._update_daily_profile(self.mode)
        self.p_fv, self.p_load = self._get_disturbances()

        init_obs = self._get_obs()
        return init_obs, info

    def step(self, action: np.ndarray, mode: str = "train") -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :param mode: mode of the environment. Can be "train" or "eval"
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Store the previous values of the variables to compute the reward
        obs = self._get_obs()
        self.p_fv, self.p_d = self._get_disturbances()
        _ = self.micro_grid.next_step(actions=action, disturbances=[self.p_fv, self.p_load])

        truncated = False
        terminated = False
        Info = {}

        self.k = self.k + 1

        if (self.k % 144) == 0:  # The time at s' is 00:00 i.e. the final day is over

            self.V_refs = [10.0 * np.random.rand() for _ in range(self.n_crops)]

        if self.k % (self.day_steps*self.days_per_episode) == 0:
            terminated = True

        observation_next = self._get_obs()

        reward = self.reward_fun(obs, action, observation_next)

        return observation_next, reward, terminated, truncated, Info


    
    def _update_daily_profile(self, mode: str = "train"):

        if mode == "train":
            day_picked_1 = np.random.randint(0, self.N_dias)
            day_picked_2 = np.random.randint(0, self.N_dias)
            solar_power_1 = self.p_pv_data[day_picked_1 * self.day_steps:(day_picked_1 + 1) * self.day_steps]
            solar_power_2 = self.p_pv_data[day_picked_2 * self.day_steps:(day_picked_2 + 1) * self.day_steps]

            p_res_1 = self.p_res_data[day_picked_1 * self.day_steps:(day_picked_1 + 1) * self.day_steps]
            p_res_2 = self.p_res_data[day_picked_2 * self.day_steps:(day_picked_2 + 1) * self.day_steps]

            alpha = np.random.rand()*0.2

            self.p_pv_daily = alpha * solar_power_1 + (1 - alpha) * solar_power_2
            self.p_res_daily = alpha * p_res_1 + (1 - alpha) * p_res_2
        
        else:
            day_picked_1 = np.random.randint(self.N_dias, self.N_dias+10)
            self.p_pv_daily = self.p_pv_data[day_picked_1 * self.day_steps:(day_picked_1 + 1) * self.day_steps]
            self.p_res_daily = self.p_res_data[day_picked_1 * self.day_steps:(day_picked_1 + 1) * self.day_steps]
        
        return


    def _create_dQ(self, V_req) -> Tuple[list, float]:
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

    def _get_obs(self) -> np.ndarray[np.floating, int]:
        """Return the observation of the environment"""
        mg_obs = self.micro_grid.get_observation()
        p_pv, p_load = self._get_disturbances()

        observation = np.concatenate((self.V_refs, mg_obs, [p_pv, p_load]))
        return observation
    
    def _get_disturbances(self) -> tuple[float, float]:
        """Get the disturbances for the environment"""

        p_pv = self.p_pv_daily[self.k % self.day_steps]
        p_load = self.p_res_daily[self.k % self.day_steps]

        return p_pv, p_load



class NormalisedMG(gym.Wrapper):
    def __init__(self, env: MicrogridEnv):
        super().__init__(env)
        self.env = env
        self.transform = generate_t_matrix(env.n_crops)
        self.prev_action: np.ndarray = np.zeros(self.env.action_space.shape[-1])
        low = np.matmul(self.transform, env.observation_space.low, dtype=np.float32)
        high = np.matmul(self.transform, env.observation_space.high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.zeros(2*self.env.n_crops, dtype=np.float32),
                                       high=np.ones(2*self.env.n_crops, dtype=np.float32),
                                       shape=(2*self.env.n_crops,),
                                       dtype=np.float32)

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
    

class EvalMG(gym.Wrapper):
    def __init__(self, env:NormalisedMG):
        env.env.mode = "eval"
        assert isinstance(env, NormalisedMG), "only NormalisedMG can be used with EvalMG"
        super().__init__(env)
        self.env:NormalisedMG = env

    def reset(self, seed: int = None, options:dict = None):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    

def default_rwd_fun(s, a, s_next, n_crops=1, weights:np.ndarray[float, int] = np.array([1.0, 4.0, 1.0])):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state
    :param n_crops: number of crops
    :return: reward"""
    reward = 0.0
    for i in range(n_crops):
        norm_next_error = (s[i] - s_next[i + 2 * n_crops]) / (s[i]+0.05)  # Normalize the error
        reward = np.clip(1 - abs(norm_next_error), -1.0, 1.0)
        if abs(norm_next_error) < 0.05: # bonus for being close to the reference
            reward += 1.0

        # penalisations    
        reward += -weights[1] * a[i + 1] if s_next[i + 2 * n_crops] > s[i] else 0.0  # penalize exceeding the irrigation requirement

        reward += -weights[1] * a[i + n_crops] if s[i + n_crops] <= Vt_min and a[
            i + n_crops] > 0 else 0.0  # penalize unfeasible action (irrigation is on and tank is empty)

        reward += -weights[1] * a[i] if s[i + n_crops] >= Vt_max and a[
            i] > 0 else 0.0  # penalize unfeasible action (pump is on and tank is full)

        reward += -weights[1] * a[i] if np.abs(s[i + 3*n_crops]) > 1 else 0.0  # penalize drawdown

    e_balance = weights[2]*s_next[7]

    reward += e_balance if e_balance < 0 else 0

    return reward
