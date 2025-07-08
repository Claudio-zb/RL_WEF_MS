# This file contains the definitions of the Energy water microgrid environment, as well  as the energy management system
# and the gymnasium extension of the environment

import gymnasium as gym
import numpy as np
import numpy.typing as npt

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
                 render: bool = True, 
                 weights:np.ndarray[float, int] = np.array([1.0, 4.0, 1.0])):
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
        self.pv_power_data: np.ndarray = np.zeros(0)
        self.res_power_data:np.ndarray = np.zeros(0)

        self.demand_data: np.ndarray = get_demand()

        self.N_dias: int = 90

        # Data variables 

        self.p_pv:float = None
        self.p_res:float = None

        self.V_refs: npt.NDArray[np.float32] = None  # V_refs

        # State variables en inputs

        self.Q_irr: List[float] = [0.0] * n_crops
        self.Q_p: List[float] = [0.0] * n_crops
        self.V_ref: List[float] = [0.0] * n_crops
        self.Vt: List[float] = [0.0] * n_crops

        self.Pbat: float = 0.0
        self.res_energy: float = 0.
        self.day_picked: int = 0

        self.reward_fun = lambda s, a, s_next: rwd_fun(s, a, s_next, n_crops=n_crops, weights=weights)

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

        if options is not None:
            if options["mode"] == "eval":
                year = 2018 #np.random.randint(2020, 2022) # fix the year
        else:
            year = np.random.randint(2013, 2018)

        self._update_solar_power(year)
        self._update_consumption()

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

        self.k = 0

        self.p_pv, self.p_res = self._get_disturbances()
        init_obs = self._get_obs()

        return init_obs, {}
    
    def _update_solar_power(self, year:int):
        """updates the solar power profile for a n-steps episodes in a specific year"""
        
        

        data = select_season(df_sept_to_jan, year)
        n_days = len(data)//144 -self.days_per_episode-1

        s_day_1 = np.random.randint(0,n_days)
        s_day_2 = np.random.randint(0,n_days)
        rad_1 = data["Radiación Directa Normal (estimado) [mean,W/m2]"][s_day_1*144:(s_day_1+self.days_per_episode)*144+1].values
        rad_2 = data["Radiación Directa Normal (estimado) [mean,W/m2]"][s_day_1*144:(s_day_1+self.days_per_episode)*144+1].values

        temp_1 = data["Temperatura [mean,C]"][s_day_1*144:(s_day_1+self.days_per_episode)*144+1].values
        temp_2 = data["Temperatura [mean,C]"][s_day_2*144:(s_day_2+self.days_per_episode)*144+1].values
        alpha = np.random.rand()*0.2
        rad = rad_1*alpha + rad_2*(1-alpha)
        temp = temp_1*alpha + temp_2*(1-alpha)
    
        self.pv_power_data = solar_power(rad, temp)
        return
    
    def _update_consumption(self):
        """updates the residential consuption profile for n-steps episodes"""
        s_day_1 = np.random.randint(0,95 - self.days_per_episode)
        s_day_2 = np.random.randint(0,95 - self.days_per_episode)

        profile_1 = self.demand_data[s_day_1*144:(s_day_1+self.days_per_episode)*144+1]
        profile_2 = self.demand_data[s_day_2*144:(s_day_2+self.days_per_episode)*144+1]
        alpha = np.random.rand()*0.2
        self.res_power_data = profile_1*alpha + profile_2*(1-alpha)
        return
        

    def step(self, action: np.ndarray, mode: str = "train") -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :param mode: mode of the environment. Can be "train" or "eval"
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Store the previous values of the variables to compute the reward
        obs = self._get_obs()
        self.p_pv, self.p_res = self._get_disturbances()
        _ = self.micro_grid.next_step(actions=action, disturbances=[self.p_pv, self.p_res])

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

    def _get_disturbances(self) -> Tuple[float, float]:
        """Returns the disturbances for the current time step
        :return: Tuple of (p_pv, p_demanded)"""
        
        p_pv = self.pv_power_data[self.k]
        p_demanded = self.res_power_data[self.k]
        return p_pv, p_demanded

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

    def _get_obs(self) -> npt.NDArray[np.float32]:
        """Return the observation of the environment"""
        mg_obs = self.micro_grid.get_observation()
        p_pv, p_load = self._get_disturbances()

        observation = np.concatenate((self.V_refs, mg_obs, [p_pv, p_load]))
        return observation
    
    



class NormalisedMG(gym.Wrapper):
    def __init__(self, env: MicrogridEnv):
        super().__init__(env)
        self.env = env
        self.transform = generate_t_matrix(env.n_crops)
        self.prev_action: np.ndarray = np.zeros(self.env.action_space.shape[-1])
        low = np.matmul(self.transform, env.observation_space.low)
        high = np.matmul(self.transform, env.observation_space.high)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.zeros(2*self.env.n_crops),
                                       high=np.ones(2*self.env.n_crops),
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
        assert isinstance(env, NormalisedMG), "only NormalisedMG can be used with EvalMG"
        super().__init__(env)
        self.env:NormalisedMG = env

    def reset(self, seed: int = None, options:dict = None):
        return self.env.reset(options = {"mode": "eval"})
    
    def step(self, action):
        return self.env.step(action)


def rwd_fun(s, a, s_next, n_crops=1, weights:np.ndarray[float, int] = np.array([1.0, 4.0, 1.0])):
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
