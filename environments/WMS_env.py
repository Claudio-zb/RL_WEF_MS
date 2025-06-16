import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Callable

from environments.Cultivates import Cultivates

class CultivateEnv(gym.Env):
    def __init__(self):
        self.weather_data: pd.DataFrame = pd.read_csv("environments/Data/WMS/extracted_data.csv")
        self.cultivates: Cultivates = Cultivates()
        self.n_crops: int = len(self.cultivates.crops)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(low=0.0, high=1.0, shape=(11 * self.n_crops,),
                                                                dtype=np.float32)
        self.action_space: gym.spaces.Box = gym.spaces.Box(low=0.0, high=20.0, shape=(self.n_crops,), dtype=np.float32)
        self.reward_function: Callable = lambda s, a, s_next: reward_function(s, a, s_next, self.n_crops)
        self.initial_year: int = None
        self.days_since_plantation: int = 0
        self.index: int = 0
        self.spec = gym.envs.registration.EnvSpec(id="CultivateEnv", entry_point="CultivateEnv")

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        if options is not None:
            if options["mode"] == "eval":
                self.initial_year = np.random.randint(2012, 2017)
            if options["mode"] == "test":
                self.initial_year = 2018
                
        else:
            self.initial_year = np.random.randint(1981, 2011)
        # then get the index of the day of the year of that year
        doy = min([crop.plantation_day for crop in self.cultivates.crops])  
        self.days_since_plantation = 0

        self.index = int(self.weather_data[(self.weather_data["doy"] == doy) & (self.weather_data["year"] == self.initial_year)].index.values.item())
        #self.weather_data = self.global_data[(self.global_data["year"] == self.initial_year) | (self.global_data["year"] == self.initial_year + 1)]
        #weather_data = self.weather_data.loc[(self.weather_data["doy"] == self.cultivates.doy) & (self.weather_data["year"] == self.initial_year)].iloc[0].to_dict()
        
        dict_obs, _ = self.cultivates.start()

        array_obs = obs_dict_2_obs_array(dict_obs)
        return array_obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated, truncated = False, False

        daily_weather_data = self.weather_data.iloc[self.index + self.days_since_plantation].to_dict()
        prev_obs = obs_dict_2_obs_array(self.cultivates.get_obs())
        self.days_since_plantation += 1            

        dict_obs, _ = self.cultivates.step(action/1000, daily_weather_data)
        array_obs = obs_dict_2_obs_array(dict_obs)

        for crop in self.cultivates.crops:
            if crop.is_active():  # if any crop is active, the episode is not terminated
                break
            terminated = True  # all crops are inactive, so the episode is terminated

        rew = self.reward_function(prev_obs, action, array_obs)

        return array_obs, rew, terminated, truncated, {}

    def render(self, mode='human'):
        pass

def reward_function(s: np.ndarray, a: np.ndarray, s_next: np.ndarray, n_crops) -> float:

    Ks = sum([s_next[i+7] for i in range(n_crops)])
    return Ks - sum(a)/15

class NormalizedWMS(gym.Wrapper):
    "Normalizes the action space and adds the relative yield as observation"
    def __init__(self, env: CultivateEnv, days_ahead: int = 1, reward_weigths:np.ndarray = None):
        super(NormalizedWMS, self).__init__(env)
        self.env: CultivateEnv = env
        self.n_crops = self.env.n_crops
        self.days_ahead:int = days_ahead
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.env.n_crops,), 
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
                                                shape=(12 * self.env.n_crops + days_ahead,), 
                                                dtype=np.float32)
        self.relative_yield:np.ndarray = np.ones(self.n_crops, dtype=np.float32)
        self.days:int = 1
        self.prev_obs_: np.ndarray = np.zeros(self.n_crops * 9 + days_ahead)
        if reward_weigths is None:
            self.reward_function = lambda s, a, s_next: reward_function2(s, a, s_next, self.n_crops)
        else:
            self.reward_function = lambda s, a, s_next: reward_function2(s, a, s_next, self.n_crops, reward_weigths)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        self.days = 1
        obs, info = self.env.reset(seed, options)
        self.relative_yield = np.ones(self.n_crops, dtype=np.float32)
        
        obs_ = np.zeros(len(obs) + 1 + self.days_ahead, dtype=np.float32)
        for i in range(self.n_crops):
            self.relative_yield[i] = (obs[(i+1)*7] * self.relative_yield[i])
            obs_[i*11:(i+1)*11] = obs  # asign the values from the observation (length = 11)
            obs_[8] = 0.0 if obs_[8] < 3 else 1.0 # normalize the drought indicator
            obs_[9] = obs_[9] / 114  # normalize the time component
            obs_[(i+1)*11] = self.relative_yield[i]**(1/self.days) #  add the relative yield as the 12th component

        # now we need to add the predictions
        index = self.env.index 
        future_precipitations = self.env.weather_data.iloc[index:index + self.days_ahead, 1:]["precipitation"].values.flatten()
        # add uncertainty to the future precipitation
        future_precipitations = np.abs(np.random.normal(future_precipitations, 0.1*future_precipitations))
        obs_[self.n_crops*11:] = future_precipitations
        self.prev_obs_ = obs_

        return obs_ , info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_ = action*20.0

        obs, _, terminated, truncated, info = self.env.step(action_)
        self.days += 1
        obs_ = np.zeros(len(obs) + 1 + self.days_ahead, dtype=np.float32)
        for i in range(self.n_crops):
            self.relative_yield[i] = (obs[(i+1)*7] * self.relative_yield[i])
            obs_[i*10:(i+1)*11] = obs  # asign the values from the observation (length = 10)
            obs_[8] = 0.0 if obs_[8] < 3 else 1.0 # normalize the drought indicator
            obs_[9] = obs_[9] / 114  # normalize the time component
            obs_[(i+1)*11] = self.relative_yield[i]**(1/self.days) #  add the relative yield as the 12th component

        # now we need to add the predictions
        index = self.env.index + self.env.days_since_plantation
        future_precipitations = self.env.weather_data.iloc[index:index + self.days_ahead, 1:]["precipitation"].values.flatten()
        obs_[self.n_crops*11:] = future_precipitations
        rew = self.reward_function(self.prev_obs_, action, obs_)

        self.prev_obs_ = obs_
        
        return obs_, rew, terminated, truncated, {}
    
class EvalWMS(gym.Wrapper):
    def __init__(self, env:NormalizedWMS):
        assert isinstance(env, NormalizedWMS), "EvalWMS wrapper can only be used with NormalizedWMS"
        super().__init__(env)
        self.env:NormalizedWMS = env

    def reset(self, seed: int = None, options:dict = None):
        return self.env.reset(options = {"mode": "eval"})
    
    def step(self, action):
        return self.env.step(action)
    
class TestWMS(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, seed: int = None, options:dict = None):
        return self.env.reset(options = {"mode": "test"})
    
    def step(self, action):
        return self.env.step(action)
    
def reward_function2(s: np.ndarray, a: np.ndarray, s_next: np.ndarray, n_crops, 
                     weights = np.array([1.0, 1.0, 0.333])) -> float:
    reward = 0.0
    for i in range(n_crops):
        Ks = s_next[(i+1)*7]
        Ky = s_next[(i+1)*10]
        delta_Ks = s_next[(i+1)*7] - s[(i+1)*7]
        reward += weights[0]*(1-Ky*(1-Ks)) + weights[1]*np.clip(delta_Ks, -np.inf, 0.0)
    return reward - sum(a)*weights[2]
    
def obs_dict_2_obs_array(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Turns an observation dictionary into a flattened array"""
    return np.array([obs[crop_name] for crop_name in obs.keys()]).flatten()