import gymnasium as gym
import numpy as np
from environments.EMS_constants import T_matrix


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
