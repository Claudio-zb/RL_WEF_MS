from typing import SupportsFloat, Any, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import ActType, ObsType


class basic_env(gym.Env):
    def __init__(self):
        self.observation_space = spaces.box.Box(low=np.array([0, 0, 0], dtype=np.float32),
                                                high=np.array([np.inf, 5, 4], dtype=np.float32),
                                                dtype=np.float32)

        self.action_space = spaces.box.Box(low=np.array([0, 0], dtype=np.float32),
                                           high=np.array([1, 1], dtype=np.float32),
                                           dtype=np.float32)
        self.k = 0
        self.req = 0.0
        self.V_tank = 0.0
        self.V_irr = 0.0
        self.error_int = 0.0
        self.w_len = 10
        self.historical_error = np.zeros(5)

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ) -> Tuple[ObsType, dict]:
        self.V_irr = 0.0
        self.V_tank = np.random.rand() * 5
        self.req = np.random.rand() * 3 + 1
        self.k = 0
        norm_error = (self.req - self.V_irr) / self.req
        self.historical_error = np.ones(self.w_len) * norm_error
        self.error_int = np.sum(self.historical_error) / self.w_len
        return np.array([self.V_irr, self.V_tank, self.req]), None

    def step(
            self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict]:
        done = False
        penalty = 0
        Irr = action[0] / 10
        Q_p = action[1] / 10
        if self.V_tank <= 5:
            self.V_tank = self.V_tank + Q_p
        if self.V_tank >= 0:
            c_V_tank = np.maximum(self.V_tank - Irr, 0.0)
            if c_V_tank == 0 and Irr > 0:
                penalty = Irr * 10
            self.V_irr += self.V_tank - c_V_tank
            self.V_tank = c_V_tank

        norm_error = (self.req - self.V_irr) / self.req
        self.historical_error[self.k % self.w_len] = norm_error

        self.error_int = np.sum(abs(self.historical_error)) / self.w_len

        obs = np.array([self.V_irr, self.V_tank, self.req])

        reward = - norm_error if norm_error > 0 else norm_error  # 1-norm_error if norm_error > 0 else 1+norm_error

        self.k += 1
        if self.k == 100:
            done = True

        return obs, reward, False, done, {}


class DuWrapper(gym.Env):
    def __init__(self, env: gym.Env):
        self.observation_space = spaces.box.Box(low=np.array([0, 0, 0, -1, -1], dtype=np.float32),
                                                high=np.array([np.inf, 5, 4, 1, 1], dtype=np.float32),
                                                dtype=np.float32)

        self.action_space = spaces.box.Box(low=np.array([-1, -1], dtype=np.float32),
                                           high=np.array([1, 1], dtype=np.float32),
                                           dtype=np.float32)
        self.env = env
        self.past_action = np.zeros(self.observation_space.shape[-1], dtype=np.float32)
        self.lb = np.zeros(2)
        self.ub = np.ones(2)

    def step(
            self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        u = np.clip(self.past_action + action, self.lb, self.ub)
        next_obs, reward, terminated, truncated, info = self.env.step(u)
        self.past_action = u
        next_obs = np.concatenate((next_obs, self.past_action))

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.past_action = np.zeros(self.action_space.shape[-1], dtype=np.float32)
        observation, info = self.env.reset()
        observation = np.concatenate((observation, self.past_action))
        return observation, info



if __name__ == "__main__":
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    env = basic_env()
    env = DuWrapper(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100_000, log_interval=10)
    model.save("simple_env")

    vec_env = model.get_env()

    del model  # remove to demonstrate saving and loading

    model = TD3.load("simple_env")
