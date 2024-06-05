import numpy as np
import torch
from gymnasium import spaces
import matplotlib
import matplotlib.pyplot as plt
from environments.custom_env import Custom_env
from environments.custom_env import ContinousCustomEnv
from typing import Union
from matplotlib.figure import Figure


class Quad_env(ContinousCustomEnv):
    """
    Environment for a discrete time system with a quadratic regressor.
    The system is defined as: y(k+1) = A*y(k) + c*y(k-1)^2 + b*u(k)
    """

    def __init__(self, render:bool = True):
        
        action_low = -1.0
        action_high = 1.0
        super().__init__(action_low, action_high)

        
        self.action_space = spaces.Box(low=action_low,
                                       high=action_high,
                                       shape=(1,))

        # obs_space: y, y_prev, y_prev_prev, u_prev, u_prev_prev, y_ref
        self.observation_space = spaces.Box(low=-np.array([100, 100, 20, 10], dtype=np.float32),
                                            high=np.array([100, 100, 20, 10], dtype=np.float32),
                                            shape=(4,))
        
        self._reference = np.array([0.1])
        self.initial_state = np.array([1.0])
        self._current_state = np.array([1.0])
        self._prev_state = self.initial_state
        self._prev_u = np.array([0.0])


        self.A = 0.85
        self.b = .8
        self.steps = 0
        self.max_steps = 50

        self.window_size = 512
        self.window = None
        self.clock = None

        if render:
            plt.ion()
            self.fig, self.axs = plt.subplots(3, 1)
            self.fig.suptitle('System Response')
            self.fig.tight_layout()
            self.fig.set_size_inches(10, 10)

    def _get_obs(self) -> np.ndarray:
        """
        Get the observation of the environment
        :return:
        """
        obs = np.concatenate((self._current_state,
                              self._prev_state,
                              self._prev_u,
                              self._reference))
        return obs.flatten()

    def seed(self, seed):
        return

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._current_state - self._reference, ord=2
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.initial_state = self.np_random.uniform(low=-6, high=6, size=1)
        self._current_state = self.initial_state
        self._prev_state = self.initial_state
        self._prev_u = np.array([0.0])
        self._reference = self.np_random.uniform(low=-6, high=6, size=1)

        while np.abs(self._reference - self.initial_state) < 0.1:
            # Ensure that the initial state is not close to the reference
            self._reference = self.np_random.uniform(low=-6, high=6, size=1)

        self.steps = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, False  # info

    def step(self, policy_output: Union[np.ndarray, torch.Tensor]) -> tuple:
        
        
        d_action = policy_output

        if d_action.ndim != 1:
            d_action = d_action.flatten()
        # We calculate the next state based on dynamics of the system
        u = d_action + self._prev_u

        terminated = False
        bound = 100

        next_state = self.A * self._current_state - 0.1 * self._prev_state ** 1 + self.b * u

        # Next timestep
        self._prev_state = self._current_state
        self._current_state = np.clip(next_state, -bound, bound)
        self._prev_u = u

        self.steps += 1

        
        if self.steps >= self.max_steps:
            truncated = True

        else:
            truncated = False

        reward = -(1 - np.exp(-0.01*((self._current_state + self._prev_state - 2*self._reference) ** 2 + 10*(self._current_state - self._prev_state) ** 2 + 0.1*u**2))) 

        # We need to return four values: observation, reward, done, info

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def show_sample(self, policy, scaler = None):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """

        states, actions, rewards = self.sample_trajectory(policy, 
                                                          rew_fun=reward_fun)
        for ax in self.axs.flat:
            ax.clear()
        t = np.arange(states.shape[0])
        self.axs[0].step(t, states[:, 0], label='y', where='post')
        self.axs[0].plot(states[:, -1], label='y_ref')
        self.axs[0].set_title('Output')
        self.axs[0].legend()

        self.axs[1].step(t[0:-1], np.cumsum(actions), label='u', where='post')
        self.axs[1].step(t[0:-1], actions, label='du', where='post')
        self.axs[1].set_title('Input')
        self.axs[1].legend()

        self.axs[2].step(t[0:-1], rewards, label='reward', where='post')
        self.axs[2].set_title('Reward')
        self.axs[2].legend()


    def close(self):
        """
        Close the environment
        :return:
        """
        plt.close()
        plt.ioff()

    def map_action(self, action: torch.Tensor) -> np.ndarray:
            """
            Map the action from the policy to the action of the environment
            :param action:
            :return:
            """
            index = action.item()
            return np.array([self.action_values[index]])
    
    def load_initial_conditions(self, initial_conditions: dict) -> np.ndarray:
        return None
    
def reward_fun(s,a,s_next):
    next_x = s_next[0]
    x = s[0]
    ref = s[-1]
    action = s_next[2]
    reward = -(1 - np.exp(-0.001 * ((next_x + x - 2*ref) ** 2 + 10*(next_x-x)**2 + 0.1*action**2)))
    return reward


    
