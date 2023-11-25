import numpy as np
import torch
from gymnasium import spaces
import matplotlib
import matplotlib.pyplot as plt
from environments.custom_env import Custom_env
from typing import Union

matplotlib.use('Qt5Agg')


class Quad_env(Custom_env):
    """
    Environment for a discrete time system with a quadratic regressor.
    The system is defined as: y(k+1) = A*y(k) + c*y(k-1)^2 + b*u(k)
    """

    def __init__(self, toRender=False, continuous=False):
        self.action_values = np.array([-20, -10, -7, -5, -3, -1, -.5, -.1, 0, .1, .5, 1, 3, 5, 7, 10, 20],
                                      dtype=np.float32)
        action_low = -25.0
        action_high = 25.0
        super().__init__(action_low, action_high, continuous=continuous)

        if continuous:
            self.action_space = spaces.Box(low=action_low,
                                           high=action_high,
                                           shape=(1,))
        else:
            self.action_space = spaces.Discrete(17)

        # obs_space: y, y_prev, y_prev_prev, u_prev, u_prev_prev, y_ref
        self.observation_space = spaces.Box(low=-np.array([100, 100, 100, 20, 20, 10], dtype=np.float32),
                                            high=np.array([100, 100, 100, 20, 20, 10], dtype=np.float32),
                                            shape=(6,))
        self._reference = np.array([0.1])
        self.initial_state = np.array([1.0])
        self._current_state = np.array([1.0])
        self._prev_state = self.initial_state
        self._prev_prev_state = np.array([0.0])
        self._prev_u = np.array([0.0])
        self._prev_prev_u = np.array([0.0])
        self._max_episode_steps = 200
        self.isContinuous = continuous

        self.A = 1.1
        self.b = .8
        self.steps = 0
        self.max_steps = 200
        self.toRender = toRender

        self.window_size = 512
        self.window = None
        self.clock = None

        plt.ion()
        fig, axs = plt.subplots(2, 1)
        self.fig = fig
        self.axs = axs
        self.fig.suptitle('Energy Management System')
        self.fig.tight_layout()
        self.fig.set_size_inches(10, 10)

    def _get_obs(self) -> np.ndarray:
        """
        Get the observation of the environment
        :return:
        """
        obs = np.concatenate((self._current_state,
                              self._prev_state,
                              self._prev_prev_state,
                              self._prev_u,
                              self._prev_prev_u,
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
        self.initial_state = self.np_random.uniform(low=-5, high=5, size=1)
        self._current_state = self.initial_state
        self._prev_state = self.initial_state
        self._prev_prev_state = self.initial_state
        self._prev_u = np.array([0.0])
        self._prev_prev_u = np.array([0.0])
        self._reference = self.np_random.uniform(low=-5, high=5, size=1)

        while np.abs(self._reference - self.initial_state) < 0.1:
            # Ensure that the initial state is not close to the reference
            self._reference = self.np_random.uniform(low=-6, high=6, size=1)

        self.steps = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, False  # info

    def step(self, policy_output: Union[np.ndarray, torch.Tensor]) -> tuple:

        if self.isContinuous:
            d_action = policy_output
        else:
            d_action = self.map_action(policy_output)

        if d_action.ndim != 1:
            d_action = d_action.flatten()
        # We calculate the next state based on dynamics of the system
        u = d_action + self._prev_u

        terminated = False
        bound = 100

        next_state = self.A * self._current_state - 0.1 * self._prev_state ** 2 + self.b * u

        cum_error = ((next_state - self._reference) + (self._current_state - self._reference) +
                     (self._prev_state - self._reference) + (self._prev_prev_state - self._reference))

        # Next timestep
        self._prev_prev_state = self._prev_state
        self._prev_state = self._current_state
        self._current_state = next_state
        self._prev_prev_u = self._prev_u
        self._prev_u = u

        self.steps += 1

        if self.steps >= self.max_steps:
            truncated = True

        else:
            truncated = False

        reward = -(self._reference - self._current_state) ** 2 - 0.1*d_action ** 2

        if abs(self._current_state) > bound:
            self._current_state = np.array([bound])*np.sign(self._current_state)
            terminated = True
            # reward = -np.array([200])

        # We need to return four values: observation, reward, done, info

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def show_sample(self, policy):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """

        states, actions = self.sample_trajectory(policy)
        for ax in self.axs.flat:
            ax.clear()
        t = np.arange(states.shape[0])
        self.axs[0].step(t, states[:, 0], label='y', where='post')
        self.axs[0].plot(states[:, 5], label='y_ref')
        self.axs[0].set_title('Output')
        self.axs[0].legend()

        self.axs[1].step(t[0:-1], np.cumsum(actions), label='u', where='post')
        self.axs[1].set_title('Input')
        self.axs[1].legend()

        plt.pause(0.1)
        plt.show(block=False)

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
