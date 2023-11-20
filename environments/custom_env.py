import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
import numpy as np


class Custom_env(ABC, gym.Env):
    """
    Abstract class for a custom environment
    """

    def __init__(self, action_low, action_high):
        self.action_low = action_low
        self.action_high = action_high

    def sample_trajectory(self, policy, max_steps: int = 288):
        """
        Sample a trajectory from the environment using the policy.
        :param policy: Policy to be used
        :param max_steps: Maximum number of steps to be taken
        :return: states and actions of the trajectory
        """
        policy.eval()
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        actions = np.zeros((max_steps, self.action_space.shape[0]))
        x0, _ = self.reset()
        states[0] = x0
        for i in range(max_steps):
            action = policy(states[i])
            action = action.squeeze()
            actions[i] = action.detach().cpu().numpy()
            x_next, _, _, _, _ = self.step(actions[i])
            states[i + 1] = x_next
        policy.train()
        return states, actions

    @abstractmethod
    def show_sample(self, policy):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """
        pass

