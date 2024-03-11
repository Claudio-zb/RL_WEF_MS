import gymnasium as gym
import torch
from gymnasium import spaces
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler


class Custom_env(ABC, gym.Env):
    """
    Abstract class for a custom environment
    """

    def __init__(self, action_low, action_high, continuous: bool = False):
        self.action_low = action_low
        self.action_high = action_high
        self.isContinuous = continuous
        self.action_values = None

    def sample_trajectory(self,
                          policy,
                          scaler: StandardScaler = None,
                          max_steps: int = 288,
                          rew_fun=None,
                          initial_conditions: dict = None):
        """
        Sample a trajectory from the environment using the policy.
        :param policy: Policy to be used
        :param scaler: Scaler to be used for the states
        :param max_steps: Maximum number of steps to be taken
        :param rew_fun: Reward function to be used
        :param initial_conditions: Initial conditions for the environment
        :return: states and actions of the trajectory
        """
        policy.eval()
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        rewards = np.zeros(max_steps)
        if self.isContinuous:
            actions = np.zeros((max_steps, self.action_space.shape[0]))
        else:
            a_shape = self.action_values.shape
            if len(a_shape) > 1:
                actions = np.zeros((max_steps, self.action_values.shape[1]))
            else:
                actions = np.zeros((max_steps, 1))
        if initial_conditions is not None:
            x0 = self.load_initial_conditions(initial_conditions)
        else:
            x0, _ = self.reset()
        states[0] = x0
        for i in range(max_steps):
            if self.isContinuous:
                if scaler is not None:
                    action = policy(scaler.transform([states[i]]))
                else:
                    action = policy((states[i]))
                action = action.squeeze().detach().cpu().numpy()
                actions[i] = action
            else:
                action = policy(states[i]).max(1).indices.view(1, 1)
                actions[i] = self.action_values[action]

            x_next, _, terminated, truncated, _ = self.step(action)
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
        policy.train()
        if rew_fun is not None:
            for i in range(max_steps):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])

        return states, actions, rewards

    @abstractmethod
    def show_sample(self, policy):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """
        pass

    @abstractmethod
    def map_action(self, policy_output: torch.Tensor) -> np.ndarray:
        """
        Map the policy's output to the action space
        :param policy_output:
        :return: the action to be taken
        """
        pass

    @abstractmethod
    def load_initial_conditions(self, initial_conditions: dict) -> np.ndarray:
        """
        Load the initial conditions for the environment
        :param initial_conditions: dictionary containing the initial conditions
        :return: initial observation
        """
        pass
