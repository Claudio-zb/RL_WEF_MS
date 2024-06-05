import gymnasium as gym
import torch
from gymnasium import spaces
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Callable


class Custom_env(ABC, gym.Env):
    """
    Abstract class for a custom environment
    """

    @abstractmethod
    def sample_trajectory(self,
                          policy: Callable,
                          scaler: StandardScaler = None,
                          max_steps: int = 288,
                          rew_fun=None,
                          initial_conditions: dict = None,
                          options: dict = None):
        """
        Sample a trajectory from the environment using the given policy.
        :param policy: Policy to be used
        :param scaler: Scaler to be used for the states
        :param max_steps: Maximum number of steps to be taken
        :param rew_fun: Reward function to be used
        :param initial_conditions: Initial conditions for the environment
        :return: states and actions of the trajectory
        """
        pass

    @abstractmethod
    def show_sample(self, policy, scaler):
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

class ContinousCustomEnv(Custom_env):
    """
    Abstract Class for continous action custom environments
    """
    def __init__(self, action_low, action_high):
        self.action_low = action_low
        self.action_high = action_high
    
    def sample_trajectory(self,
                          policy: Callable,
                          scaler: StandardScaler = None,
                          max_steps: int = 288,
                          rew_fun=None,
                          initial_conditions: dict = None,
                          options: dict = None):

        policy.eval()
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        
        if initial_conditions is not None:
            x0 = self.load_initial_conditions(initial_conditions)
        elif options is not None:
            x0, _ = self.reset(options={"t_init": 0})
        else:
            x0, _ = self.reset()
        states[0] = x0

        actions = np.zeros((max_steps, self.action_space.shape[0]))

        for i in range(max_steps):

            if scaler is not None:
                action = policy(scaler.transform([states[i]]))
            else:
                action = policy((states[i]))
            action = action.squeeze().detach().cpu().numpy()
            actions[i] = action

            x_next, _, terminated, truncated, _ = self.step(action)
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
        policy.train()
        rewards = np.zeros_like(actions)
        if rew_fun is not None:
            for i in range(len(actions)):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])

        return states, actions, rewards


class DiscreteCustomEnv(Custom_env):
    """
    Abstract Class for continous action custom environments
    """

    def __init__(self, action_values):
        self.action_values = action_values
    
    def sample_trajectory(self,
                          policy: Callable,
                          scaler: StandardScaler = None,
                          max_steps: int = 288,
                          rew_fun=None,
                          initial_conditions: dict = None,
                          options: dict = None):
        policy.eval()
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        
        if initial_conditions is not None:
            x0 = self.load_initial_conditions(initial_conditions)
        elif options is not None:
            x0, _ = self.reset(options={"t_init": 0})
        else:
            x0, _ = self.reset()
        states[0] = x0

        a_shape = self.action_values.shape
        if len(a_shape) > 1:
            actions = np.zeros((max_steps, self.action_values.shape[1]))
        else:
            actions = np.zeros((max_steps, 1))

        for i in range(max_steps):
            if scaler is not None:
                action = policy(scaler.transform(states[i:i+1])).max(1).indices.view(1, 1)
            else:
                action = policy(states[i]).max(1).indices.view(1, 1) # the index
            actions[i] = self.action_values[action]

            x_next, _, terminated, truncated, _ = self.step(action)
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
        policy.train()
        rewards = np.zeros(max_steps)
        if rew_fun is not None:
            for i in range(len(actions)):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])
        else:
            rew_fun = self.reward_fun
            for i in range(len(actions)):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])
        return states, actions, rewards
