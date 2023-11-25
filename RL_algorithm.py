from abc import ABC, abstractmethod
from torch import nn


class RL_algorithm(ABC):
    """
    Abstract class for RL algorithms
    """

    @abstractmethod
    def learn(self, n_iter: int) -> tuple[dict, nn.Module]:
        """
        Learn the policy for the environment
        :param n_iter:
        :return: training statistics and the learned policy
        """
        pass

    @abstractmethod
    def show_trajectory(self, policy):
        """
        Render the environment using the current policy
        :param policy:
        :return:
        """
        pass
