from abc import ABC, abstractmethod
from torch import nn
from matplotlib.figure import Figure


class RL_algorithm(ABC):
    """
    Abstract class for RL algorithms. All RL algorithms should inherit from this class. The main methods are:
    - learn: learn the policy for the environment. It executes the training loop. Returns the training statistics and
    the learned policy.
    - show_trajectory: render the environment using the current policy
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
    

    @abstractmethod
    def update_training_plots(self, i_episode: int):
        """
        Update the training plots
        :return:
        """
        pass
    
    @abstractmethod
    def get_training_fig(self) -> Figure:
        """
        Get the training figure
        :return:
        """
        pass 
    
    @abstractmethod
    def one_ep_training(self, i_episode: int) -> tuple[float, float, float, float, float]:
        """
        Execute one episode of training
        :param i_episode: episode number
        :return mean_ep_rwd, std_ep_rwd, action_randomness, q_values_target, q_values_policy:
        """
        pass