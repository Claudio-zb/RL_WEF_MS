from abc import ABC, abstractmethod
from torch import nn
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import os
from collections import deque
import random
import math



class RL_algorithm(ABC):
    """
    Abstract class for RL algorithms. All RL algorithms should inherit from this class. The main methods are:
    - learn: learn the policy for the environment. It executes the training loop. Returns the training statistics and
    the learned policy.
    - show_trajectory: render the environment using the current policy.
    - one_ep_training: execute one episode of training. Returns the mean reward, std reward, action randomness, 
                       q_values_target and q_values_policy.
    - get_policy: get the policy.
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

    @abstractmethod
    def get_policy(self) -> nn.Module:
        """
        Get the policy
        :return:
        """
        pass

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()