import torch
from torch import nn
import numpy as np
from torch.nn import init


class ActorNN(nn.Module):
    """
    Policy network
    """

    def __init__(self, input_dim: int, output_dim: int,
                 upper_bound: np.ndarray, lower_bound: np.ndarray):
        super(ActorNN, self).__init__()
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32).cuda()
        self.lower_bound = torch.tensor(lower_bound, dtype=torch.float32).cuda()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self._init_weights()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).cuda()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        shared_output = self.shared_fc(obs)
        x = torch.sigmoid(shared_output)
        x = x * (self.upper_bound - self.lower_bound) + self.lower_bound
        return x

    def _init_weights(self):
        for layer in self.shared_fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


class ValueNN(nn.Module):
    """
    Value function network
    """

    def __init__(self, input_dim):
        super(ValueNN, self).__init__()

        self.structure = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
            obs = obs.unsqueeze(0).cuda()
        return self.structure(obs)


class Q_network(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(Q_network, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.structure = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim)
        )
        self._init_weights()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
            obs = obs.unsqueeze(0).to(self.device)
        return self.structure(obs)
    
    def _init_weights(self):
        for layer in self.structure:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
