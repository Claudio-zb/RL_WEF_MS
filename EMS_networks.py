import torch
from torch import nn
import numpy as np


class ActorNN(nn.Module):
    """
    Policy network
    """

    def __init__(self, input_dim, output_dim):
        super(ActorNN, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).cuda()
        shared_output = self.shared_fc(obs)
        # d_I_1_output = self.d_I_1(shared_output)
        # d_I_2_output = self.d_I_2(shared_output)
        # d_Q_p_output = self.d_Q_p(shared_output)
        # Pbat_output = self.Pbat(shared_output)
        return shared_output


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
