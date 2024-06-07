import torch
from torch import nn
import numpy as np
from torch.nn import init
from torch.nn import functional as F


class ActorNN(nn.Module):
    """
    Policy network
    """

    def __init__(self, input_dim: int, output_dim: int,
                 upper_bound: torch.Tensor, lower_bound: torch.Tensor, device = 'cuda'):
        super(ActorNN, self).__init__()
        self.device = device
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.shared_fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self._init_weights()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        shared_output = self.shared_fc(obs)
        shared_output = torch.tanh(shared_output)*self.upper_bound/2 + .5
        
        return shared_output

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
        self.width = 128 
        self.structure = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, output_dim)
        )
        self._init_weights()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            if obs.dim() == 1: 
                obs = obs.unsqueeze(0)
        return self.structure(obs)
    
    def _init_weights(self):
        for layer in self.structure:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

class TD3Critic(nn.Module):
    def __init__(self, input_dim, output_dim, width = 128):
        super(TD3Critic, self).__init__()
        self.width = width
        self.q1_sequence = nn.Sequential(
            nn.BatchNorm1d(input_dim + output_dim),
            nn.Linear(input_dim + output_dim, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim)
        )
        self.q2_sequence = nn.Sequential(
            nn.BatchNorm1d(input_dim + output_dim),
            nn.Linear(input_dim + output_dim, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim)
        )
     
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.q1_sequence(sa)
        q2 = self.q2_sequence(sa)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_sequence(sa)
        return q1

class Continous_Q_network(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(Continous_Q_network, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = 128
        self.structure = nn.Sequential(
            nn.BatchNorm1d(input_dim + output_dim),
            nn.Linear(input_dim + output_dim, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 48),
            nn.ReLU(),
            nn.Linear(48, 1)
        )
        self._init_weights()

    def forward(self, observation, action):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
            observation = observation.unsqueeze(0).to(self.device)

        input = torch.cat([observation, action], dim=-1)
        return self.structure(input)
    
    def _init_weights(self):
        for layer in self.structure:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

class LSTM_Q_Network(nn.Module):
    """ Q network with LSTM structure. This network is used to learn the Q function. 
    It uses a LSTM layer to process the sequence of observations and a MLP to process the last observation."""
    def __init__(self, input_dim, output_dim, device):
        super(LSTM_Q_Network, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lstm_input = nn.LSTMCell(input_dim, 128)
        self.mlp_input = nn.Linear(input_dim, 128)
        self.hidden_structure = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim)
        )
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
            obs = obs.unsqueeze(1).to(self.device)
        
        # compute the features
        for i in range(obs.size(0)):
            if i == 0:
                h, c = self.lstm_input(obs[i])
            else:
                h, c = self.lstm_input(obs[i], (h, c))
        
        last_obs = obs[-1]
        mlp_output = self.mlp_input(last_obs)
        x = torch.cat([h, mlp_output], dim=1)
        return self.hidden_structure(x)
    