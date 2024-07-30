import torch
from torch import nn
import numpy as np
from torch.nn import init
from torch.nn import functional as F
import RL_algorithms.RL_algorithm as utils


class ActorNN(nn.Module):
    """
    Policy network
    """

    def __init__(self, input_dim: int, output_dim: int,
                 upper_bound: torch.Tensor, lower_bound: torch.Tensor, device):
        super(ActorNN, self).__init__()
        self.device = device
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
        self._init_weights()

    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_fc(obs)
        shared_output = (torch.tanh(shared_output)/2 + .5) * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        return shared_output
    
    def get_action(self, obs:np.ndarray) -> np.ndarray:
        shared_output = self.shared_fc(obs)
        shared_output = (torch.tanh(shared_output)/2 + .5) * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        return shared_output

    
    def get_action(self, obs:np.ndarray) -> np.ndarray:
        """Returns the action given the observation. Necessary for use in the environment"""
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        obs = obs.unsqueeze(0)
        return self.forward(obs).detach().cpu().numpy().flatten()

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
        width = 256
        self.structure = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
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
            nn.Linear(input_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, output_dim)
        )
        self._init_weights()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
            if obs.dim() == 1: 
                obs = obs.unsqueeze(0)
        return self.structure(obs.to(self.device))
    
    def _init_weights(self):
        for layer in self.structure:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(DuelingQNetwork, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = 128
        self.Value = nn.Linear(self.width, 1)
        self.Advantange = nn.Linear(self.width, output_dim)
        self.fc1 = nn.Linear(input_dim, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
    
    def forward(self, obs)->tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            if obs.dim() == 1: 
                obs = obs.unsqueeze(0)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.Value(x)
        advantage = self.Advantange(x)
        Q = value + (advantage - torch.mean(advantage, dim=-1, keepdim=True))
        return Q
class TD3Critic(nn.Module):
    def __init__(self, input_dim, output_dim, width = 128):
        super(TD3Critic, self).__init__()
        self.width = width
        self.q1_sequence = nn.Sequential(
            nn.Linear(input_dim + output_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1)
        )
        self.q2_sequence = nn.Sequential(
            nn.Linear(input_dim + output_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1)
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
        self.width = 256
        self.structure = nn.Sequential(
            nn.Linear(input_dim + output_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1)
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

class RecurrentPolicy(nn.Module):
    """ Actor network with LSTM structure. This network is used to learn the policy function. 
    It uses a LSTM layer to process the sequence of observations and a MLP to process the last observation."""
    def __init__(self, input_dim, output_dim: int,
                 upper_bound: torch.Tensor, lower_bound: torch.Tensor, device):
        super(RecurrentPolicy, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.lstm_input = nn.LSTM(input_dim, 128, batch_first=True)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        
        self.hidden_structure = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.to(device)
    def forward(self, obs:torch.Tensor):
        """Forward pass of the network. It processes the sequence of observations and returns the action"""
        out, _ = self.lstm_input(obs)
        if out.dim() > 2:
            unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            last_obs = unpacked[:,-1,:]
        else:
            last_obs = out[:,-1,:]
        x = self.hidden_structure(last_obs)
        return (torch.tanh(x)/2 + .5) * (self.upper_bound - self.lower_bound) + self.lower_bound
    
    def get_action(self, obs:np.ndarray) -> np.ndarray:
        """Returns the action given the observation. Necessary for use in the environment"""
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() == 1: 
            obs = obs.unsqueeze(0).unsqueeze(0)
        else:
            obs = obs.unsqueeze(0)
        out, _ = self.lstm_input(obs)
        x = self.hidden_structure(out.squeeze(0).squeeze(0))
        return x.detach().cpu().numpy()
    
class RecurrentCritic(nn.Module):
    """Critic network with LSTM structure. This network is used to learn the Q function. 
    It uses a LSTM layer to process the sequence of observations and a MLP to process the last observation."""
    def __init__(self, input_dim, action_space, device):
        super(RecurrentCritic, self).__init__()
        self.device = device
        self.input_dim = input_dim

        self.lstm_input = nn.LSTM(input_dim, 128, batch_first=True)
        self.hidden_structure = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, obs:torch.Tensor, action:torch.Tensor):
        """Forward pass of the network. It processes the sequence of observations and returns the Q values"""
        out, _ = self.lstm_input(obs)
        if out.dim() > 2:
            unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            last_obs = unpacked[:,-1,:]
        else:
            last_obs = out[:,-1,:]
        return self.hidden_structure(torch.cat([last_obs, action], dim=-1))

    
class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)