import random
from itertools import count
import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from EMS_networks import Q_network
from utils_functions.ReplayMemory import ReplayMemory, Transition
from environments.custom_env import Custom_env
import numpy as np
from RL_algorithms.RL_algorithm import RL_algorithm


class DQN(RL_algorithm):
    """
    This class implements the DQN algorithm. It can handle discrete action spaces.
    It is based on Adam Paszke and Mark Towers' implementation:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def show_trajectory(self, policy):
        self.env.show_sample(policy)

    def __init__(self, env: Custom_env, options=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.memory = ReplayMemory(1000)
        self._init_hyperparameters(options)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy_net = Q_network(self.obs_dim, self.action_dim).to(device)
        self.target_net = Q_network(self.obs_dim, self.action_dim).to(device)
        self.optimizer = Adam(self.policy_net.parameters(), lr=self.lr)
        self._training_stats = {"mean_episode_rewards": [],
                                "std_episode_rewards": [],
                                "episodes": [],
                                "steps": []}

    def learn(self, n_updates: int) -> tuple[dict, nn.Module]:
        device = self.device
        n_episodes = self.episodes_per_batch * n_updates
        self._training_stats = {"mean_episode_rewards": np.zeros(n_episodes),
                                "std_episode_rewards": np.zeros(n_episodes),
                                "episodes": np.zeros(n_episodes),
                                "steps": np.zeros(n_episodes)}
        k_update = 0
        for i_episode in range(n_episodes):
            # Initialize the environment and get it's state
            if i_episode % 100 == 0:
                self.env.show_sample(self.policy_net)
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                self.policy_net.eval()
                action = self.select_action(state, t)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                reward = torch.tensor(reward, device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                if len(self.memory) > 15:
                    self.memory.plot_memory()
                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                # Optimization is done every batch_size steps
                k_update = self.optimize_model(k_update)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    plt.plot(self.episode_durations)
                    break
        return self._training_stats, self.policy_net

    def optimize_model(self, k_update: int) -> int:
        """
        This method optimizes the policy network
        :param k_update: the number of updates performed so far
        :return: the number of updates performed so far"""

        self.policy_net.train()
        if len(self.memory) < self.BATCH_SIZE:
            return k_update
        
        if k_update % 50 == 0:
            transitions = self.memory.sample(self.BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.cat(batch.reward)

            mean_reward = torch.mean(reward_batch)
            std_reward = torch.std(reward_batch)

            try:
                self._training_stats["mean_episode_rewards"][k_update] = mean_reward.item()
                self._training_stats["std_episode_rewards"][k_update] = std_reward.item()
            except:
                pass

            #print(f"Iteration: {k_update}")
            #print(f"Mean reward: {mean_reward.item()}")
            #print("------------------------")
            

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            return k_update + 1
        else:
            return k_update

    def select_action(self, state, steps_done: int) -> torch.Tensor:

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        np.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def _init_hyperparameters(self, options=None):
        """
        This method initializes the hyperparameters of the algorithm
        :param options: a dictionary containing the hyperparameters
        :return:
        """
        if options is None:
            self.BATCH_SIZE = 100
            self.gamma = 0.91
            self.n_epochs_critic = 10
            self.n_epochs_policy = 6
            self.clip = 0.2
            self.lr = 0.001
            self.ent_coef = 0.01
            self.max_grad_norm = 0.5
            self.lam = 0.95
            self.EPS_START = 0.9
            self.EPS_END = 0.05
            self.EPS_DECAY = 200
            self.TAU = 0.001
            self.episode_durations = []
            self.episodes_per_batch = 10
            self.max_timesteps_per_episode = 200

        else:
            self.BATCH_SIZE = options['BATCH_SIZE']
            self.max_timesteps_per_episode = options['max_timesteps_per_episodes']
            self.episodes_per_batch = options['episodes_per_batch']
            self.gamma = options['gamma']
            self.n_epochs_critic = options['n_epochs_critic']
            self.n_epochs_policy = options['n_epochs_policy']
            self.clip = options['clip']
            self.lr = options['lr']
            self.ent_coef = options['ent_coef']
            self.max_grad_norm = options['max_grad_norm']
            self.lam = options['lam']
