import random
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, RAdam

from utils_functions.EMS_networks import Q_network, DuelingQNetwork
from utils_functions.ReplayMemory import Transition, PrioritizedReplayMemory
from utils_functions.ReplayMemory import ReplayBuffer, PrioritizedReplayBuffer
from environments.custom_env import CustomEnv
import numpy as np
from RL_algorithms.RL_algorithm import RL_algorithm
from typing import Union, Tuple, Any
from matplotlib.figure import Figure
import copy


class DDQN(RL_algorithm):
    """
    Double DQN with prioritized experience replay
    """

    def __init__(self, env: CustomEnv, options=None) -> None:
        """Initializes the DQN algorithm class. 
        It creates the policy and target networks, the optimizer and the memory."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._init_hyperparameters(options)
        self.global_steps = 1

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.transform_dim = env.transform.shape[0]

        self.transform = torch.tensor(env.transform).to(device)

        self.memory = PrioritizedReplayBuffer(state_size=self.transform_dim, 
                                              action_size=1, # 1 becuase it's discrete action algorithm
                                              buffer_size=100_000)

        self.policy_net = Q_network(self.transform_dim, self.action_dim, device).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)

        self.optimizer = RAdam(self.policy_net.parameters(), lr=self.lr)
        self._training_stats = None

        self.stats_fig = Figure(figsize=(5, 4), dpi=100)
        self.stats_axs = self.stats_fig.subplots(3, 1)

        self.ep_steps = 0
        self.ep_random_steps = 0

        # priority experience replay
        self.replay_period = 4
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.delta = 0


    def learn(self, n_episodes: int) -> tuple[dict, nn.Module]:
        self.EPS_DECAY = n_episodes/5
        device = self.device
        self._training_stats = {"mean_episode_rewards": np.zeros(n_episodes),
                                "std_episode_rewards": np.zeros(n_episodes),
                                "episodes": np.array(np.arange(n_episodes)),
                                "Q_values_policy": np.zeros(n_episodes),
                                "Q_values_target": np.zeros(n_episodes),
                                "steps": np.zeros(n_episodes),
                                "action_randomness": np.zeros(n_episodes)}

        for i_episode in range(n_episodes):
            # Initialize the environment and get it's state
            if i_episode % 10 == 0:
                self.env.show_sample(self.target_net, self.scaler)
            state, info = self.env.reset()
            self.ep_random_steps = 0
            self.ep_steps = 0
            ep_rewards = []
            mean_rwd, std_rwd, randomness, q_val, q_val_target = self.one_ep_training(i_episode)
            self._training_stats["mean_episode_rewards"][i_episode] = mean_rwd
            self._training_stats["std_episode_rewards"][i_episode] = std_rwd
            self._training_stats["action_randomness"][i_episode] = randomness
            self._training_stats["Q_values_policy"][i_episode] = q_val
            self._training_stats["Q_values_target"][i_episode] = q_val_target
            self.update_training_plots(i_episode)   
            

        return self._training_stats, self.policy_net, self.scaler

    def optimize_model(self) -> None:
        self.policy_net.train()
        if self.global_steps < self.BATCH_SIZE:
            return 
        batch, weights, tree_idxs = self.memory.sample(self.BATCH_SIZE)
        state, action, reward, next_state, done = batch
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            a_next = self.policy_net(next_state).max(1).indices
            next_state_action_values = self.target_net(next_state).gather(1, a_next.unsqueeze(1)).squeeze(1)
            expected_state_action_values = reward + self.gamma * next_state_action_values * (1-done)
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

        td_error = torch.abs(state_action_values - expected_state_action_values).detach()
        loss = torch.mean((state_action_values - expected_state_action_values)**2 * weights.to(self.device))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 80)
        self.optimizer.step()

        with torch.no_grad():
            # Update the frozen target models
            for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        self.memory.update_priorities(tree_idxs, td_error.cpu().squeeze(1).numpy())
        return loss.detach().cpu().numpy().item()

    def select_action(self, state, i_episode: int) -> torch.Tensor:

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * (i_episode) / (self.EPS_DECAY))
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            self.ep_random_steps += 1
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def _init_hyperparameters(self, options=None):
        """
        This method initializes the hyperparameters of the algorithm
        :param options: a dictionary containing the hyperparameters
        :return:
        """
        if options is None:
            self.BATCH_SIZE = 256
            self.gamma = 0.99
            self.lr = 1e-3
            self.EPS_START = 0.95
            self.EPS_END = 0.05
            self.EPS_DECAY = 100 #100
            self.TAU = 0.001

        else:
            self.BATCH_SIZE = options['BATCH_SIZE']
            self.gamma = options['gamma']
            self.n_epochs_critic = options['n_epochs_critic']
            self.lr = options['lr']
            self.EPS_START = options['EPS_START']
            self.EPS_END = options['EPS_END']
            self.EPS_DECAY = options['EPS_DECAY']
            self.TAU = options['TAU']

    def map_action(self, action: torch.Tensor) -> np.ndarray:
        """
            Map the action from the policy to the action of the environment
            :param action:
            :return:
            """
        index = action.item()
        return np.array([self.action_values[index]])

    def update_training_plots(self, episode: int):
        """Updates the training plots"""

        self.stats_axs[0].clear()
        self.stats_axs[1].clear()
        self.stats_axs[2].clear()

        window_length = 10

        if episode < 1000:

            mean_rewards = self._training_stats["mean_episode_rewards"][0:episode]
            #draw the std deviation
            std_rewards = self._training_stats["std_episode_rewards"][0:episode]

            windowed_rewards = np.convolve(mean_rewards, np.ones(10) / 10, mode='valid')

            # durations = self._training_stats["steps"][0:episode]
            target_values = self._training_stats["Q_values_target"][0:episode]
            policy_values = self._training_stats["Q_values_policy"][0:episode]
            # windowed_duration = np.convolve(durations, np.ones(10) / 10, mode='valid')

            self.stats_axs[0].plot(range(0, episode), mean_rewards, label="Mean reward")
            self.stats_axs[0].plot(range(window_length - 1, episode), windowed_rewards, label="Windowed reward")
            self.stats_axs[0].fill_between(range(0, episode), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

            self.stats_axs[1].plot(range(0, episode), target_values, label="Target values")
            self.stats_axs[1].plot(range(0, episode), policy_values, label="Policy value")
            # self.stats_axs[1].plot(range(window_length - 1, episode), windowed_duration, label="Windowed duration")

            self.stats_axs[2].plot(range(0, episode), self._training_stats["action_randomness"][0:episode],
                                   label="Action randomness")
        else:
            mean_rewards = self._training_stats["mean_episode_rewards"][episode - 1000:episode]
            # durations = self._training_stats["steps"][episode - 1000:episode]
            target_values = self._training_stats["Q_values_target"][episode - 1000:episode]
            policy_values = self._training_stats["Q_values_policy"][episode - 1000:episode]
            windowed_rewards = np.convolve(mean_rewards, np.ones(window_length) / window_length, mode='valid')
            # windowed_duration = np.convolve(durations, np.ones(10) / 10, mode='valid')

            self.stats_axs[0].plot(range(episode - 1000, episode), mean_rewards, label="Mean reward")
            self.stats_axs[0].plot(range(episode - 1000 + window_length - 1, episode),
                                   windowed_rewards, label="Windowed reward")

            self.stats_axs[1].plot(range(episode - 1000, episode), target_values, label="Target Values")
            self.stats_axs[1].plot(range(episode - 1000, episode), policy_values, label="Policy Values")

            # self.stats_axs[1].plot(range(episode - 1000 + window_length - 1, episode),
            #                       windowed_duration, label="Windowed duration")
            self.stats_axs[2].plot(range(episode - 1000, episode),
                                   self._training_stats["action_randomness"][episode - 1000:episode],
                                   label="Action randomness")
        self.stats_axs[1].legend()
        self.stats_axs[0].legend()

    def save(self):
        """ Saves the training statistics and the model"""
        pass

    def get_training_fig(self) -> plt.Figure:
        return self.stats_fig
    
    def one_ep_training(self, i_episode: int = 0) -> Tuple[float, float, float, float, float]:
        """Train the agent for one episode and return the statistics of the episode
        :param i_episode: the index of the episode
        :return: the mean reward, the standard deviation of the rewards, the action randomness, the Q values of the target and policy networks"""
        device = self.device
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        self.ep_random_steps = 0
        self.ep_steps = 0
        ep_rewards = []
        for t in count():  # begin episode
            t_state = torch.matmul(self.transform, state.T).T
            self.policy_net.eval()
            action = self.select_action(t_state, i_episode)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            self.ep_steps += 1
            ep_rewards.append(reward)
            reward = torch.tensor(reward, device=device)
            done = terminated or truncated
                
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            t_next_state = torch.matmul(self.transform, next_state.T).T

            # Store the transition in memory
            self.memory.add((t_state, action, reward, t_next_state, int(done)))

            # Move to the next state
            state = next_state
            self.optimize_model()

            self.global_steps += 1

            if done:
                ep_rewards = np.array(ep_rewards).flatten()
                mean_ep_rwd = ep_rewards.mean()
                std_ep_rwd = ep_rewards.std()
                action_randomness = self.ep_random_steps / (t + 1)
                break

        return mean_ep_rwd, std_ep_rwd, action_randomness, 0, 0#q_values_target, q_values_policy
    
    def get_policy(self) -> nn.Module:
        return self.target_net
    
class DuelingDDQN(DDQN):
    """Dueling double DQN with prioritized experienced replay"""
    def __init__(self, env: CustomEnv, options=None):
        super().__init__(env, options)
        self.policy_net = DuelingQNetwork(self.transform_dim, self.action_dim, self.device).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)

        self.optimizer = RAdam(self.policy_net.parameters(), lr=self.lr, decoupled_weight_decay=True)
        