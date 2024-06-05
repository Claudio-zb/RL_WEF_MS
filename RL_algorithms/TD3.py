import random
import copy
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

from utils_functions.EMS_networks import *
from utils_functions.ReplayMemory import ReplayMemory, Transition
from environments.custom_env import ContinousCustomEnv
import numpy as np
from RL_algorithms.RL_algorithm import RL_algorithm
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Any
from matplotlib.figure import Figure

class TD3(RL_algorithm):
    def __init__(self, env:ContinousCustomEnv, options = None) -> None:
        self.env = env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.memory = ReplayMemory(10000)
        self.env: ContinousCustomEnv = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        self._init_hyperparameters(options)

        self.critic:nn.Module = TD3Critic(self.obs_dim, self.action_dim).to(device)
        self.target_critic:nn.Module = copy.deepcopy(self.critic)


        self.policy_net:nn.Module = ActorNN(self.obs_dim, self.action_dim, self.action_high, self.action_low).to(device)
        self.target_policy_net:nn.Module = copy.deepcopy(self.policy_net)


        self.critic_optimizer = AdamW(self.critic.parameters(), lr=self.lr, amsgrad=True)
        self.policy_optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.99)
        
        self._training_stats = None

        self.stats_fig = Figure(figsize=(5, 4), dpi=100)
        self.stats_axs = self.stats_fig.subplots(3, 1)

        self.ep_steps = 0
        self.ep_random_steps = 0
        

    def one_ep_training(self, i_episode: int = 0):
        '''Train the agent for one episode'''
        device = self.device
        state, info = self.env.reset()
        self.ep_random_steps = 0
        self.ep_steps = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        ep_rewards = []
        for t in count():  # begin episode
            self.critic.eval()
            self.policy_net.eval()

            action = self.policy_net(state)
            observation, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy().flatten())
            self.ep_steps += 1
            ep_rewards.append(reward)
            reward = torch.tensor(reward, device=device)
            done = terminated or truncated

            if done:
                ep_rewards = np.array(ep_rewards).flatten()
                mean_ep_rwd = ep_rewards.mean()
                std_ep_rwd = ep_rewards.std()
                action_randomness = self.ep_random_steps / (t + 1)

                if len(self.memory) >= self.BATCH_SIZE:
                    transitions = self.memory.sample(self.BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)

                    q_values = self.critic.Q1(state_batch, action_batch).mean().item()
                    q_target_values = self.target_critic.Q1(state_batch, action_batch).mean().item()
                else:
                    q_values = 0
                    q_target_values = 0
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # Optimization is done every batch_size steps
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 − τ )θ′

            if done:
                break

        return mean_ep_rwd, std_ep_rwd, action_randomness, q_values, q_target_values
    
    def optimize_model(self) -> None:
        '''Optimize the model'''
        self.critic.train()
        self.target_critic.train()
        self.policy_net.train()
        self.target_policy_net.train()

        if len(self.memory) < self.BATCH_SIZE:
            return 
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state = torch.cat(batch.state)
        action = torch.cat(batch.action).to(self.device).detach()
        reward = torch.cat(batch.reward).type(torch.float32)
        next_state = torch.cat(batch.next_state)
        done = torch.tensor(batch.isdone)
        not_done = torch.logical_not(done).to(self.device)

        with torch.no_grad():
            noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
				self.target_policy_net(next_state) + noise
			).clamp(self.action_low, self.action_high)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.max(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(-1) + not_done.unsqueeze(-1) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

		# Delayed policy updates
        if self.ep_steps % self.policy_freq == 0:

			# Compute actor losse
            actor_loss = -self.critic.Q1(state, self.policy_net(state)).mean()

            # Optimize the actor 
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            return
    def _init_hyperparameters(self, options=None):
        """Initialize the hyperparameters of the algorithm"""
        if options is None:
            self.gamma = 0.95   
            self.lr = 0.0001
            self.BATCH_SIZE = 256
            self.TAU = 0.005
            self.policy_noise = 0.2*self.action_high
            self.noise_clip = 0.5*self.action_high
            self.policy_freq = 2
        else:
            self.gamma = options['gamma']
            self.lr = options['lr']
            self.BATCH_SIZE = options['BATCH_SIZE']
            self.TAU = options['TAU']
            self.policy_noise = options['policy_noise']
            self.noise_clip = options['noise_clip']
            self.policy_freq = options['policy_freq']

        return

    def learn(self, n_iter: int) -> tuple[dict, nn.Module]:
        training_stats = {
            "mean_episode_rewards": [],
            "std_episode_rewards": [],
            "Q_values_target": [],
            "Q_values_policy": [],
            "action_randomness": [],
            "steps": []
        }
        for i in range(n_iter):
            mean_ep_rwd, std_ep_rwd, action_randomness, q_target_1_values, q_target_2_values = self.one_ep_training(i)
            training_stats["mean_episode_rewards"].append(mean_ep_rwd)
            training_stats["std_episode_rewards"].append(std_ep_rwd)
            training_stats["Q_values_target"].append(q_target_1_values)
            training_stats["Q_values_policy"].append(q_target_2_values)
            training_stats["action_randomness"].append(action_randomness)
            training_stats["steps"].append(self.ep_steps)
            
            if i % 10 == 0:
                self.update_training_plots(i)
        return training_stats, self.target_policy_net
    
    def get_training_fig(self) -> Figure:
        return self.stats_fig

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
        return

    def get_policy(self) -> nn.Module:
        return self.target_policy_net