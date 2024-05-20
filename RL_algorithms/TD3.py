import random
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

from utils_functions.EMS_networks import Continous_Q_network, ActorNN
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
        self._init_hyperparameters(options)

        self.memory = ReplayMemory(10000)
        self.env: ContinousCustomEnv = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low

        self.critic_1:nn.Module = Continous_Q_network(self.obs_dim, self.action_dim, device).to(device)
        self.target_critic_1:nn.Module = Continous_Q_network(self.obs_dim, self.action_dim, device).to(device)

        self.critic_2:nn.Module = Continous_Q_network(self.obs_dim, self.action_dim, device).to(device)        
        self.target_critic_2:nn.Module = Continous_Q_network(self.obs_dim, self.action_dim, device).to(device)

        self.policy_net:nn.Module = ActorNN(self.obs_dim, self.action_dim, self.action_high, self.action_low).to(device)
        self.target_policy_net:nn.Module = ActorNN(self.obs_dim, self.action_dim, self.action_high, self.action_low).to(device)

        # make sure the weights are the same
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())


        self.critic_1_optimizer = AdamW(self.critic_1.parameters(), lr=self.lr, amsgrad=True)
        self.critic_2_optimizer = AdamW(self.critic_2.parameters(), lr=self.lr, amsgrad=True)
        self.policy_optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
        self.critic_1_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_1_optimizer, gamma=0.9999)
        self.critic_2_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_2_optimizer, gamma=0.9999)
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.9999)
        
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
            self.critic_1.eval()
            self.critic_2.eval()
            self.policy_net.eval()

            action = self.policy_net(state)
            action = torch.clamp(action + torch.randn_like(action) * self.policy_noise, 
                                 torch.tensor(self.action_low).to(device), 
                                 torch.tensor(self.action_high).to(device))

            observation, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy().flatten())
            self.ep_steps += 1
            ep_rewards.append(reward)
            reward = torch.tensor(reward, device=device)
            done = terminated or truncated

            if done:
                next_state = None
                ep_rewards = np.array(ep_rewards).flatten()
                mean_ep_rwd = ep_rewards.mean()
                std_ep_rwd = ep_rewards.std()
                action_randomness = self.ep_random_steps / (t + 1)

                if len(self.memory) >= self.BATCH_SIZE:
                    transitions = self.memory.sample(self.BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)

                    q_target_1_values = self.critic_1(state_batch, action_batch).mean().item()
                    q_target_2_values = self.critic_2(state_batch, action_batch).mean().item()
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # Optimization is done every batch_size steps
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 − τ )θ′
            
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            if done:
                break

        return mean_ep_rwd, std_ep_rwd, action_randomness, q_target_1_values, q_target_2_values
    
    def optimize_model(self) -> None:
        '''Optimize the model'''
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()
        self.policy_net.train()
        self.target_policy_net.train()

        if len(self.memory) < self.BATCH_SIZE:
            return 
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).to(self.device).detach()
        reward_batch = torch.cat(batch.reward)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device, dtype=torch.float32)
        a_next = self.target_policy_net(non_final_next_states)

        q_next_1 = self.target_critic_1(non_final_next_states, a_next).squeeze(-1)
        q_next_2 = self.target_critic_2(non_final_next_states, a_next).squeeze(-1)

        q_next = torch.min(q_next_1, q_next_2)

        with torch.no_grad():
            next_state_values[non_final_mask] = q_next 
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.detach() * self.gamma) + reward_batch.squeeze(-1)

        # Update Both Critic Networks
        criterion = nn.MSELoss(reduction='none')

        self.critic_1_optimizer.zero_grad()
        state_action_values_1 = self.critic_1(state_batch, action_batch)
        loss1 = criterion(state_action_values_1, expected_state_action_values.unsqueeze(1)).mean()
        loss1.backward()
        torch.nn.utils.clip_grad_value_(self.critic_1.parameters(), 100)
        self.critic_1_optimizer.step()
        self.critic_1_scheduler.step()

        self.critic_2_optimizer.zero_grad()
        state_action_values_2 = self.critic_2(state_batch, action_batch)
        loss2 = criterion(state_action_values_2, expected_state_action_values.unsqueeze(1)).mean()
        loss2.backward()
        torch.nn.utils.clip_grad_value_(self.critic_2.parameters(), 100)
        self.critic_2_optimizer.step()
        self.critic_2_scheduler.step()

        # Update policy network
        if self.ep_steps % self.policy_freq == 0:
            policy_loss = -self.critic_1(state_batch, self.policy_net(state_batch)).mean()
            policy_loss = policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.policy_scheduler.step()
    
        return
    def _init_hyperparameters(self, options=None):
        """Initialize the hyperparameters of the algorithm"""
        if options is None:
            self.gamma = 0.99
            self.lr = 0.001
            self.BATCH_SIZE = 128
            self.TAU = 0.005
            self.policy_noise = 0.2
            self.noise_clip = 0.5
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