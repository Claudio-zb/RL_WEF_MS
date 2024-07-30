import copy
from itertools import count

import torch
import torch.nn as nn
from torch.optim import RAdam

from utils_functions.EMS_networks import *
from utils_functions.ReplayMemory import PrioritizedReplayBuffer, Transition, ReplayMemory, RReplayMemory
from environments.custom_env import ContinousCustomEnv
import numpy as np
from RL_algorithms.RL_algorithm import RL_algorithm, soft_update_params
from typing import Union, Tuple, Any
from matplotlib.figure import Figure
class TD3(RL_algorithm):
    def __init__(self, env:ContinousCustomEnv, options = None) -> None:
        self.env:ContinousCustomEnv = env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.global_steps = 0
        self.env: ContinousCustomEnv = env
        try:
            self.obs_dim = env.observation_dim
        except:
            self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        self._init_hyperparameters(options)

        self.memory = ReplayMemory(obs_dim=self.obs_dim, action_dim=self.action_dim, capacity=1_000_000, device=device)

        self.critic:nn.Module = TD3Critic(self.obs_dim, self.action_dim).to(device)
        self.target_critic:nn.Module = copy.deepcopy(self.critic)

        self.policy_net:nn.Module = ActorNN(self.obs_dim, self.action_dim, self.action_high, self.action_low, device).to(device)
        self.target_policy_net:nn.Module = copy.deepcopy(self.policy_net)

        self.critic_optimizer = RAdam(self.critic.parameters(), lr=self.lr)
        self.policy_optimizer = RAdam(self.policy_net.parameters(), lr=self.lr)
        
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=1.0)
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=1.0)
        
        self._training_stats = None

        self.stats_fig = Figure(figsize=(5, 4), dpi=100)
        self.stats_axs = self.stats_fig.subplots(3, 1)

        self.ep_steps = 0
        self.ep_random_steps = 0
        self.eps_threshold = 1.0

    def one_ep_training(self, i_episode: int = 0) -> Tuple[float, float, float, float, float]:
        '''Train the agent for one episode'''
        device = self.device
        state, info = self.env.reset()
        self.ep_random_steps = 0
        self.ep_steps = 0
        state = torch.tensor(state, dtype=torch.float32, device=device)
        ep_rewards = []
        action_randomness = self.eps_exploration
        for t in count():  # begin episode
            self.critic.eval()
            self.policy_net.eval()
            explo_noise = torch.randn(self.action_dim, device = device)*action_randomness*(self.action_high-self.action_low)
            action = torch.clamp(self.policy_net(state).to(device) + explo_noise, self.action_low, self.action_high).detach()
            observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy().flatten())
            self.ep_steps += 1
            ep_rewards.append(reward)
            reward = torch.tensor(reward, device=device)
            done = terminated or truncated

            # Store the transition in memory
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            self.memory.add(state, action, reward, next_state, int(done))
            self.optimize_model()
            state = next_state
            self.global_steps += 1
            if done:
                ep_rewards = np.array(ep_rewards).flatten()
                mean_ep_rwd = ep_rewards.mean()
                std_ep_rwd = ep_rewards.std()
                if self.global_steps >= self.BATCH_SIZE:
                    transitions = self.memory.sample(self.BATCH_SIZE)
                    state_batch, action_batch, reward, next_state, done = transitions
                    with torch.no_grad():
                        q_values = self.critic.Q1(state_batch, action_batch).mean().item()
                        q_target_values = self.target_critic.Q1(state_batch, action_batch).mean().item()
                else:
                    q_values = 0
                    q_target_values = 0
                break
        return mean_ep_rwd, std_ep_rwd, action_randomness, q_values, q_target_values
    
    def optimize_model(self) -> None:
        '''Optimize the model'''
        self.critic.train()
        self.policy_net.train()

        if self.global_steps < self.BATCH_SIZE:
            return 
        transitions = self.memory.sample(self.BATCH_SIZE)
        state, action, reward, next_state, done = transitions

        with torch.no_grad():
            noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-torch.tensor(self.noise_clip*self.action_low, dtype=torch.float32, device=self.device), 
           torch.tensor(self.noise_clip*self.action_high, dtype=torch.float32, device=self.device))

            next_action = (
				self.target_policy_net(next_state) + noise
			).clamp(self.action_low, self.action_high)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.gamma * target_Q
                
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = ((current_Q1-target_Q)**2 + (current_Q2-target_Q)**2).mean()
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 50)
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
            with torch.no_grad():
                soft_update_params(self.critic, self.target_critic, self.TAU)
                soft_update_params(self.policy_net, self.target_policy_net, self.TAU)

        return
    def _init_hyperparameters(self, options=None):
        """Initialize the hyperparameters of the algorithm"""
        if options is None:
            self.gamma = 0.5   
            self.lr = 0.0001
            self.BATCH_SIZE = 256
            self.TAU = 0.005
            self.policy_noise = 0.2*(self.action_high - self.action_low)
            self.noise_clip = 0.5
            self.policy_freq = 2
            self.eps_exploration = 0.2
        else:
            self.gamma = options['gamma']
            self.lr = options['lr']
            self.BATCH_SIZE = options['BATCH_SIZE']
            self.TAU = options['TAU']
            self.policy_noise = options['policy_noise']
            self.noise_clip = options['noise_clip']
            self.policy_freq = options['policy_freq']
            self.eps_exploration = options['eps_exploration']

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
    
class PrioritizedTD3(TD3):
    def __init__(self, env:ContinousCustomEnv, options = None) -> None:
        super(PrioritizedTD3, self).__init__(env, options)
        self.memory = PrioritizedReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], 1_000_000, discrete_action_space=False)

    def one_ep_training(self, i_episode: int = 0) -> Tuple[float, float, float, float, float]:
        '''Train the agent for one episode'''
        device = self.device
        state, info = self.env.reset()
        self.ep_random_steps = 0
        self.ep_steps = 0
        state = torch.tensor(state, dtype=torch.float32, device=device)
        ep_rewards = []
        action_randomness = self.eps_exploration
        for t in count():  # begin episode
            self.critic.eval()
            self.policy_net.eval()
            explo_noise = torch.randn(self.action_dim, device = device)*action_randomness
            action = torch.clamp(self.policy_net(state).to(device) + explo_noise, self.action_low, self.action_high).detach()
            observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy().flatten())
            self.ep_steps += 1
            ep_rewards.append(reward)
            reward = torch.tensor(reward, device=device)
            done = terminated or truncated

            # Store the transition in memory
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            self.memory.add((state, action, reward, next_state, int(done)))
            self.optimize_model()
            state = next_state
            self.global_steps += 1

            if done:
                ep_rewards = np.array(ep_rewards).flatten()
                mean_ep_rwd = ep_rewards.mean()
                std_ep_rwd = ep_rewards.std()
                if self.global_steps >= self.BATCH_SIZE:
                    batch, weights, tree_idxs = self.memory.sample(self.BATCH_SIZE)
                    state_batch, action_batch, reward, next_state, done = batch
                    with torch.no_grad():
                        q_values = self.critic.Q1(state_batch, action_batch).mean().item()
                        q_target_values = self.target_critic.Q1(state_batch, action_batch).mean().item()
                else:
                    q_values = 0
                    q_target_values = 0
                break
        return mean_ep_rwd, std_ep_rwd, action_randomness, q_values, q_target_values
    
    def optimize_model(self) -> None:
        '''Optimize the model'''
        self.critic.train()
        self.policy_net.train()

        if self.global_steps < self.BATCH_SIZE:
            return 
        batch, weights, tree_idxs = self.memory.sample(self.BATCH_SIZE)
        state, action, reward, next_state, done = batch

        with torch.no_grad():
            noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip*(self.action_high-self.action_low), self.noise_clip*(self.action_high-self.action_low))

            next_action = (
				self.target_policy_net(next_state) + noise
			).clamp(self.action_low, self.action_high)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(-1) + (1-done).unsqueeze(-1) * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        critic_loss = ((current_Q1-target_Q)**2*weights.to(self.device) + (current_Q2-target_Q)**2*weights.to(self.device)).mean()
        with torch.no_grad():
            td_error = (current_Q1 - target_Q).abs()

        self.memory.update_priorities(tree_idxs, td_error.cpu().squeeze(1).numpy())
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 80)
        self.critic_optimizer.step()

		# Delayed policy updates
        if self.ep_steps % self.policy_freq == 0:

			# Compute actor losse
            actor_loss = -self.critic.Q1(state, self.policy_net(state)).mean()
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            with torch.no_grad():
                soft_update_params(self.critic, self.target_critic, self.TAU)
                soft_update_params(self.policy_net, self.target_policy_net, self.TAU)

        return
    def get_action(self, state):
        if self.eps < np.random.rand():
            action = torch.rand(self.action_dim)*(self.action_high-self.action_low) + self.action_low
        else:
            action = self.policy_net.get_action(state)
        return action
    

class RecurrentTD3(TD3):
    """lstm critic policy version of td3 algorithm """
    
    def __init__(self, env:ContinousCustomEnv, options = None) -> None:
        super(RecurrentTD3, self).__init__(env, options)
        # defining nets again
        self.critic:nn.Module = RecurrentCritic(self.obs_dim, self.action_dim, self.device)
        self.target_critic:nn.Module = copy.deepcopy(self.critic)
        self.policy_net:nn.Module = RecurrentPolicy(self.obs_dim, self.action_dim, self.action_high, self.action_low, self.device)
        self.target_policy_net:nn.Module = copy.deepcopy(self.policy_net)

        self.memory = RReplayMemory(1_000_000, self.device)


    def one_ep_training(self, i_episode: int) -> Tuple[float]:
        '''Train the agent for one episode'''
        device = self.device
        state, info = self.env.reset()
        self.ep_random_steps = 0
        self.ep_steps = 0
        state = torch.tensor(state, dtype=torch.float32, device=device)
        ep_rewards = []
        action_randomness = self.eps_exploration
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for t in count():  # begin episode
            self.critic.eval()
            self.policy_net.eval()
            explo_noise = np.random.randn(self.action_dim)*action_randomness*(self.action_high-self.action_low)
            action = np.clip(self.policy_net.get_action(state) + explo_noise, self.action_low, self.action_high)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.ep_steps += 1
            ep_rewards.append(reward)
            reward = torch.tensor(reward, device=device)
            done = terminated or truncated

            # Store the transition in memory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(int(done))

            self.memory.add(state, action, reward, next_state, int(done))

            self.optimize_model()
            state = next_state
            self.global_steps += 1
            if done:
                ep_rewards = np.array(ep_rewards).flatten()
                mean_ep_rwd = ep_rewards.mean()
                std_ep_rwd = ep_rewards.std()
                if self.global_steps >= self.BATCH_SIZE:
                    transitions = self.memory.sample(self.BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    state_batch = torch.stack(batch.state)
                    action_batch = torch.stack(batch.action)
                    reward = torch.stack(batch.reward)
                    next_state = torch.stack(batch.next_state)
                    done = torch.tensor(batch.isdone, dtype=torch.long, device=device).unsqueeze(1)
                    with torch.no_grad():
                        q_values = self.critic.Q1(state_batch, action_batch).mean().item()
                        q_target_values = self.target_critic.Q1(state_batch, action_batch).mean().item()
                else:
                    q_values = 0
                    q_target_values = 0
                break
        return mean_ep_rwd, std_ep_rwd, action_randomness, q_values, q_target_values

    def learn(self, n_iter: int) -> Tuple[dict, nn.Module]:
        return super().learn(n_iter)
    
    def get_policy(self) -> nn.Module:
        return self.target_policy_net
    
    def get_training_fig(self) -> Figure:
        return self.stats_fig
    
    def update_training_plots(self, episode: int) -> None:
        return super().update_training_plots(episode)