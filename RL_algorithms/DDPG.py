import random
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

from utils_functions.EMS_networks import Continous_Q_network, ActorNN
from utils_functions.ReplayMemory import ReplayMemory, Transition, PrioritizedReplayMemory
from environments.custom_env import Custom_env
import numpy as np
from RL_algorithms.RL_algorithm import RL_algorithm
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Any
from matplotlib.figure import Figure

class DDPG(RL_algorithm):
    def __init__(self, env:Custom_env, options = None) -> None:
        self.env = env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        #self._init_hyperparameters(options)

        self.memory = PrioritizedReplayMemory(10000)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.critic_net = Continous_Q_network(self.obs_dim, self.action_dim, device).to(device)
        self.target_critic_net = Continous_Q_network(self.obs_dim, self.action_dim, device).to(device)

        self.policy_net = ActorNN(self.obs_dim, self.action_dim, device).to(device)
        self.target_policy_net = ActorNN(self.obs_dim, self.action_dim, device).to(device)

        # make sure the weights are the same
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())


        self.critic_optimizer = AdamW(self.critic_net.parameters(), lr=self.lr, amsgrad=True)
        self.policy_optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9999)
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.9999)
        self._training_stats = None

        self.stats_fig = Figure(figsize=(5, 4), dpi=100)
        self.stats_axs = self.stats_fig.subplots(3, 1)

        self.ep_steps = 0
        self.ep_random_steps = 0
        #self.scaler = StandardScaler()
        #self._init_scaler()

        # priority experience replay
        self.replay_period = 4
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.delta = 0

    def one_ep_training(self, i_episode: int = 0):
        '''Train the agent for one episode'''
        device = self.device
        state, info = self.env.reset()
        self.ep_random_steps = 0
        self.ep_steps = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state = self.normalize_state(state)
        ep_rewards = []
        for t in count():  # begin episode
            self.critic_net.eval()
            self.policy_net.eval()

            action = self.policy_net(state).detach().cpu().numpy().flatten()
            action = np.clip(action + np.random.normal(0, 0.1, self.action_dim), self.action_low, self.action_high)
            
            observation, reward, terminated, truncated, _ = self.env.step(action)
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
                    transitions, indices = self.memory.sample(self.BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)

                    q_values_target = self.target_critic_net(state_batch, action_batch).mean().item()
                    q_values_policy = self.critic_net(state_batch, action_batch).mean().item()
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
            target_critic_params = self.target_critic_net.state_dict()
            critic_params = self.critic_net.state_dict()
            for key in target_critic_params:
                target_critic_params[key] = target_critic_params[key] * self.TAU + critic_params[key] * (
                        1 - self.TAU)
            self.target_critic_net.load_state_dict(target_critic_params)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 − τ )θ′
            target_policy_params = self.target_policy_net.state_dict()
            policy_params = self.policy_net.state_dict()
            for key in target_policy_params:
                target_policy_params[key] = target_policy_params[key] * self.TAU + policy_params[key] * (
                        1 - self.TAU)
            self.target_policy_net.load_state_dict(target_policy_params)

            if done:
                break

        return mean_ep_rwd, std_ep_rwd, action_randomness, q_values_target, q_values_policy
    
    def optimize_model(self) -> None:
        '''Optimize the model'''
        self.critic_net.train()
        self.policy_net.train()

        if len(self.memory) < self.BATCH_SIZE:
            return 
        transitions, indices = self.memory.sample(self.BATCH_SIZE)
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.critic_net_(state_batch, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device, dtype=torch.float32)
        a_next = self.target_policy_net(non_final_next_states)
        q_next = self.target_critic_net(non_final_next_states, a_next)
        with torch.no_grad():
            next_state_values[non_final_mask] = q_next
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(-1)

        # Update Critic Network
        # Compute Huber loss
        #criterion = nn.SmoothL1Loss(reduction='none')
        criterion = nn.MSELoss(reduction='none')
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Update priorities
        priorities = loss + 1e-5
        self.memory.update_priorities(indices, priorities.detach().cpu().numpy())

        # Average loss
        loss = loss.mean()

        # Optimize the model
        self.critic_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # Update policy network

        policy_loss = -self.critic_net(state_batch, self.policy_net(state_batch)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_scheduler.step()
    
        return

