import random
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

from utils_functions.EMS_networks import Q_network
from utils_functions.ReplayMemory import ReplayMemory, Transition, PrioritizedReplayMemory
from environments.custom_env import Custom_env
import numpy as np
from RL_algorithms.RL_algorithm import RL_algorithm
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Any
from matplotlib.figure import Figure


class DQN(RL_algorithm):
    """
    This class implements the DQN algorithm. It can handle discrete action spaces.
    It is based on Adam Paszke and Mark Towers' implementation:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, env: Custom_env, options=None) -> None:
        """Initializes the DQN algorithm class. 
        It creates the policy and target networks, the optimizer and the memory."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._init_hyperparameters(options)

        self.memory = PrioritizedReplayMemory(10000)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy_net = Q_network(self.obs_dim, self.action_dim, device).to(device)
        self.target_net = Q_network(self.obs_dim, self.action_dim, device).to(device)

        self.optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        self._training_stats = None

        self.stats_fig = Figure(figsize=(5, 4), dpi=100)
        self.stats_axs = self.stats_fig.subplots(3, 1)

        self.ep_steps = 0
        self.ep_random_steps = 0
        self.scaler = StandardScaler()
        self._init_scaler()

        # priority experience replay
        self.replay_period = 4
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.delta = 0



    def _init_scaler(self):
        """Initialize the scaler with the mean and std of the observations."""
        n_samples = 10000
        sample_states = np.zeros((n_samples, self.obs_dim))
        for idx in range(n_samples):  # Collect 10000 samples
            state, _ = self.env.reset()
            state[-1] = np.random.randint(0, 2) 
            state[-2] = np.random.rand()*2 - 1 
            state[-3] = np.random.rand() *2 - 1 
            sample_states[idx] = state
        self.scaler.fit(sample_states)

    def normalize_state(self, state: Union[np.array, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize a state with the scaler.
        :param state: The state to normalize
        :return: The normalized state. If the input is a tensor, the output is a tensor. 
        If the input is a numpy array, the output is a numpy array."""

        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy().squeeze()
            state = self.scaler.transform([state])[0]
            return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            return self.scaler.transform([state])[0]

    def learn(self, n_episodes: int) -> tuple[dict, nn.Module, StandardScaler]:
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
                self.env.show_sample(self.target_net)
            state, info = self.env.reset()
            self.ep_random_steps = 0
            self.ep_steps = 0
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            state = self.normalize_state(state)
            ep_rewards = []
            if i_episode >= 2000:
                print("Time to debug!")
            for t in count():  # begin episode
                self.policy_net.eval()
                action = self.select_action(state, i_episode)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                self.ep_steps += 1
                ep_rewards.append(reward)
                reward = torch.tensor(reward, device=device)
                done = terminated or truncated

                if done:
                    next_state = None
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
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:  # end episode
                    ep_rewards = np.array(ep_rewards).flatten()
                    self._training_stats["mean_episode_rewards"][i_episode] = ep_rewards.mean()
                    self._training_stats["std_episode_rewards"][i_episode] = ep_rewards.std()
                    self._training_stats["steps"][i_episode] = t + 1
                    self._training_stats["action_randomness"][i_episode] = self.ep_random_steps / (t + 1)

                    if len(self.memory) >= self.BATCH_SIZE:
                        transitions, indices = self.memory.sample(self.BATCH_SIZE)
                        batch = Transition(*zip(*transitions))
                        state_batch = torch.cat(batch.state)

                        self._training_stats["Q_values_target"][i_episode] = self.target_net(state_batch).max(
                            1).values.mean().item()
                        self._training_stats["Q_values_policy"][i_episode] = self.policy_net(state_batch).max(
                            1).values.mean().item()

                    print(f"Episode {i_episode} finished after {t + 1} timesteps")
                    print(f"Mean reward: {ep_rewards.mean()}")
                    print(f"Std of reward: {ep_rewards.std()}")
                    print(f"Action randomness: {self.ep_random_steps / (t + 1)}")

                    if i_episode > 10:
                        self.update_training_plots(i_episode)
                    break

        return self._training_stats, self.policy_net, self.scaler

    def optimize_model(self) -> None:
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
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device, dtype=torch.float32)
        a_next = self.policy_net(non_final_next_states).max(1).indices
        q_next = self.target_net(non_final_next_states)
        with torch.no_grad():
            next_state_values[non_final_mask] = q_next.gather(1, a_next.unsqueeze(1)).squeeze()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(-1)

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
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.scheduler.step()

        return

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
            self.BATCH_SIZE = 128
            self.gamma = 0.95
            self.lr = 1e-3
            self.EPS_START = 0.95
            self.EPS_END = 0.001
            self.EPS_DECAY = 100
            self.TAU = 0.005

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
            self.policy_net.eval()
            action = self.select_action(state, i_episode)
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

                    q_values_target = self.target_net(state_batch).max(1).values.mean().item()
                    q_values_policy = self.policy_net(state_batch).max(1).values.mean().item()
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
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                        1 - self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

            if done:
                break

        return mean_ep_rwd, std_ep_rwd, action_randomness, q_values_target, q_values_policy
    

