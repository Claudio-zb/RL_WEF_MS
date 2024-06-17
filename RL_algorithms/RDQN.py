from RL_algorithms.RL_algorithm import RL_algorithm
from environments.custom_env import Custom_env
from utils_functions.ReplayMemory import PrioritizedReplayMemory, Transition
from utils_functions.EMS_networks import Q_network
import copy
import torch
from itertools import count
import numpy as np
import torch.nn as nn
from torch.optim import RAdam

class RDQN(RL_algorithm):
    """Implementation of rainbow dqn"""
    def __init__(self, env:Custom_env, options = None) -> None:
        super().__init__()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = PrioritizedReplayMemory(100_000)
        self.critic = Q_network(self.obs_dim, self.action_dim)
        self.target_critic = copy.deepcopy(self.critic)
        self.optimizer = RAdam(self.critic.parameters(), lr=self.lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_hyperparams_()

    def one_ep_training(self, i_episode: int) -> tuple[float, float, float, float, float]:
        s, info = self.env.reset()
        for t in count():
            action = self.select_action(s, i_episode)
            s_next, reward, done, truncated, info = self.env.step(self.env.map_action(action))
            reward = torch.tensor(reward, device=self.device)
            isDone = done or truncated
            self.buffer.push(s, action, s_next, reward, isDone)
            if done or truncated:
                pass
            pass

        self.optimize_model()

    def select_action(self, state, i_episode: int) -> torch.Tensor:
        """sample an action from e policy"""
        sample = np.random.rand()
        eps_threshold = self.eps_end + (self.eps_init - self.eps_end) * np.exp(-1. * (i_episode) / (self.eps_decay))
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            self.ep_random_steps += 1
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

            
    def optimize_model(self):
        # generate batch of data
        transitions, indices = self.buffer.sample()
        batch = Transition(*zip(*transitions))

        s = torch.cat(batch.state)
        a = torch.cat(batch.action)
        s_next = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        done = torch.cat(batch.isdone)
        not_done = torch.logical_not(done).to(self.device)

        with torch.no_grad():
            # compute target Q value
            current_Q = self.critic(s).gather(1, a)
            a_next = self.critic(s_next).max(1).indices
            Q_next = self.target_critic(s).gather(1, a_next)
            target_Q = rewards.unsqueeze(-1) + not_done.unsqueeze(-1) * self.gamma * Q_next
        
        critic_loss = nn.MSELoss(current_Q, target_Q)
        
        # Update priorities
        priorities = critic_loss + 1e-5
        self.memory.update_priorities(indices, priorities.detach().cpu().numpy())
        critic_loss = critic_loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_value_(self.critic.parameters(), 100)
        self.optimizer.step()
        self.scheduler.step()

        return
    
    def _init_hyperparams_(self):
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.eps_init = 0.95
        self.eps_end = 0.5
        self.eps_decay = 1
        self.lr = 1e-3

