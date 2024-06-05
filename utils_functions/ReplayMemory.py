from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'isdone'))

PPOTransition = namedtuple('PPOTransition',
                           ('state', 'action', 'next_state', 'reward', 'reward_to_go'))


class ReplayMemory(object):
    """
    Replay memory class for storing transitions
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PPOReplayMemory(object):
    """
    Replay memory class for storing transitions
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(PPOTransition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(1)
        self.memory[self.position] = Transition(*args)
        # Set max priority
        self.priorities[self.position] = max(self.priorities, default=1)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        if priorities.ndim == 1:
            priorities = np.expand_dims(priorities, 1)
        probs = priorities ** beta
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs.squeeze(-1))
        samples = [self.memory[idx] for idx in indices]

        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority[0]

    def __len__(self):
        return len(self.memory)
