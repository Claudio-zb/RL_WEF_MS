from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
    
    def plot_memory(self):
        x = np.zeros(len(self.memory))
        y = np.zeros(len(self.memory))
        z = np.zeros(len(self.memory))
        for idx, item in enumerate(self.memory):
            if item[2] is not None:
                x[idx] = item[2][0, 0].cpu().detach().numpy() # actual value
                y[idx] = item[0][0, 0].cpu().detach().numpy() # prev value
                z[idx] = item[3][0, 0].cpu().detach().numpy() # reward

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, 'gray')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

