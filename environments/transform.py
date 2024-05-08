import numpy as np

class LSTM_transform:
    """Class that transforms the state of the environments to makes
    possible the use of LSTM networks for the Q functions."""
    def __init__(self, length = 6):
        self.buffer = []

    def add(self, state):
        self.buffer.append(state)
        if len(self.buffer) > 5:
            self.buffer.pop(0)

    def get(self):
        return np.array(self.buffer)
