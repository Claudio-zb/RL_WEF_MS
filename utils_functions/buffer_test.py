import unittest
from ReplayMemory import ReplayMemory
import torch

class TestReplayMemory(unittest.TestCase):

    def init_size_test(self):
        obs_dim = 4
        action_dim = 2
        size = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rb = ReplayMemory(size, obs_dim, action_dim, device)
        self.assertEqual(rb.size(), 0)

    def foo_test(self):
        obs_dim = 4
        action_dim = 2
        size = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rb = ReplayMemory(size, obs_dim, action_dim, device)
        for i in range(size):
            s = torch.rand(obs_dim).to(device)
            a = torch.rand(action_dim).to(device)
            s_next = torch.rand(obs_dim).to(device)
            r = torch.rand(1).to(device)
            done = torch.rand(1).to(device)
            rb.push(s,a,s_next,r,done)

        self.assertEqual(rb.size(), size)

if __name__ == '__main__':
    unittest.main()