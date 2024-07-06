from RL_algorithms.DQN import DDQN, DuelingDDQN
from RL_algorithms.TD3 import TD3, PrioritizedTD3
from environments.EMS_env import ContinousEMSEnv, DiscreteEMSEnv, NormalizedEnv
from environments.GH_env import GH_env
from environments.Quad_env import Quad_env
from RL_algorithms.Trainer import TrainerUI 
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
env = NormalizedEnv(ContinousEMSEnv())
algorithm = TD3(env)

trainer = TrainerUI(env, algorithm)
trainer.run()


