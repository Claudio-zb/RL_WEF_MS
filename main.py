from RL_algorithms.DQN import DDQN, DuelingDDQN
from RL_algorithms.TD3 import TD3
from environments.EMS_env import ContinousEMSEnv, DiscreteEMSEnv
from environments.GH_env import GH_env
from environments.Quad_env import Quad_env
from RL_algorithms.Trainer import TrainerUI 
import gymnasium as gym

env = ContinousEMSEnv()
algorith = TD3(env)

trainer = TrainerUI(env, algorith)
trainer.run()