from RL_algorithms.DQN import DDQN, DuelingDDQN
from RL_algorithms.TD3 import TD3
from environments.EMS_env import ContinousEMSEnv, DiscreteEMSEnv
from environments.GH_env import GH_env
from RL_algorithms.Trainer import TrainerUI 
import gymnasium as gym

env = ContinousEMSEnv()
from RL_algorithms.TD3 import TD3
rl_alg = TD3(env)
N = 1000

env = DiscreteEMSEnv()
rl_alg = DuelingDDQN(env)
#trainer = TrainerUI(env, rl_alg)
#trainer.run()

N = 1000
for i in range(N):
    stats = rl_alg.one_ep_training(i)
    print(f"Episode {i}: {stats[0], stats[1]}")