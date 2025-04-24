#%%
from RL_algorithms.PPO2 import *
from environments.WMS_env import NormalizedWMS, CultivateEnv, TestWMS

import numpy as np

env = TestWMS(NormalizedWMS(CultivateEnv(), 1))

ppo_agent = ActorCritic(state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.shape[0],
                        has_continuous_action_space=True,
                        action_std_init=0.6)

ppo_agent.load_state_dict(torch.load(r"logs\wms\weights_0\ppo\CultivateEnv\ppo_CultivateEnv_best.pth"))

obs, info = env.reset()
done = False
actions = []
observations = []
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# %%
