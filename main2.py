# This script is going to be used for testing of the different environments
import numpy as np
from environments.EMS_env import EnergyWaterMG, MicrogridEnv

mg_env = MicrogridEnv(n_crops=1)
#%%

policy = lambda x: np.array([0., 0])

obs, _ = mg_env.reset()
for i in range(50):
    action = policy(obs)
    obs, rew, done, _, _ = mg_env.step(action)
    print(obs)
    if done:
        break

