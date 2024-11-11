# This script is going to be used for testing of the different environments
import numpy as np
from environments.EMS_env import EnergyWaterMG, MicrogridEnv

mg_env = MicrogridEnv(n_crops=1)
#%%

policy = lambda x: np.array([0.1, 0.1])

x = []
obs, _ = mg_env.reset()
x.append(obs)
for i in range(50):
    action = policy(obs)
    obs, rew, done, _, _ = mg_env.step(action)
    x.append(obs)
    print(obs)
    if done:
        break

#%%
x = np.array(x)
import matplotlib.pyplot as plt
plt.plot(x[:, 0], label='v_ref')
plt.plot(x[:, 1], label='v')
plt.legend()
plt.show()








