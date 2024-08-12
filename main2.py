import numpy as np
from matplotlib import pyplot as plt
from environments.WMS_env import WMS_env

env = WMS_env()
n_steps = 50
states = np.zeros((n_steps+1, 8))

obs, _ = env.reset()

for i in range(n_steps):
    obs, info = env.step(0)
    states[i+1] = obs

#%%

plt.plot(states[:,2], label="Depletion")
plt.plot(states[:,4], label="TAW")
plt.show()

#%%
plt.plot(states[:,5]/states[:,6], label="%depletion")
plt.show()
#%%
plt.plot(states[:,7], label="Ks")
plt.show()



