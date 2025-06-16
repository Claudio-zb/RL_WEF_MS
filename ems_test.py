#%%
import pandas as pd
from environments.EnergyWaterMG import EnergyWaterMG
from environments.EMS_policies import RBPumpingPolicy, MPCPumpingPolicy
from environments.utils.funcionesEMS import get_demand, get_rad, get_temperatura, solar_power
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
import matplotlib.pyplot as plt


#sac_agent = SAC.load("logs/ems/sac/best_model.zip")
rb_policy = RBPumpingPolicy(1)
mpc_policy = MPCPumpingPolicy(n_crops=1, model=EnergyWaterMG(), horizon=10)

ew_mg = EnergyWaterMG()

power_demanded = get_demand()
radiation = get_rad()
temperature = get_temperatura()
pv_power = solar_power(radiation, temperature)

#%%
obs = ew_mg.start()
observations = []
observations.append(obs)
actions = []
v_reqs = np.array([1.0])
for i in range(144*3):
    disturbances = pv_power[i], power_demanded[i]
    action = mpc_policy.get_action(v_reqs, obs)
    actions.append(action)
    obs = ew_mg.next_step(action, disturbances)
    observations.append(obs)
observations = np.array(observations)
actions = np.array(actions)
#%%

fig, axs = plt.subplots(2,1)

axs[0].plot(observations[:,0])
axs[1].plot(actions[:,0])

#%%
fig, axs = plt.subplots(2,1)

axs[0].plot(observations[:,1])
axs[1].plot(actions[:,1])

#%%
fig, ax = plt.subplots(1,1)
ax.plot(observations[:,4])
ax.plot(observations[:,3])

#%%
fig, axs = plt.subplots(2,1)
axs[0].plot(observations[:,5])
axs[1].plot(actions[:,0])   

# %%
plt.plot(power_demanded[0:144*3])