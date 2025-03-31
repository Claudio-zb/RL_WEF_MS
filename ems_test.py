#%%
import pandas as pd
from environments.EMS_env import EnergyWaterMG, RuleBasedEMS
from environments.utils.funcionesEMS import get_demand, get_rad, get_temperatura, solar_power
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
import matplotlib.pyplot as plt



sac_agent = SAC.load("logs/ems/sac/best_model.zip")
sac_policy = RuleBasedEMS(1, sac_agent, isNormalized=True)

ew_mg = EnergyWaterMG()

def mg_tuple2array(tup, n_crops = 1):
    return np.array([tup[0][0], tup[1][0], tup[2][0], tup[3], tup[4]])

power_demanded = get_demand()
radiation = get_rad()
temperature = get_temperatura()
pv_power = solar_power(radiation, temperature)
#%%
disturbances = np.array([power_demanded[0], pv_power[0]])
obs = ew_mg.start(50)
observations = [mg_tuple2array(obs)]
actions = []
v_reqs = [1.0]
for i in range(144*3):
    disturbances = pv_power[i], power_demanded[i]
    p_bat, pumps = sac_policy.get_action(obs, v_reqs, disturbances)
    actions.append(np.array(pumps[0]))
    obs = ew_mg.next_step((p_bat, pumps))
    observations.append(mg_tuple2array(obs))
observations = np.array(observations)
actions = np.array(actions)
#%%

fig, axs = plt.subplots(2,1)

axs[0].plot(observations[:,1])
axs[1].plot(actions[:,1])

#%%
fig, axs = plt.subplots(2,1)

axs[0].plot(observations[:,0])
axs[1].plot(actions[:,0])




