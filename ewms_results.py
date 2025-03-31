#%%
from environments.SimuEnv import SimuEnv
from environments.WMS_policy import LearnedIrrigationPolicy, TriggeredIrrigationPolicy
from environments.EMS_env import RuleBasedEMS
from stable_baselines3 import SAC
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#%%
#irrigation_rl_model = TD3.load("logs/wms/td3/best_model.zip")
# irrigation_policy = LearnedIrrigationPolicy(1, irrigation_rl_model)
irrigation_policy = TriggeredIrrigationPolicy(1, 3, 3.0)

ems_rl_policy = SAC.load("logs/ems/sac/best_model.zip")
ems_policy = RuleBasedEMS(1, ems_rl_policy, isNormalized=True)

simulation_env = SimuEnv(irrigation_policy=irrigation_policy,
                         ems_policy=ems_policy)

#%%
#simulation_env.start(1)
simulation_env.run(init_doy=295, total_days=115)
data = simulation_env.get_simu_data()

#%%
soil_data = pd.DataFrame(simulation_env.soil_data[0])

#%%

crop_obs = [item["potato"] for item in data["cultivate_obs"]]
#mg_obs = np.array([mg_tuple2array(item) for item in data["mg_obs"]])
crop_actions = [item[0] for item in data["wms_actions"]]
#mg_actions = [item[0] for item in data["ems_actions"]]
q_ps = data["qp_actions"]
q_is = data["qi_actions"]

#%%
soil_data = pd.DataFrame(simulation_env.soil_data[0])
mg_obs = data["mg_obs"]
wms_actions_2 = data["wms_actions_2"]
#%%
t = np.arange(0, len(wms_actions_2))
plt.scatter(t, wms_actions_2[:,1])
plt.scatter(t, data["wms_actions"])

#%%
#plt.plot(crop_actions)
fig, axs = plt.subplots(2,1)
axs[0].plot(mg_obs[:3*144,0])
axs[1].plot(q_ps[:3*144])

plt.show()

# %%

fig, axs = plt.subplots(2,1)
axs[0].plot(mg_obs[:3*144,1])
axs[0].axhline(y = 1.0, color= 'r')
axs[1].plot(q_is[:3*144])

plt.show()

# %%

plt.plot(mg_obs[:3*144,2])

# %%
