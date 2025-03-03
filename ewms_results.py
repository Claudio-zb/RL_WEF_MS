from environments.SimuEnv import SimuEnv
from environments.WMS_policy import LearnedIrrigationPolicy, TriggeredIrrigationPolicy
from environments.EMS_env import RuleBasedEMS
from stable_baselines3 import TD3
import pandas as pd

import matplotlib.pyplot as plt

#%%
irrigation_rl_model = TD3.load("logs/wms/td3/best_model.zip")
# irrigation_policy = LearnedIrrigationPolicy(1, irrigation_rl_model)
irrigation_policy = TriggeredIrrigationPolicy(1, 1, 1)

ems_rl_policy = TD3.load("logs/ems/td3/best_model.zip")
ems_policy = RuleBasedEMS(1, ems_rl_policy, isNormalized=True)

simulation_env = SimuEnv(irrigation_policy=irrigation_policy,
                         ems_policy=ems_policy)

#%%
#simulation_env.start(1)
simulation_env.run(init_doy=295, total_days=10)
data = simulation_env.get_simu_data()

#%%
soil_data = pd.DataFrame(simulation_env.soil_data[0])

#%%

crop_obs = [item["tomato"] for item in data["cultivate_obs"]]
mg_obs = [item for item in data["mg_obs"]]
crop_actions = [item[0] for item in data["wms_actions"]]
mg_actions = [item[0] for item in data["ems_actions"]]

#%%
soil_data = pd.DataFrame(simulation_env.soil_data[0])

#%%
plt.plot(crop_actions)
plt.plot(mg_actions)
plt.show()

print("a")
