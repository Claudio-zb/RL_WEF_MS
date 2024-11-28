from environments.SimuEnv import SimuEnv
from environments.WMS_env import IrrigationPolicy
from environments.EMS_env import RuleBasedEMS
from stable_baselines3 import SAC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

irrigation_rl_model = SAC.load("logs/wms/sac/best_model.zip")
irrigation_policy = IrrigationPolicy(1, irrigation_rl_model)

ems_rl_policy = SAC.load("logs/ems/sac/best_model.zip")
ems_policy = RuleBasedEMS(1, ems_rl_policy, isNormalized=True)
# necesito un simulation environment que pueda aceptar dos politicas, irrigacion y gestion
# las politicas deben de ser objetos que tomen una observacion en formato diccionario, la transformen en un array
# y regresen un array de acciones

simulation_env = SimuEnv(irrigation_policy=irrigation_policy,
                         ems_policy=ems_policy)


#%%

#simulation_env.start(1)
simulation_env.run(init_doy=295, total_days=60)
data = simulation_env.get_simu_data()

#%%

crop_obs = [item["tomato"] for item in data["cultivate_obs"]]
mg_obs = [item for item in data["mg_obs"]]
crop_actions = [item[0] for item in data["wms_actions"]]
mg_actions = [item[0] for item in data["ems_actions"]]


#%%
plt.plot(crop_actions)
plt.plot(mg_actions)
plt.show()
