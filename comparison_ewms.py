#%%
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy, RuleBasedIrrigationPolicy, ScheduledIrrigationPolicy, MPCIrrigationPolicy
from environments.EMS_policies import RBPumpingPolicy, RLPumpingPolicy
from stable_baselines3 import SAC, TD3, PPO
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random

#%% Set the experiments  

irrigation_rl_model = RLIrrigationPolicy(1, TD3.load("logs/wms/weights_4/td3/best_model.zip"), year=2018)

ems_rl_model = SAC.load("experimental_logs/ems/sac/best_model.zip")
ems_rl = RLPumpingPolicy(1, ems_rl_model, isNormalized=True)
ems_rb = RBPumpingPolicy(1, v_tank_max=5)


simulation_env_rl_rl = SimuEnv(irrigation_policy=irrigation_rl_model,
                         ems_policy=ems_rl)

simulation_env_rl_rb = SimuEnv(irrigation_policy=irrigation_rl_model,
                         ems_policy=ems_rb)



#%% lets prepare the weatheer data
doy = simulation_env_rl_rl.cultivate_env.crops[0].plantation_day
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
year = 2018
index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])

#%% RL + RL case study
np.random.seed(42)
random.seed(42)
simulation_env_rl_rl.run(init_doy=295, total_days=115-30)
data_rl_rl = simulation_env_rl_rl.get_simu_data()
soil_data_rl_rl = pd.DataFrame(simulation_env_rl_rl.soil_data[0])

#%% RL + RB case study
np.random.seed(42)
random.seed(42)
simulation_env_rl_rb.run(init_doy=295, total_days=115-30)
data_rl_rb = simulation_env_rl_rb.get_simu_data()
soil_data_rl_rb = pd.DataFrame(simulation_env_rl_rb.soil_data[0])

#%%

crop_obs = [item["potato"] for item in data_rl_rl["cultivate_obs"]]

v_reqs = [item[0] for item in data_rl_rl["wms_actions"]]
#mg_actions = [item[0] for item in data_rl_rl["ems_actions"]]
q_ps_rl = data_rl_rl["qp_actions"]
q_is_rl = data_rl_rl["qi_actions"]

soil_data = pd.DataFrame(simulation_env_rl_rl.soil_data[0])
mg_obs_rl = np.array(data_rl_rl["mg_obs"])
mg_dis_rl = np.array(data_rl_rl["mg_dis"])
mg_obs_144_rl = data_rl_rl["end_of_day_samples"]

#%%

crop_obs_rb = [item["potato"] for item in data_rl_rb["cultivate_obs"]]
#mg_obs = np.array([mg_tuple2array(item) for item in data["mg_obs"]])
v_reqs_rb = [item[0] for item in data_rl_rb["wms_actions"]]
#mg_actions = [item[0] for item in data["ems_actions"]]
q_ps_rb = data_rl_rb["qp_actions"]
q_is_rb = data_rl_rb["qi_actions"]

soil_data_rb = pd.DataFrame(simulation_env_rl_rb.soil_data[0])
mg_obs_rb = np.array(data_rl_rb["mg_obs"])
mg_dis_rb = np.array(data_rl_rb["mg_dis"])
mg_obs_144_rb = data_rl_rb["end_of_day_samples"]

#%%
day_k =10
n_days = 1
plt.plot(q_ps_rl[day_k*144:day_k*144+n_days*144], label="q_p")
plt.plot(q_is_rl[day_k*144:day_k*144+n_days*144], label="q_i")
plt.legend()


#%%
plt.plot(mg_dis_rb[day_k*144:day_k*144+n_days*144, 0])
plt.plot(mg_dis_rl[day_k*144:day_k*144+n_days*144, 0])
#%%
plt.plot(mg_obs_rb[day_k*144:day_k*144+n_days*144,0])

#%%
plt.plot(mg_obs_rl[day_k*144:day_k*144+n_days*144,1], label="V_irrigation")
#%%
plt.plot(mg_obs_rl[day_k*144:day_k*144+n_days*144,4], label="SoE")

#%%

plt.plot(mg_obs_rb[day_k*144:day_k*144+n_days*144,2])

#%%
plt.plot(mg_obs_rl[day_k*144:day_k*144+n_days*144,5], label = "e_residual")
plt.plot(mg_obs_rl[day_k*144:day_k*144+n_days*144,3], label = "p_pumps")
plt.plot(mg_dis_rl[day_k*144:day_k*144+n_days*144,0]/10, label="p_pv")
print(np.sum(np.clip(mg_obs_rl[:,5], -np.inf, 0)), np.sum(np.clip(mg_obs_rb[:,5], -np.inf, 0)))


#%%
t = np.arange(0, len(mg_obs_144_rl))

plt.step(t[:-1], data_rl_rl["wms_actions"][:-1], label=r"$V_{req}$", where="post" )
plt.step(t[:-1], mg_obs_144_rl[:-1,1],label = r"$V_{irr}|_{\text{end of the day}}$", where="post")
plt.xlabel("Time since plantation (days)")
plt.ylabel("Water volume (m3)")
plt.grid()
plt.legend()

fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.tight_layout()

plt.savefig("WEFMS.png", dpi = 300)

#%%
t = np.arange(0, len(mg_obs_144_rb))

plt.step(t[:-1], data_rl_rb["wms_actions"][:-1], label=r"$V_{req}$", where="post" )
plt.step(t[:-1], mg_obs_144_rb[:-1,1],label = r"$V_{irr}|_{\text{end of the day}}$", where="post")
plt.xlabel("Time since plantation (days)")
plt.ylabel("Water volume (m3)")
plt.grid()
plt.legend()

fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.tight_layout()

plt.savefig("WEFMS.png", dpi = 300)
   #%%
#plt.plot(crop_actions)

simu_days = 3


fig, axs = plt.subplots(2,1)
axs[0].plot(mg_obs[:simu_days*144,0])
axs[1].plot(q_ps[:simu_days*144])

plt.show()
# %%
