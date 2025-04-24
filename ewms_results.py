#%%
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy, RuleBasedIrrigationPolicy, ScheduledIrrigationPolicy, MPCIrrigationPolicy
from environments.EMS_env import RuleBasedEMS
from stable_baselines3 import SAC, TD3, PPO
import pandas as pd
from stable_baselines3 import PPO
import numpy as np
import time
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

#%%
irrigation_rl_model = RLIrrigationPolicy(1,PPO.load("logs/wms/sac/best_model.zip"))

ems_rl_policy = TD3.load("experimental_logs/ems/td3/best_model.zip")
ems_policy = RuleBasedEMS(1, ems_rl_policy, isNormalized=True)

simulation_env = SimuEnv(irrigation_policy=irrigation_rl_model,
                         ems_policy=ems_policy)

#%%
#simulation_env.start(1)
t0 = time.time()
simulation_env.run(init_doy=295, total_days=115)

print(f"execution done in: {time.time() - t0} seconds")

data = simulation_env.get_simu_data()
soil_data = pd.DataFrame(simulation_env.soil_data[0])

#%%

crop_obs = [item["potato"] for item in data["cultivate_obs"]]
#mg_obs = np.array([mg_tuple2array(item) for item in data["mg_obs"]])
v_reqs = [item[0] for item in data["wms_actions"]]
#mg_actions = [item[0] for item in data["ems_actions"]]
q_ps = data["qp_actions"]
q_is = data["qi_actions"]

soil_data = pd.DataFrame(simulation_env.soil_data[0])
mg_obs = data["mg_obs"]
wms_actions_2 = data["end_of_day_samples"]
#%%
t = np.arange(0, len(wms_actions_2))


plt.step(t, wms_actions_2[:,1], label=r"$V_{req}$", where="post")
plt.step(t, data["wms_actions"], label = r"$V_{irr}|_{\text{end of the day}}$", where="post" )
plt.xlabel("Time since plantation [days]")
plt.ylabel("Water amount[m3]")
plt.grid()
plt.legend()


plt.title("Joint action of two agents")



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

fig, axs = plt.subplots(2,1)
c = ["r", "g", "b"]
for idx in range(simu_days):
    axs[0].hlines(v_reqs[idx], 144*idx, 144*(idx+1), color= c[idx])
axs[0].plot(mg_obs[:simu_days*144,1])

axs[1].plot(q_is[:simu_days*144])

plt.show()

# %%

plt.plot(mg_obs[:simu_days*144,-2])

# %%
