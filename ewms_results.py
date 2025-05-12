#%%
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy, RuleBasedIrrigationPolicy, ScheduledIrrigationPolicy, MPCIrrigationPolicy
from environments.EMS_policies import RBPumpingPolicy, RLPumpingPolicy
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
irrigation_rl_model = RLIrrigationPolicy(1, TD3.load("logs/wms/weights_4/td3/best_model.zip"), year=2018)

ems_rl_policy = SAC.load("experimental_logs/ems/sac/best_model.zip")
ems_policy = RLPumpingPolicy(1, ems_rl_policy, isNormalized=True)

simulation_env = SimuEnv(irrigation_policy=irrigation_rl_model,
                         ems_policy=ems_policy)

#%% lets prepare the weatheer data
doy = simulation_env.cultivate_env.crops[0].plantation_day
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
year = 2018
index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])

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
mg_obs = np.array(data["mg_obs"])
mg_obs_144 = data["end_of_day_samples"]
#%%
t = np.arange(0, len(mg_obs_144))



plt.step(t[:-1], data["wms_actions"][:-1], label = r"$V_{irr}|_{\text{end of the day}}$", where="post" )
plt.step(t[:-1], mg_obs_144[:-1,1], label=r"$V_{req}$", where="post")
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

t = np.arange(0, simu_days*24, 24/144)
fig, axs = plt.subplots(2,1)
fig.set_size_inches(8, 5)
c = ["tab:orange"]*3 #["r", "g", "b"]
for idx in range(simu_days):
    if idx == 0:
        axs[0].hlines(v_reqs[idx], 24*idx, 24*(idx+1), color= c[idx], label=r"$V_{req}$")
    else:
        axs[0].hlines(v_reqs[idx], 24*idx, 24*(idx+1), color= c[idx])
axs[0].plot(t, mg_obs[:simu_days*144,1], color="tab:blue", label=r"$V_{irr}$")
axs[0].set_ylabel(r"Water Volume (m3)")
axs[0].legend()

axs[0].grid()


axs[1].plot(t, q_is[:simu_days*144])
axs[1].grid()
axs[1].set_ylabel(r"Flow rate (l/s)")
axs[1].set_xlabel("Time since plantation (hours)")

plt.tight_layout()
plt.savefig("WEFMS_2.png", dpi = 300)
plt.show()

# %%
e_purchased = 0
e_sold = 0
for i in range(113):
    e_res = mg_obs[(i+1)*144,4]
    e_purchased += min(e_res, 0)
    e_sold += max(e_res, 0)
print(f"Energy purchased: {e_purchased} kWh")
print(f"Energy sold: {e_sold} kWh")

# %%
errors = []
for i in range(113):
    a = mg_obs_144[i,1]
    b = v_reqs[i]
    errors.append((a - b))
    
errors = np.array(errors)
plt.plot(errors)

print(f"Mean error: {np.mean(abs(errors))}")
print(f"Cumulative error: {np.sum(errors)}")

#%%
e_purchased = 0
e_sold = 0
for i in range(len(mg_obs)):

    e_res = mg_obs[i,4]
    e_purchased += min(e_res, 0)
    e_sold += max(e_res, 0)
print(f"Energy purchased: {e_purchased} kWh")
print(f"Energy sold: {e_sold} kWh")



# %%
