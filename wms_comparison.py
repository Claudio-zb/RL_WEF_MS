# On this scriit the performance of different implementations of irrigation policies is compared

#%%
from environments.Cultivates import Cultivates
from environments.WMS_policies import MPCIrrigationPolicy, RLIrrigationPolicy, RBIrrigationPolicy, PPOIrrigationPolicy, IrrigationPolicy
import numpy as np
from copy import deepcopy
from stable_baselines3 import SAC, TD3, PPO
import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


#%%

potatos = Cultivates()
year = 2018
doy = potatos.crops[0].plantation_day
season_duration = 115 - 30
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")

mpc_policy = MPCIrrigationPolicy(n_crops=1, model=deepcopy(potatos), horizon=7, reward_weights=[1.0, 1.0, 1.25], year=year)
rl_policy = RLIrrigationPolicy(n_crops=1, rl_policy=SAC.load("logs/wms/weights_5/sac/best_model.zip"), year=year)
rb_policy = RBIrrigationPolicy(n_crops=1, model=deepcopy(potatos), year=year)

policies = [rb_policy, rl_policy, mpc_policy]


# %%
doy = potatos.crops[0].plantation_day
season_duration = 115-30

obs_data = []
actions_data = []
seed = 42
policy_names = ["Rule-Based", "RL-Based", "MPC-Based"]
policies: list[IrrigationPolicy] = [rb_policy, rl_policy, mpc_policy]
run = True
if run:
    for idx, policy in enumerate(policies):
        index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
        observations = []
        obs_dict, _ = potatos.start(seed=seed)
        observations.append(obs_dict["potato"])
        actions = []

        for day_since_plantation in range(season_duration):
            daily_weather_data = weather_data.iloc[index + day_since_plantation].to_dict()
            action = policy.get_action(obs_dict, [index + day_since_plantation], doy)
            actions.append(action[0])
            obs_dict, _ = potatos.step(action, daily_weather_data)
            observations.append(obs_dict["potato"])
        observations = np.array(observations)
        actions = np.array(actions)

        soil_data = potatos.get_soil_data()[0]

        save_path = f"simu_results/wf_ms/{policy_names[idx]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the generated data
        with open(save_path + "/observations.pkl", "wb") as f:
            pickle.dump(observations, f)
        with open(save_path + "/actions.pkl", "wb") as f:
            pickle.dump(actions, f)
        with open(save_path + "/soil_data.pkl", "wb") as f:
            pickle.dump(soil_data, f)



#%% Load the data
relative_yields = []
water_usages = []

policies_name = ["RL-Based", "Rule-Based","MPC-Based"] 
for idx, name in enumerate(policies_name):
    observations = pickle.load(open(f"simu_results/wf_ms/{name}/observations.pkl", "rb"))
    actions = pickle.load(open(f"simu_results/wf_ms/{name}/actions.pkl", "rb"))
    ry = np.exp(np.mean(np.log(1 - observations[:, 10]*(1 - observations[:,7]))))
    relative_yields.append(ry)

    water_usage = np.sum(actions)
    water_usages.append(water_usage*1000) 
    print(f"Total water for {policy.__class__.__name__} is {water_usage} m3")
    print(f"Relative yield for {policy.__class__.__name__} is {ry}")        

relative_yields = np.array(relative_yields)
water_usages = np.array(water_usages)


#%%
fig, ax1 = plt.subplots(figsize=(8, 3.2))

# Bar width
bar_width = 0.35

# Indices for the bars
indices = np.arange(len(policies))

# Plot water usage
water_bars = ax1.bar(indices - bar_width/2, water_usages, bar_width, label='Water Usage', color='b', alpha=0.7)
#ax1.set_xlabel('Policies')
ax1.set_ylabel('Water Usage (m³)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticks(indices)
ax1.set_xticklabels(policies_name)

# Add values on top of water usage bars
for bar in water_bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}', ha='center', va='bottom', color='b')

ax1.set_ylim(400, max(water_usages) * 1.05)  # Set y-limit for water usage

# Create a second y-axis for relative yield
ax2 = ax1.twinx()
yield_bars = ax2.bar(indices + bar_width/2, [ry * 100 for ry in relative_yields], bar_width, label='Relative Yield', color='g', alpha=0.7)
ax2.set_ylabel('Relative Yield (\%)', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Add values on top of relative yield bars
for bar in yield_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, height, fr'{height:.1f}\%' , ha='center', va='bottom', color='g')

ax2.set_ylim(80, max(relative_yields) * 100 * 1.05)  # Set y-limit for relative yield
# Show plot
plt.tight_layout()
plt.savefig("simu_results/wms_comparison.png", dpi=300)
plt.show()


# %%
