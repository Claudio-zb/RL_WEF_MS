#%%
from environments.Cultivates import Cultivates
from environments.WMS_policies import MPCIrrigationPolicy, RLIrrigationPolicy, RBIrrigationPolicy, PPOIrrigationPolicy
import numpy as np
from copy import deepcopy
from stable_baselines3 import SAC, TD3, PPO
import pandas as pd
from matplotlib import pyplot as plt
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

relative_yields = []
water_usages = []

obs_data = []
actions_data = []
seed = 42

results_dict = {"sac":[], "td3":[]}
for alg, algorithm in zip(["sac", "td3"], [SAC, TD3]):
    policies = [RLIrrigationPolicy(n_crops=1, rl_policy=algorithm.load(f"logs/wms/weights_{i}/{alg}/best_model.zip"), year=year) for i in range(8)]
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
        obs_data.append(observations)
        
        ry = np.exp(np.mean(np.log(1 - observations[:, 10]*(1 - observations[:,7]))))
        relative_yields.append(ry)
        water_usage = np.sum(actions)
        water_usages.append(water_usage) 
        print(f"Total water for {alg} with {idx} is {water_usage} m3")
        print(f"Relative yield for {alg} with {idx} is {ry}")
    results_dict[alg].append(relative_yields)


    


# %%

plt.plot(obs_data[0][:, 7], label="RB Policy")
plt.plot(obs_data[1][:, 7], label="RL Policy")
# %%

# %%
doy = potatos.crops[0].plantation_day
season_duration = 115-30

relative_yields = []
water_usages = []

obs_data = []
actions_data = []
seed = 42



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
    obs_data.append(observations)
    actions_data.append(actions)
    
    ry = np.exp(np.mean(np.log(1 - observations[:, 10]*(1 - observations[:,7]))))
    relative_yields.append(ry)
    water_usage = np.sum(actions)
    water_usages.append(water_usage) 
    print(f"Total water for {policy.__class__.__name__} is {water_usage} m3")
    print(f"Relative yield for {policy.__class__.__name__} is {ry}")
    
    water_usages = np.array(water_usages)*1000

# %%

plt.plot(obs_data[0][:, 10], label="RB Policy")
plt.plot(obs_data[1][:, 10], label="RL Policy")

#%%
plt.plot(actions_data[0], label="RB Policy")
plt.plot(actions_data[1], label="RL Policy")
plt.plot(actions_data[2], label="MPC Policy")
# %%


#%%
fig, ax1 = plt.subplots(figsize=(8, 3.2))
policies_name = ["Rule-Based", "RL-Based", "MPC-Based"]

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

# Create a second y-axis for relative yield
ax2 = ax1.twinx()
yield_bars = ax2.bar(indices + bar_width/2, [ry * 100 for ry in relative_yields], bar_width, label='Relative Yield', color='g', alpha=0.7)
ax2.set_ylabel('Relative Yield (\%)', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Add values on top of relative yield bars
for bar in yield_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}%', ha='center', va='bottom', color='g')

# Add legend
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.4))

# Add title
#plt.title('Comparison of Water Usage and Relative Yield Across Policies')

# Show plot
plt.tight_layout()
plt.savefig("wms_comparison.png", dpi=300)
plt.show()


# %%
