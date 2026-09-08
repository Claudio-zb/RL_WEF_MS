#This script compares the performance of different implementations of irrigation policies

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
reward_weights = [1.0, 1.0, 1.25]  # [yield, delta_yield, water_usage]
rl_policy = RLIrrigationPolicy(n_crops=1, rl_policy=SAC.load("logs/wms/weights_5/sac/best_model.zip"), year=year)
mpc_policy = MPCIrrigationPolicy(n_crops=1, model=deepcopy(potatos), horizon=7, reward_weights=reward_weights, year=year)
rb_policy = RBIrrigationPolicy(n_crops=1, model=deepcopy(potatos), year=year)

# %%
doy = potatos.crops[0].plantation_day
season_duration = 115-30
obs_data = []
actions_data = []
seed = 42
policy_names = ["RL-Based", "MPC-Based", "Rule-Based"]
policies: list[IrrigationPolicy] = [rl_policy, mpc_policy, rb_policy]
run = False
index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
if run:
    for idx, policy in enumerate(policies):
        index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
        observations = []
        potatos = Cultivates()
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
        if idx == 1:
            break

#%% define cost functions

def mpc_rew(s:np.ndarray, a:np.ndarray, s_next:np.ndarray,
            weights: np.ndarray = np.array([1.0, 1.0, 1.0])):

    lambda_1 = weights[1]
    lambda_2 = weights[2]

    Ky = s_next[10]
    Ks = s_next[7]
    yield_r = 1 - (1 - Ky) * (1 - Ks)
    delta_yield = np.min((0, Ks - s[7]))
    water_usage = np.sum(a)
    total_cost = yield_r**2 - lambda_1 * delta_yield**2 - lambda_2 * water_usage**2
    return total_cost
def rl_rew(s:np.ndarray, a:np.ndarray, s_next:np.ndarray,
           weights: np.ndarray = np.array([1.0, 1.0, 1.0])):

    lambda_1 = weights[1]
    lambda_2 = weights[2]

    Ky = s_next[10]
    Ks = s_next[7]
    yield_r = 1 - (1 - Ky) * (1 - Ks)
    delta_yield = np.clip(Ks - s[7], -np.inf, 0)
    water_usage = np.sum(a)
    total_cost =  yield_r + lambda_1 * delta_yield - lambda_2 * water_usage
    return total_cost

def mpc_rew2(s:np.ndarray, a:np.ndarray, s_next:np.ndarray,
            weights: np.ndarray = np.array([1.0, 1.0, 1.0])):

    lambda_1 = weights[1]
    lambda_2 = weights[2]

    Ky = s_next[:, 10]
    Ks = s_next[:, 7]
    yield_r = np.sum((1 - (1 - Ky) * (1 - Ks))**2)
    delta_yield = lambda_1 * np.sum(np.clip(Ks - s[:, 7], -np.inf, 0)**2)
    water_usage = lambda_2 * np.sum(a)
    total_cost = yield_r - delta_yield - water_usage
    return total_cost, (yield_r, delta_yield, water_usage)

def rl_rew2(s:np.ndarray, a:np.ndarray, s_next:np.ndarray,
           weights: np.ndarray = np.array([1.0, 1.0, 1.0])):

    lambda_1 = weights[1]
    lambda_2 = weights[2]

    Ky = s_next[:, 10]
    Ks = s_next[:, 7]
    yield_r = np.sum((1 - (1 - Ky) * (1 - Ks)))
    delta_yield = lambda_1 * np.sum(np.clip(Ks - s[:, 7], -np.inf, 0))
    water_usage = lambda_2 * np.sum(a)
    total_cost = yield_r + delta_yield - water_usage
    return total_cost, (yield_r, delta_yield, water_usage)

#%% Load the data
relative_yields = []
water_usages = []
mpc_rews = []
rl_rews = []

preps = weather_data.iloc[index:index+85]["precipitation"].values
tt = np.arange(0,85, 1)
fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (7,3.5))
colors = ["tab:orange", "tab:green", "tab:purple"]
#axs[0].step(tt, preps)
#axs[0].set_title("Precipitations")
for idx, name in enumerate(policy_names):
    observations = pickle.load(open(f"simu_results/wf_ms/{name}/observations.pkl", "rb"))
    actions = pickle.load(open(f"simu_results/wf_ms/{name}/actions.pkl", "rb"))
    soil_data = pickle.load(open(f"simu_results/wf_ms/{name}/soil_data.pkl", "rb"))
    ry = np.exp(np.mean(np.log(1 - observations[:, 10]*(1 - observations[:,7]))))
    relative_yields.append(ry)

    a, b = mpc_rew2(observations[:-1,:], actions, observations[1:,:], weights=reward_weights)
    print(a)
    print(b)
    water_usage = np.sum(actions)
    water_usages.append(water_usage*1000)
    print(f"Total water for {name} is {water_usage} m3")
    print(f"Relative yield for {name} is {ry}")

    mpc_cum_rew = a
    rl_cum_rew = rl_rew2(observations[:-1,:], actions, observations[1:,:], weights=reward_weights)[0]
    mpc_rews.append(mpc_cum_rew)
    rl_rews.append(rl_cum_rew)

    axs[0].step(tt, actions*1000, label = "Irrigation", color=colors[idx] )
    axs[0].step(tt, preps, label = "Precipitations")
    #axs[0].step(tt, preps+ actions*1000, color = "tab:green")
    if name.upper()[:-6] == "RULE":
        name = "RB      "
    axs[0].set_title("Infiltration events")
    axs[0].set_ylabel("Water amount (mm)", fontsize=12)
    axs[0].set_ylim(0,11)
    axs[0].legend()
    #axs[idx+1].legend()


    print(f"{policy_names[idx]} Total MPC Rewards :", mpc_cum_rew)
    print(f"{policy_names[idx]} Total RL Rewards :", rl_cum_rew)
    print(".................")
    break

axs[1].step(tt, observations[:-1,8]>=3, color = "black")
axs[1].set_title("Water stress indicator $t^{stress}$")
axs[1].set_ylabel("Activation", fontsize = 12)
fig.set_size_inches((6,4))
for ax in axs:
    ax.grid(which = "both")
fig.tight_layout()
fig.savefig("simu_results/figures/Irrigation_rl.png", dpi = 300)

#    fig, ax = plt.subplots(figsize=(8, 4))
#    ax.plot(soil_data["layer_0_0"])
#    ax.plot(soil_data["layer_1_0"])
#    ax.plot(soil_data["layer_2_0"])
#    ax.plot(soil_data["layer_3_0"])
#    ax.plot(soil_data["layer_4_0"])


relative_yields = np.array(relative_yields)
water_usages = np.array(water_usages)
mpc_rews = np.array(mpc_rews)
rl_rews = np.array(rl_rews)

#%%
crop_obs = observations
fig, axs = plt.subplots(observations.shape[1], 1, figsize = (10, 20))

for idx, ax in enumerate(axs):
    ax.plot(crop_obs[:, idx])

#%%

plt.plot(crop_obs[:, 8]>=3)


#%%


#%%
# Create a LaTeX-formatted table with policies as columns
table_data = {
    "Metric": ["MPC Reward", "RL Reward"],
    "RL-Based": [mpc_rews[0], rl_rews[0]],
    "MPC-Based": [mpc_rews[1], rl_rews[1]],
    "Rule-Based": [mpc_rews[2], rl_rews[2]],
}

latex_table = r"""\begin{table}[ht]
\centering
\begin{tabular}{lccc}
\hline
Metric & Rule-Based & RL-Based & MPC-Based \\
\hline
"""
for i, metric in enumerate(table_data["Metric"]):
    latex_table += f"{metric} & {table_data['Rule-Based'][i]:.2f} & {table_data['RL-Based'][i]:.2f} & {table_data['MPC-Based'][i]:.2f} \\\\\n"
latex_table += r"""\hline
\end{tabular}
\caption{Comparison of MPC and RL rewards for each policy.}
\label{tab:policy_rewards}
\end{table}
"""

with open("simu_results/wf_ms/wms_comparison_table.tex", "w") as f:
    f.write(latex_table)


#%%

fig, ax1 = plt.subplots(figsize=(7, 3.5))

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
ax1.set_xticklabels(policy_names)

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
#%%
# %%
fig, ax1 = plt.subplots(figsize=(5, 3))

# Bar width
bar_width = 0.35

# Indices for the bars (only MPC and RL)
indices = np.arange(2)
policy_names_filtered = policy_names[:2]
water_usages_filtered = water_usages[:2]
relative_yields_filtered = relative_yields[:2]

# Plot water usage
water_bars = ax1.bar(indices - bar_width/2, water_usages_filtered, bar_width, label='Water Usage', color='b', alpha=0.7)
ax1.set_ylabel('Water Usage (m³)', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticks(indices)
ax1.set_xticklabels(policy_names_filtered)

# Add values on top of water usage bars
for bar in water_bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}', ha='center', va='bottom', color='b')

ax1.set_ylim(400, max(water_usages_filtered) * 1.05)

# Create a second y-axis for relative yield
ax2 = ax1.twinx()
yield_bars = ax2.bar(indices + bar_width/2, [ry * 100 for ry in relative_yields_filtered], bar_width, label='Relative Yield', color='g', alpha=0.7)
ax2.set_ylabel('Relative Yield (\%)', color='g', fontsize=12)
ax2.tick_params(axis='y', labelcolor='g')

# Add values on top of relative yield bars
for bar in yield_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}%', ha='center', va='bottom', color='g')

ax2.set_ylim(80, max(relative_yields_filtered) * 100 * 1.05)

plt.tight_layout()
plt.savefig("simu_results/wms_comparison_mpc_rl.png", dpi=300)
plt.show()
