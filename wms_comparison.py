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
run = True
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
    
    print(f"{policy_names[idx]} Total MPC Rewards :", mpc_cum_rew)
    print(f"{policy_names[idx]} Total RL Rewards :", rl_cum_rew)
    print(".................")


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
