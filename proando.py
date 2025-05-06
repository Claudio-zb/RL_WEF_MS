#%%
from environments.Cultivates import * 
from environments.WMS_policies import RBIrrigationPolicy, IrrigationPolicy, RLIrrigationPolicy, MPCIrrigationPolicy
from environments.WMS_env import reward_function2
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, SAC, TD3

import pandas as pd
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")

print("yeah whatever")
year = 2012

rb_irr_policy = RBIrrigationPolicy(n_crops=1, model=Cultivates(), year=year)
rl_irr_policy = RLIrrigationPolicy(1, rl_policy=TD3.load("logs/wms/weights_6/td3/best_model.zip"), days_ahead=1, year=year)
mpc_irr_policy = MPCIrrigationPolicy(n_crops=1, model=Cultivates(), year=year, horizon=7)

def evaluate_policy(policy:IrrigationPolicy, reward_function=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]: 
    cultivates = Cultivates()
    plantation_day = cultivates.crops[0].plantation_day
    season_duration = 114
    index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == plantation_day)].index.values[0])
    observations = []
    obs_dict, doy = cultivates.start(seed=0)
    observations.append(obs_dict["potato"])
    actions = []
    for day_since_plantation in range(season_duration):
        daily_weather_data = weather_data.iloc[index + day_since_plantation].to_dict()
        disturbances = np.array([daily_weather_data["precipitation"], daily_weather_data["ET_0"]])
        action = policy.get_action(obs_dict, disturbances, doy)
        obs_dict, doy = cultivates.step(action, daily_weather_data)
        
        actions.append(action)
        observations.append(obs_dict["potato"])

    observations = np.array(observations)
    actions = np.array(actions)

    if reward_function is not None:
        rewards = np.zeros_like(actions)
        for i in range(len(actions)):
            rewards[i] = reward_function(observations[i], actions[i], observations[i+1])
    else:
        rewards = None

    total_water = np.sum(actions)
    relative_yield = np.exp(np.mean(np.log(observations[:, 7] + 1e-10)))

    return observations, actions.flatten(), rewards, relative_yield, total_water
rwd_fun = lambda s, a, s_next: reward_function2(s, a, s_next, 1, np.array([1., 1., 1.]))
#%%
observations, actions, rewards, rl_relative_yield, rl_total_water = evaluate_policy(rl_irr_policy, rwd_fun)

print(f"Total water for RL with weights {0} is {rl_total_water} m3")
print(f"Relative yield for RL with weights {0} is {rl_relative_yield}")
print(f"The return is {sum(rewards)}")
#%%
observations, actions, rewards, rb_relative_yield, rb_total_water = evaluate_policy(rb_irr_policy, rwd_fun)

print(f"Total water for RB with weights {0} is {rb_total_water} m3")
print(f"Relative yield for RB with weights {0} is {rb_relative_yield}")
print(f"The return is {sum(rewards)}")
#%%
observations, actions, rewards, mpc_relative_yield, mpc_total_water = evaluate_policy(mpc_irr_policy)

print(f"Total water for MPC with weights {0} is {mpc_total_water} m3")
print(f"Relative yield for MPC with weights {0} is {mpc_relative_yield}")
print(f"The return is {sum(rewards)}")
# %%

#%% comparison of policies
#mpc_data = np.load("logs/wms/mpc/6/mpc_data.npz")
mpc_relative_yield, mpc_total_water = 0.9966193574308968, 0.6260920839268992
#rl_relative_yield, rl_total_water = 0.99476784, 658.038
fig, ax1 = plt.subplots(figsize=(8, 4))

# Bar plot for Relative Yield
yield_color = "red"
water_color = "blue"

bar_width = 0.4
x = np.arange(3)
ax1.bar(x - bar_width/2, [rl_relative_yield*100, mpc_relative_yield*100, rb_relative_yield*100], 
    width=bar_width, label="Relative Yield (\%)", color=yield_color )
for i, value in enumerate([rl_relative_yield*100, mpc_relative_yield*100, rb_relative_yield*100]):
    ax1.text(i - bar_width/2, value + 0.1, f"{value:.2f}\%", ha='center', va='bottom', fontsize=11)
ax1.set_ylabel("Relative Yield (\%)", color=yield_color)
ax1.set_ylim(95, 100.45)
ax1.tick_params(axis='y', labelcolor=yield_color)

# Bar plot for Water Usage
ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, [rl_total_water*1000, mpc_total_water*1000, rb_total_water*1000], 
    width=bar_width, label="Water Usage (m3)", color=water_color)
for i, value in enumerate([rl_total_water*1000, mpc_total_water*1000, rb_total_water*1000]):
    ax2.text(i + bar_width/2, value + 5, f"{value:.1f}", ha='center', va='bottom', fontsize=11)
ax2.set_ylabel("Water Usage (m3)", color=water_color)
ax2.set_ylim(500, 900)
ax2.tick_params(axis='y', labelcolor=water_color)

# X-axis labels and legend
plt.xticks(x, ["RL", "MPC", "RB"])
fig.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.show()


# %%
print(mpc_relative_yield, mpc_total_water)