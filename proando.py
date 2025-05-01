#%%
from environments.Cultivates import * 
from environments.WMS_policies import RBIrrigationPolicy, IrrigationPolicy, RLIrrigationPolicy, MPCIrrigationPolicy
from environments.WMS_env import reward_function2
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, SAC, TD3

import pandas as pd
import numpy as np


weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")

print("yeah whatever")
year = 2012

rb_irr_policy = RBIrrigationPolicy(n_crops=1, model=Cultivates(), year=year)
rl_irr_policy = RLIrrigationPolicy(1, rl_policy=TD3.load("logs/wms/weights_0/td3/best_model.zip"), days_ahead=1, year=year)
mpc_irr_policy = MPCIrrigationPolicy(n_crops=1, model=Cultivates(), year=year, horizon=7)

def evaluate_policy(policy:IrrigationPolicy, reward_function=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]: 
    cultivates = Cultivates()
    plantation_day = cultivates.crops[0].plantation_day
    season_duration = 114
    index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == plantation_day)].index.values[0])
    observations = []
    obs_dict, doy = cultivates.start()
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
observations, actions, rewards, relative_yield, total_water = evaluate_policy(rl_irr_policy, rwd_fun)

print(f"Total water for RL with weights {0} is {total_water} m3")
print(f"Relative yield for RL with weights {0} is {relative_yield}")
print(f"The return is {sum(rewards)}")
#%%
observations, actions, rewards, relative_yield, total_water = evaluate_policy(rb_irr_policy, rwd_fun)

print(f"Total water for RB with weights {0} is {total_water} m3")
print(f"Relative yield for RB with weights {0} is {relative_yield}")
print(f"The return is {sum(rewards)}")
#%%
observations, actions, rewards, relative_yield, total_water = evaluate_policy(mpc_irr_policy)

print(f"Total water for MPC with weights {0} is {total_water} m3")
print(f"Relative yield for MPC with weights {0} is {relative_yield}")
print(f"The return is {sum(rewards)}")
# %%
