#%%
from environments.Cultivates import * 
from environments.WMS_policies import RBIrrigationPolicy, IrrigationPolicy
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")

print("yeah whatever")
year = 2018

irrigation_policy = RBIrrigationPolicy(n_crops=1, model=Cultivates(), year=year)

def evaluate_policy(policy:IrrigationPolicy) -> tuple[np.ndarray, np.ndarray, float, float]: 
    cultivates = Cultivates()
    doy = cultivates.crops[0].plantation_day
    season_duration = 114
    index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
    observations = []
    obs_dict, _ = cultivates.start()
    observations.append(obs_dict["potato"])
    actions = []
    for day_since_plantation in range(season_duration):
        daily_weather_data = weather_data.iloc[index + day_since_plantation].to_dict()
        disturbances = np.array([daily_weather_data["precipitation"], daily_weather_data["ET_0"]])
        action = policy.get_action(obs_dict, disturbances, doy)
        actions.append(action[0])
        obs_dict, _ = cultivates.step(action, daily_weather_data)
        observations.append(obs_dict["potato"])
    observations = np.array(observations)

    observations = np.array(observations)
    actions = np.array(actions)

    total_water = np.sum(actions)
    relative_yield = np.exp(np.mean(np.log(observations[:, 7] + 1e-10)))

    return observations, actions, relative_yield, total_water

    
observations, actions, relative_yield, total_water = evaluate_policy(irrigation_policy)