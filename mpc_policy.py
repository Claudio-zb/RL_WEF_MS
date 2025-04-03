#%%
import numpy as np
import pandas as pd
from typing import Dict
from abc import ABC, abstractmethod
from environments.WMS_policy import IrrigationPolicy
from environments.WMS_env import Cultivates
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
import copy
#%%
class MPCPolicy():
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.crops = 1
    
    def get_action(self, obs, model, disturbances):
        self.model = model
        # Set-up hyperparameters
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
        # Create bounds
        max_bound = 20. * np.ones(self.horizon)
        min_bound = np.zeros(self.horizon) 
        bounds = (min_bound, max_bound)

        # Call instance of PSO
        optimizer = ps.single.LocalBestPSO(n_particles=10*self.horizon, dimensions=self.horizon, options=options, bounds=bounds)

        cost_fun = lambda x: self.cost_function(obs, x, disturbances=disturbances)

        # Perform optimization
        cost, action = optimizer.optimize(cost_fun, iters=100, n_processes=None)
        return action[0]
    
    def cost_function(self, obs: np.ndarray, actions: np.ndarray, disturbances: list[dict]) -> float:
        cost = np.zeros(actions.shape[0])
        for particle, action in enumerate(actions):
            pso_model = copy.deepcopy(self.model)
            for crop in self.model.crops:
                for idx in range(self.horizon):
                    obs_dict = pso_model.step([action[idx]], disturbances[idx])
                    obs_array = obs_dict["potato"]
                    Ks = obs_array[-1]
                    cost[particle] += -Ks + action[idx]
        return cost

def obs_dict_to_array(obs: dict) -> np.ndarray:
    """
    Converts a dictionary of observations to a numpy array.
    :param obs: dictionary of observations
    :return: numpy array of observations
    """
    obs_array = np.zeros((len(obs), len(obs[0])))
    for i, crop in enumerate(obs.keys()):
        obs_array[i] = obs[crop]
    return obs_array

#%%

cultivate_env = Cultivates()
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
daily_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()

horizon = 5
mpc = MPCPolicy(horizon=horizon)
#%%
obs, info = cultivate_env.start(daily_weather_data)
simu_days = 10
prev_action = 0
actions = []
observations = []
for i in range(simu_days):
    future_weather_data = []
    for j in range(horizon):
        future_weather_data.append(weather_data.loc[weather_data["doy"] == cultivate_env.doy+j].iloc[0].to_dict())
    daily_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()
    action = mpc.get_action(obs, cultivate_env, future_weather_data)# irr_policy(obs)
    actions.append(action)
    obs = cultivate_env.step([action], daily_weather_data)
    observations.append(obs)
    prev_action = action

crop_data = cultivate_env.crops[0].hist_data
crop_data = np.array(crop_data)
soil_data = pd.DataFrame(cultivate_env.get_soil_data()[0])
observatios = np.array(observations)
# %%

plt.plot(actions)
# %%
observations = np.array([observation["potato"] for observation in observations])
plt.plot(observations[:, -1], label="Ks")

# %%
