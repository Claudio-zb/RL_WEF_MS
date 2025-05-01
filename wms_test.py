#%%
from stable_baselines3 import PPO, SAC, TD3
from predictive_models.utils import NN_soil_mdl
from environments.Cultivates import *
from matplotlib import pyplot as plt
import pandas as pd
import os
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
cultivate_env = Cultivates()
from environments.WMS_policies import MPCIrrigationPolicy, RLIrrigationPolicy, ScheduledIrrigationPolicy, Predictor

set_of_weights = np.array([[.9, .9, 1.2],
                           [.95, .95, 1.1],
                           [1.0, 1.0, 1.0], 
                           [1.05, 1.05, 0.9], 
                           [1.1, 1.1, 0.8]])

set_of_weights2 = np.array([[0.90, 1.2, 0.90],
                            [0.8, 1.4, 0.8],
                            [0.7, 1.6, 0.7],
                            [0.6, 1.8, 0.6],])

set_of_weights = np.concatenate((set_of_weights, set_of_weights2), axis=0)


def evaluate_mpc(reward_weights:np.ndarray):
    for iteration, weights in enumerate(reward_weights):
        
        year = 2018 
        cultivates = Cultivates()
        doy = cultivates.crops[0].plantation_day
        season_duration = 114
        index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
        observations = []
        mpc_policy = MPCIrrigationPolicy(cultivates, n_crops=1, horizon=7, reward_weights=weights)
        obs_dict, _ = cultivates.start()
        observations.append(obs_dict["potato"])
        actions = []
        for day_since_plantation in range(season_duration):
            daily_weather_data = weather_data.iloc[index + day_since_plantation].to_dict()
            action = mpc_policy.get_action(obs_dict, [index + day_since_plantation])
            actions.append(action[0])
            obs_dict, _ = cultivates.step(action, daily_weather_data)
            observations.append(obs_dict["potato"])
        observations = np.array(observations)

        
        # save the geenrated data
        relative_yield = np.exp(np.mean(np.log(np.array(observations)[:, 7] + 1e-10)))
        water_usage = np.sum(actions)
        path = f"logs/wms/mpc/{iteration}/"
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez(f"{path}mpc_data.npz", observations=observations, actions=actions, relative_yield=relative_yield, water_usage=water_usage)
        print(f"Total water for MPC with weights {iteration} is {water_usage} m3")
        print(f"Relative yield for MPC with weights {iteration} is {relative_yield}")

if __name__ == "__main__":
    evaluate_mpc(set_of_weights)



        


            
        

