#%%
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy, MPCIrrigationPolicy
from environments.EMS_policies import RLPumpingPolicy, MPCPumpingPolicy
from stable_baselines3 import SAC, TD3
from environments.Cultivates import Cultivates
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from environments.utils.predict_utils import Forecaster, Predictor, load_model
import copy
import os
import pickle
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

#%% Top level controllers

wms_rl_model = SAC.load("logs/wms/weights_5/sac/best_model")
irrigation_rl_policy = RLIrrigationPolicy(n_crops=1, rl_policy=wms_rl_model, isNormalized=True, year=2018)
irrigation_mpc_policy = MPCIrrigationPolicy(1, Cultivates(), year=2018)

# Bottom level controllers
ems_rl_model = TD3.load("logs/ems/weights_6/td3/best_model")
pv_model = Forecaster(load_model("predictive_models/pv_model.pt"))
pd_model = Forecaster(load_model("predictive_models/pd_model.pt"))

ems_mpc = MPCPumpingPolicy(pv_model=pv_model, pd_model=pd_model, days_ahead=0)
ems_rl = RLPumpingPolicy(1, ems_rl_model, isNormalized=True)

#%% Setting up the simulation environments
random_seed = 42


simu_mpc_mpc = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                       ems_policy=ems_mpc, seed=random_seed)


#setting up the mpc

pv_data = np.concatenate((simu_mpc_mpc.prev_pv_daily_profile, simu_mpc_mpc.pv_daily_profile))
pd_data = simu_mpc_mpc.pd_daily_profile

ems_mpc.init_buffer(pv_data, pd_data)
ems_mpc.init_buffer(pv_data, pd_data)

simu_mpc_mpc.ems_policy = copy.deepcopy(ems_mpc)

#%% lets prepare the weather data
doy = simu_mpc_mpc.cultivate_env.crops[0].plantation_day
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
year = 2018
index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
path = "./simu_results/wef_ms/"
#%% running the cases
run = False
save_data = True
if run:
    print(f"Running mpc_mpc case study...")
    start_time = time.time()
    np.random.seed(random_seed), random.seed(random_seed)
    simu_mpc_mpc.run(init_doy=295, total_days=2)#115-30) #115-30)

    end_time = time.time()
    print(f"Finished mpc_mpc case study in {end_time - start_time:.2f} seconds.")

    mg_data, crop_data, soil_data = simu_mpc_mpc.get_simu_data()
    simu_path = path + "mpc_mpc"

    if save_data:

        if not os.path.exists(simu_path):
            os.makedirs(simu_path)

        with open(simu_path + "/mg_data.pkl", "wb") as f:
            pickle.dump(mg_data, f)

        with open(simu_path + "/crop_data.pkl", "wb") as f:
            pickle.dump(crop_data, f)

        with open(simu_path + "/soil_data.pkl", "wb") as f:
            pickle.dump(soil_data, f)
#%% Analyse the trajectory

mg_obs = mg_data["mg_obs"]
plt.plot(mg_obs[:,1])
# %%
v_reqs = crop_data["v_reqs"]
costs = (1 - (mg_obs[:,1]-v_reqs[0])**2/v_reqs[0]**2)
plt.plot(costs[0:144])
plt.ylim(-10,10)
# %%
