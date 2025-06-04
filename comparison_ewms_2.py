#%%
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy, RuleBasedIrrigationPolicy, ScheduledIrrigationPolicy, MPCIrrigationPolicy
from environments.EMS_policies import RBPumpingPolicy, RLPumpingPolicy, MPCPumpingPolicy
from stable_baselines3 import SAC, TD3, PPO
from environments.Cultivates import Cultivates
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from environments.utils.predict_utils import Forecaster, Predictor, load_model
import copy
import os

#%% Top level controllers  

irrigation_mpc_policy = MPCIrrigationPolicy(1, Cultivates(), year=2018)

wms_rl_model = TD3.load("logs/wms/weights_4/td3/best_model.zip")
irrigation_rl_policy = RLIrrigationPolicy(n_crops=1, rl_policy=wms_rl_model, isNormalized=True, year=2018)

# Bottom level controllers
ems_rl_model = SAC.load("experimental_logs/ems/sac/best_model.zip")
pv_model = Forecaster(load_model("predictive_models/pv_model.pt"))
pd_model = Forecaster(load_model("predictive_models/pd_model.pt"))

ems_mpc_1_day = MPCPumpingPolicy(pv_model=pv_model, pd_model=pd_model, days_ahead=0)
ems_mpc_2_day = MPCPumpingPolicy(pv_model=pv_model, pd_model=pd_model, days_ahead=1)
ems_rl = RLPumpingPolicy(1, ems_rl_model, isNormalized=True)
ems_rb = RBPumpingPolicy(1, v_tank_max=5)

#%% Setting up the simulation environments
random_seed = 42

np.random.seed(random_seed), random.seed(random_seed)
simu_mpc_rl = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                      ems_policy=copy.deepcopy(ems_rl))

np.random.seed(random_seed), random.seed(random_seed)
simu_mpc_rb = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                      ems_policy=copy.deepcopy(ems_rb))

np.random.seed(random_seed), random.seed(random_seed)
simu_mpc_mpc = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                       ems_policy=ems_mpc_2_day)

np.random.seed(random_seed), random.seed(random_seed)
simu_rl_mpc = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_rl_policy),
                        ems_policy=copy.deepcopy(ems_mpc_1_day))

np.random.seed(random_seed), random.seed(random_seed)
simu_rl_rb = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_rl_policy),
                      ems_policy=copy.deepcopy(ems_rb))

np.random.seed(random_seed), random.seed(random_seed)
simu_rl_rl = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_rl_policy),
                      ems_policy=copy.deepcopy(ems_rl))

#setting up the mpc

pv_data = np.concatenate((simu_mpc_mpc.prev_pv_daily_profile, simu_mpc_mpc.pv_daily_profile))
pd_data = simu_mpc_mpc.pd_daily_profile

ems_mpc_1_day.init_buffer(pv_data, pd_data)
ems_mpc_2_day.init_buffer(pv_data, pd_data)

simu_mpc_mpc.ems_policy = copy.deepcopy(ems_mpc_2_day)
simu_rl_mpc.ems_policy = copy.deepcopy(ems_mpc_1_day) 

simu_cases = [simu_mpc_rl, simu_mpc_rb, simu_mpc_mpc, simu_rl_rl, simu_rl_rb, simu_rl_mpc]
simu_names = ["mpc_rl", "mpc_rb", "mpc_mpc", "rl_rl", "rl_rb", "rl_mpc"]



#%% lets prepare the weather data
doy = simu_mpc_rl.cultivate_env.crops[0].plantation_day
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
year = 2018
index = int(weather_data.loc[(weather_data["year"] == year) & (weather_data["doy"] == doy)].index.values[0])
path = "./simu_results/wef_ms/"
#%% running the cases

for simu, name in zip(simu_cases, simu_names):
    print(f"Running {name} case study...")
    start_time = time.time()    
    np.random.seed(random_seed), random.seed(random_seed)
    simu.run(init_doy=295, total_days=115-30)

    end_time = time.time()
    print(f"Finished {name} case study in {end_time - start_time:.2f} seconds.")
    
    data = simu.get_simu_data()
    soil_data = pd.DataFrame(simu.soil_data[0])

    # Crop response to water data

    # Energy-water microgrid data

    crop_obs = np.array([item["potato"] for item in data["cultivate_obs"]])
    v_reqs = data["wms_actions"]

    q_ps = data["qp_actions"]
    q_is = data["qi_actions"]

    mg_obs = np.array(data["mg_obs"])
    mg_dis = np.array(data["mg_dis"])
    mg_obs_144 = np.array(data["end_of_day_samples"])

    # Save the data in the respective folder 

    simu_path = path + name

    if not os.path.exists(simu_path):
        os.makedirs(simu_path)
        np.save(simu_path + "/v_reqs.npy", v_reqs)
        np.save(simu_path + "/q_ps.npy", q_ps)
        np.save(simu_path + "/q_is.npy", q_is)
        np.save(simu_path + "/mg_obs.npy", mg_obs)
        np.save(simu_path + "/mg_dis.npy", mg_dis)
        np.save(simu_path + "/mg_obs_144.npy", mg_obs_144)
        soil_data.to_csv(simu_path + "/soil_data.csv", index=False)
        np.save(simu_path + "/crop_obs", crop_obs)

#%% compute the metrics

energy_purchased = []
ref_tracking_error = []
relative_yields = []
water_usages = []

for name in simu_names:
    simu_path = path + name
    mg_obs = np.load(simu_path + "/mg_obs.npy")
    v_reqs = np.load(simu_path + "/v_reqs.npy")
    mg_obs_144 = np.load(simu_path + "/mg_obs_144.npy")
    energy_purchased.append(np.sum(np.clip(mg_obs[:,5], -np.inf, 0)))

    errors = v_reqs[:-1] - mg_obs_144[:-1,1]
    ref_tracking_error.append(np.mean(np.abs(errors)))

    #compute the relative yield
    crop_obs = np.load(simu_path + "/crop_obs.npy")
    k_s = crop_obs[:, 7]
    k_y = crop_obs[:, 10]

    relative_yield = np.exp(np.mean(np.log(1 - k_y * (1 - k_s))))
    relative_yields.append(relative_yield)
    water_usage = np.sum(v_reqs)
    water_usages.append(water_usage)
    break
#%%

#%% Plotting the results in bar plots

# plot the relative yields

plt.figure(figsize=(10, 6))
plt.bar(simu_names, relative_yields)
plt.xlabel('Simulation Cases')
plt.ylabel('Relative Yield')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "relative_yields.png")

# plot the water usages
plt.figure(figsize=(10, 6))
plt.bar(simu_names, water_usages)
plt.xlabel('Simulation Cases')
plt.ylabel('Water Usage (m3)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "water_usages.png")

# plot the energy purchased
plt.figure(figsize=(10, 6))
plt.bar(simu_names, energy_purchased)
plt.xlabel('Simulation Cases')
plt.ylabel('Energy Purchased (kWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "energy_purchased.png")


# plot the reference tracking error
plt.figure(figsize=(10, 6))
plt.bar(simu_names, ref_tracking_error)
plt.xlabel('Simulation Cases')
plt.ylabel('Reference Tracking Error (kW)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "ref_tracking_error.png")




# %%
