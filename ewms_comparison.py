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

ems_mpc_1_day = MPCPumpingPolicy(pv_model=pv_model, pd_model=pd_model, days_ahead=0)
ems_mpc_2_day = MPCPumpingPolicy(pv_model=pv_model, pd_model=pd_model, days_ahead=1)
ems_rl = RLPumpingPolicy(1, ems_rl_model, isNormalized=True)
ems_rb = RBPumpingPolicy(1, v_tank_max=5)

#%% Setting up the simulation environments
random_seed = 42

simu_mpc_rl = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                      ems_policy=copy.deepcopy(ems_rl), seed=random_seed)

simu_mpc_rb = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                      ems_policy=copy.deepcopy(ems_rb), seed=random_seed)

simu_mpc_mpc = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_mpc_policy),
                       ems_policy=ems_mpc_2_day, seed=random_seed)

simu_rl_rl = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_rl_policy),
                      ems_policy=copy.deepcopy(ems_rl), seed=random_seed)

simu_rl_rb = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_rl_policy),
                      ems_policy=copy.deepcopy(ems_rb), seed=random_seed)

simu_rl_mpc = SimuEnv(irrigation_policy=copy.deepcopy(irrigation_rl_policy),
                        ems_policy=copy.deepcopy(ems_mpc_1_day), seed=random_seed)

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
run = False
if run:
    for simu, name in zip(simu_cases, simu_names):
        print(f"Running {name} case study...")
        start_time = time.time()    
        np.random.seed(random_seed), random.seed(random_seed)
        simu.run(init_doy=295, total_days=115-30) #115-30)

        end_time = time.time()
        print(f"Finished {name} case study in {end_time - start_time:.2f} seconds.")
        
        mg_data, crop_data, soil_data = simu.get_simu_data()
        simu_path = path + name

        if not os.path.exists(simu_path):
            os.makedirs(simu_path)

        with open(simu_path + "/mg_data.pkl", "wb") as f:
            pickle.dump(mg_data, f)

        with open(simu_path + "/crop_data.pkl", "wb") as f:
            pickle.dump(crop_data, f)

        with open(simu_path + "/soil_data.pkl", "wb") as f:
            pickle.dump(soil_data, f)

#%% compute the metrics

energy_purchased = []
net_energy = []
ref_tracking_error = []
relative_yields = []
water_usages = []

plot_days = 2
tt = np.linspace(0,plot_days*24, plot_days*144)

fig, axs = plt.subplots(3, 1, figsize = (7,7))
fig2, axs2 = plt.subplots(2, 1, figsize = (7,5))
linestyles = ["-", "-", "-"]#["dashed", "solid", "-."]
colors = ["tab:blue", "tab:orange", "tab:green"]

for idx, name in enumerate(simu_names):
    simu_path = path + name
    crop_data = pickle.load(open(simu_path + "/crop_data.pkl", "rb"))
    soil_data = pickle.load(open(simu_path + "/soil_data.pkl", "rb"))
    mg_data = pickle.load(open(simu_path + "/mg_data.pkl", "rb"))

    mg_obs = mg_data["mg_obs"]
    energy_purchased.append(np.sum(np.clip(mg_obs[:,5], -np.inf, 0)))
    
    mg_obs_144 = mg_data["end_of_day_samples"]
    v_reqs = crop_data["v_reqs"]
    #compute the relative yield
    crop_obs = crop_data["cultivate_obs"]
    k_s = crop_obs[:, 7]
    k_y = crop_obs[:, 10]
    energy_balance = np.sum(mg_obs[:, 5])  # energy balance
    net_energy.append(energy_balance)

    relative_yield = np.exp(np.mean(np.log(1 - k_y * (1 - k_s))))
    relative_yields.append(relative_yield)
    water_usage = np.sum(mg_obs_144[:,1])
    mask = v_reqs != 0
    errors = np.zeros_like(v_reqs)
    errors[mask] = np.abs((v_reqs[mask] - mg_obs_144[:, 1][mask]) / v_reqs[mask]) * 100  # MAPE for nonzero v_reqs
    #print(errors[:5])

    ref_tracking_error.append(np.mean(np.abs(errors)))  # in percentage
    water_usages.append(water_usage)
    if idx > 2:
        label = f"{name[3:].upper()}"
        if label == "MPC":
            label = "MPC-1"
        #label2 = r"$\pi_{we}^{" + f"{name[3:].upper()}" + "}$"
        
        axs[2].plot(tt, mg_data["mg_actions"][144:144*(plot_days+1), 0],label = label, ls = linestyles[np.mod(idx, 3)])
        
        n_vreqs = np.array([v_reqs[1]]*144 + [v_reqs[2]]*144) 
        axs2[1].plot(tt, n_vreqs - mg_data["mg_obs"][144+1:144*(plot_days+1)+1, 1], label = label)
    else:
        label = f"{name[4:].upper()}"
        if label == "MPC":
            label = "MPC-2"
        #label2 = r"$\pi_{we}^{" + f"{name[4:].upper()}" + "}$"
        axs[1].plot(tt, mg_data["mg_actions"][144:144*(plot_days+1), 0],label = label, ls = linestyles[np.mod(idx, 3)])


        n_vreqs = np.array([v_reqs[1]]*144 + [v_reqs[2]]*144) 
        axs2[0].plot(tt, n_vreqs - mg_data["mg_obs"][144+1:144*(plot_days+1)+1, 1], label = label)

axs[0].plot(tt, mg_data["mg_dis"][144:144*(plot_days+1), 0], color = "tab:purple")
axs[0].set_ylabel(r"Solar radiation $(kW/m^2$)")
axs[0].grid(which = "both")
#axs[0].plot(mg_data["mg_obs"][144:144*(plot_days+1), 5])
axs[2].set_title("RL-based WF-MS")
axs[2].set_ylabel(r"Water extraction $(l/s)$")
axs[2].set_xlabel("Time (hours)")
axs[2].set_ylim(0,1)
axs[2].grid(which = "both")

axs[1].set_title("MPC-based WF-MS")
axs[1].set_ylabel(r"Water extraction $(l/s)$")
axs[1].set_ylim(0,1)
axs[1].grid(which = "both")

axs2[0].set_title("MPC-based WF-MS")
axs2[0].legend()
axs2[1].set_title("RL-based WF-MS")
axs2[1].legend()
axs2[0].grid(which = "both")
axs2[1].grid(which = "both")

axs2[0].set_ylabel(r"Irrigation Error $(l/s)$")
axs2[1].set_ylabel(r"Irrigation Error $(l/s)$")

axs2[1].set_xlabel(r"Time (hours)")


for ax in axs[1:]: ax.legend()
fig.tight_layout()
fig2.tight_layout()

fig.savefig("pumpings.pdf", dpi=300)
fig2.savefig("errors.pdf", dpi=300)


#%% Plotting the results in bar plots
simu_names2 = [name.replace("_", "+").upper() for name in simu_names]

#%%
# plot the relative yields


colors = plt.cm.tab10.colors  # Use a colormap to assign different colors

# Plot the relative yields
plt.figure(figsize=(8, 4))
plt.bar(simu_names2, np.array(relative_yields)*100, color=colors[:len(simu_names)])
#plt.xlabel('Simulation Cases')
plt.ylabel('Relative Yield \%')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "relative_yields.png", dpi=300)

# Plot the water usages
plt.figure(figsize=(8, 4))
plt.bar(simu_names2, water_usages, color=colors[:len(simu_names)])
plt.xlabel('Simulation Cases')
plt.ylabel('Water Usage (m3)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "water_usages.png", dpi=300)

# Plot the energy purchased
plt.figure(figsize=(8, 4))
plt.grid(axis='y', alpha=0.75)
plt.bar(simu_names2, np.abs(energy_purchased), color=colors[:len(simu_names)])
#plt.xlabel('Simulation Cases')
plt.ylabel('Energy Purchased (kWh)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "energy_purchased.png", dpi=300)

plt.figure(figsize=(8, 4))
plt.grid(axis='y', alpha=0.75)
plt.bar(simu_names2, net_energy, color=colors[:len(simu_names)])
#plt.xlabel('Simulation Cases')
plt.ylabel('Energy Purchased (kWh)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "net_energy.png", dpi=300)

# Plot the reference tracking error
plt.figure(figsize=(8, 4))
plt.bar(simu_names2, ref_tracking_error, color=colors[:len(simu_names)])
plt.xlabel('Simulation Cases')
plt.ylabel('Reference Tracking Error %')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(path + "ref_tracking_error.png", dpi=300)


# %%

plt.plot(crop_data["v_irrs"])
plt.plot(crop_data["v_reqs"])

# %%
