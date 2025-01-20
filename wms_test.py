import numpy as np
import pandas as pd
import torch

from predictive_models.utils import NN_soil_mdl
from environments.WMS_env import *
from matplotlib import pyplot as plt
weather_data = pd.read_csv("environments/Data/WMS/weather_data.csv")
cultivate_env = Cultivates()
from environments.WMS_policy import ModelBasedIrrigationPolicy, TriggeredIrrigationPolicy


irr_policy = TriggeredIrrigationPolicy(1, 4, 2)

def create_rb_policy() -> ModelBasedIrrigationPolicy:
    """
    Creates a rule-based irrigation policy
    """
    theta_models = [torch.load(f"predictive_models/soil_moisture/theta_{5-j}.pth") for j in range(1, 5)]
    theta_models = [torch.load("predictive_models/soil_moisture/theta_evp.pth")] + theta_models
    root_length_model = torch.load("predictive_models/soil_moisture/root_depth.pth")
    theta_a_mdl = NN_soil_mdl(theta_models, root_length_model)

    return ModelBasedIrrigationPolicy(n_crops=1, neural_model=theta_a_mdl, root_length_model=root_length_model)

rb_policy = create_rb_policy()

print("a")
#%%

obs, info = cultivate_env.start()
simu_days = 135
prev_action = 0
actions = []
for i in range(simu_days):
    daily_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()
    action = rb_policy.get_action(obs["tomato"], prev_action, 0, 0)# irr_policy(obs)
    actions.append(action)
    obs = cultivate_env.step([action], daily_weather_data)

crop_data = cultivate_env.crops[0].hist_data
crop_data = np.array(crop_data)
soil_data = pd.DataFrame(cultivate_env.get_soil_data()[0])

#%%

plt.plot(actions)
plt.show()

#%% let's check after a period of simulation

plt.plot(-crop_data[:, 0], label="root depth")
for i in range(1, 5):
    plt.hlines(-0.15*i, 0, simu_days, color='gray', linestyles="--")
plt.ylim(-5*.15, 0)
plt.ylabel("Depth [m]")
plt.xlabel("Days since plantation")
plt.title("Root depth")
plt.savefig("root_depth.png", dpi=300)
plt.show()

#%%
plt.plot(crop_data[:,-1], label="Ke_bound")
plt.plot(crop_data[:,-2], label="K_r")
plt.plot(crop_data[:,-3], label="K_e")
plt.plot(crop_data[:,-4], label="K_s")
plt.legend()
plt.title("Kr")
plt.show()

#%%

plt.plot(crop_data[:,1], label="potential crop et")
plt.plot(crop_data[:,2], label="ref et")
plt.plot(crop_data[:,4], label="actual et evap")
plt.legend()
plt.title("Evapotranspiration")
plt.show()


#%% Lets plot the soil moisture data
soil_moisture = cultivate_env.get_soil_data()[0]
pd_soil_moisture = pd.DataFrame(soil_moisture)

plt.plot(soil_moisture["layer_4_0"], label="evp_layer")
plt.plot(soil_moisture["layer_3_0"], label="layer_4")
plt.plot(soil_moisture["layer_2_0"], label="layer_3")
plt.plot(soil_moisture["layer_1_0"], label="layer_2")
plt.plot(soil_moisture["layer_0_0"], label="layer_1")

theta_wp = cultivate_env.crops[0].soil.evp_layer.theta_wp
theta_fc = cultivate_env.crops[0].soil.evp_layer.theta_fc
theta_sat = cultivate_env.crops[0].soil.evp_layer.theta_sat
theta_res = cultivate_env.crops[0].soil.evp_layer.theta_res


plt.hlines(theta_wp,0, simu_days, color='gray', linestyles="--")
plt.text(simu_days, theta_wp, 'theta_wp', va='top', ha='right', color='black')

plt.hlines(theta_fc,0, simu_days, color='gray', linestyles="--")
plt.text(simu_days, theta_fc, 'theta_fc', va='bottom', ha='right', color='black')

plt.hlines(theta_sat, 0, simu_days, color='black', linestyles="--")
plt.text(simu_days, theta_sat, 'theta_sat', va='top', ha='right', color='black')

plt.hlines(theta_res,0, simu_days, color='black', linestyles="--")
plt.text(simu_days, theta_res, 'theta_res', va='bottom', ha='right', color='black')

fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.legend(ncol=2, loc = "upper center")
plt.title("Soil moisture")
plt.tight_layout()
plt.ylabel("Water content [m3/m3]")
plt.xlabel("Days since plantation")
plt.savefig("soil_moisture.png", dpi=300)
plt.show()

#%% Lets plot the water content data

plt.plot(soil_moisture["layer_4_3"]*1e3, label="evp_layer")
plt.plot(soil_moisture["layer_3_3"]*1e3, label="layer_4")
plt.plot(soil_moisture["layer_2_3"]*1e3, label="layer_3")
plt.plot(soil_moisture["layer_1_3"]*1e3, label="layer_2")
plt.plot(soil_moisture["layer_0_3"]*1e3, label="layer_1")
plt.hlines(cultivate_env.crops[0].soil.evp_layer.theta_wp*0.1*1e3,
           0, simu_days, color='gray', linestyles="--")
plt.hlines(cultivate_env.crops[0].soil.evp_layer.theta_fc*0.1*1e3,
              0, simu_days, color='gray', linestyles="--")

plt.hlines(cultivate_env.crops[0].soil.evp_layer.theta_sat*0.1*1e3,
                0, simu_days, color='black', linestyles="--")
plt.hlines(cultivate_env.crops[0].soil.evp_layer.theta_res*0.1*1e3,
                  0, simu_days, color='black', linestyles="--")
plt.ylabel("Water content [mm]")
fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.legend()
plt.title("Soil moisture")
plt.tight_layout()
plt.show()

#%% Let's plot the transpiration
plt.plot(crop_data[:, 2]*crop_data[:, 6], label="evaporation")

plt.plot(soil_moisture["layer_4_2"]*crop_data[:, 2]*crop_data[:,5], label="evp_layer")
plt.plot(soil_moisture["layer_3_2"]*crop_data[:, 2]*crop_data[:,5], label="layer_4")
plt.plot(soil_moisture["layer_2_2"]*crop_data[:, 2]*crop_data[:,5], label="layer_3")
plt.plot(soil_moisture["layer_1_2"]*crop_data[:, 2]*crop_data[:,5], label="layer_2")
plt.plot(soil_moisture["layer_0_2"]*crop_data[:, 2]*crop_data[:,5], label="layer_1")

plt.legend()
plt.title("Transpiration")
plt.show()

#%% Let's plot the partial Ks
plt.plot(soil_moisture["layer_4_2"])
plt.plot(soil_moisture["layer_3_2"])
plt.plot(soil_moisture["layer_2_2"])
plt.plot(soil_moisture["layer_1_2"])
plt.plot(soil_moisture["layer_0_2"])

# check if the sum is 1
plt.plot(soil_moisture["layer_4_2"] + soil_moisture["layer_3_2"] + soil_moisture["layer_2_2"] +
         soil_moisture["layer_1_2"] + soil_moisture["layer_0_2"])

plt.show()

#%%


