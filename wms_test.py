#%%
from stable_baselines3 import PPO, SAC, TD3
from predictive_models.utils import NN_soil_mdl
from environments.Cultivates import *
from matplotlib import pyplot as plt
import pandas as pd
from environments.WMS_env import NormalizedWMS, CultivateEnv
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
cultivate_env = Cultivates()
from environments.WMS_policies import MPCIrrigationPolicy, RLIrrigationPolicy, ScheduledIrrigationPolicy
cultivate_gym_env = NormalizedWMS(CultivateEnv(), 1)

agent = TD3.load(r"logs\wms\weights_1.0_1.0_1.0\td3\best_model")

scheduled_policy = ScheduledIrrigationPolicy(n_crops=1, frequency=0, irr_amount=20.0)
mpc_policy = MPCIrrigationPolicy(model=Cultivates(), n_crops=1, horizon=10)
#rl_policy = RLIrrigationPolicy(n_crops=1, model_path="models/ppo_model.zip")
year = 2003 
# then get the index of the day of the year of that year
doy = min([crop.plantation_day for crop in cultivate_env.crops])  
index = weather_data[(weather_data["doy"] == doy) & (weather_data["year"] == year)].index.values.item()

days_ahead = 1

#%%

obs, info = cultivate_gym_env.reset()
actions = []
rewards = []
observations = []
done = False
while not done:
    #action = scheduled_policy.get_action(obs)/20*1000/10
    action = agent.predict(obs, deterministic=True)[0]
    actions.append(action)
    obs, reward, terminated, truncated, info = cultivate_gym_env.step(action)
    observations.append(obs)
    done = terminated or truncated
    rewards.append(reward)
observations = np.array(observations)
#%%
plt.plot(observations[:,8])
#%%
plt.plot(rewards)
plt.title("Scheduled irrigation policy")
#%%

crop_data = pd.DataFrame(cultivate_gym_env.env.cultivates.get_hist_data()["potato"][0])
soil_data = pd.DataFrame(cultivate_gym_env.env.cultivates.get_hist_data()["potato"][1])

#%%

soil_data.plot(y=["layer_0_0", "layer_1_0", "layer_2_0", "layer_3_0", "layer_4_0"])
plt.title("Soil moisture")

#%%

obs, info = cultivate_env.start()
simu_days = 115
prev_action = 0
actions = []
for i in range(simu_days):
    daily_weather_data = weather_data.iloc[index + i].to_dict()
    precipitation = weather_data.iloc[index+i:index+i+days_ahead]["precipitation"].values
    et0 = weather_data.iloc[index+i:index+i+days_ahead]["ET_0"].values
    timestamp = np.array([index+i])
    disturbances = np.hstack((precipitation, et0, timestamp))
    action = mpc_policy.get_action(obs, disturbances)
    actions.append(action)
    obs, _ = cultivate_env.step(action, daily_weather_data)
    prev_action = action

#%%
# let's check the data

crop_data, soil_data = cultivate_env.get_hist_data()["potato"]

soil_data = pd.DataFrame(soil_data)
soil_data.to_csv("soil_data_2.csv")
crop_data = pd.DataFrame(crop_data)
crop_data.to_csv("crop_data_2.csv")

#%%
actions_ = np.array([action[0] for action in actions])
np.save("actions_2.npy", actions_)
#%%

plt.plot(actions_*1000)
#plt.ylim(0, 10)
plt.show()

#%% Compute the total irrigation and yield
total_irrigation = np.zeros(simu_days)



#%% let's check after a period of simulation

plt.plot(-crop_data["root_depth"], label="root depth")
for i in range(1, 5):
    plt.hlines(-0.15*i, 0, simu_days, color='gray', linestyles="--")
plt.ylim(-5*.15, 0)
plt.ylabel("Depth [m]")
plt.xlabel("Days since plantation")
plt.title("Root depth")
#plt.savefig("root_depth.png", dpi=300)
plt.show()

#%%
plt.plot(crop_data["Kcb"], label="K_r")
plt.plot(crop_data["K_e"], label="K_e")
plt.plot(crop_data["K_s"], label="K_s")
plt.legend()
plt.title("Kr")
plt.show()

#%%

plt.plot(crop_data["ET_p"], label="potential crop et")
plt.plot(crop_data["ET_0"], label="ref et")
plt.plot(crop_data["ET_a"], label="actual et evap")
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
#plt.savefig("soil_moisture.png", dpi=300)
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


