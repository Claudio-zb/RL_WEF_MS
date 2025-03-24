#%%
import torch

from predictive_models.utils import NN_soil_mdl
from environments.WMS_env import *
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


weather_data = pd.read_csv("environments/Data/WMS/weather_data.csv")
cultivate_env = Cultivates()
from environments.WMS_policy import ModelBasedIrrigationPolicy, TriggeredIrrigationPolicy


irr_policy = TriggeredIrrigationPolicy(1, 4, 20)


print("a")
#%%

obs, info = cultivate_env.start()
simu_days = 135
prev_action = 0
actions = []
for i in range(simu_days):
    daily_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()
    action = irr_policy(obs)[0]
    actions.append(action)
    obs = cultivate_env.step([action], daily_weather_data)

crop_data = cultivate_env.crops[0].hist_data
crop_data = np.array(crop_data)
soil_data = pd.DataFrame(cultivate_env.get_soil_data()[0])
actions = np.array(actions)

#%%

plt.plot(actions*1000)
fig = plt.gcf() 
fig.set_size_inches(7.5, 3)
plt.ylabel("Irrigated water [mm]")
plt.xlabel("Day since plantation")
plt.tight_layout()
plt.savefig("irrigation.png", dpi=300)
plt.show()

#%%
print(len)

#%% let's check after a period of simulation
t = np.array(range(0, simu_days))
fig, axs = plt.subplots(2,1)

fig.set_size_inches(7.5, 5)
axs[0].plot(t, -crop_data[:, 1], label="root depth")
for i in range(1, 5):
    axs[0].hlines(-0.15*i, 0, simu_days, color='gray', linestyles="--")
#axs[0].set_ylim(-5*.17, 0)
axs[0].set_ylabel("Depth [m]")
#plt.xlabel("Days since plantation")
axs[0].set_title("Root depth")
axs[0].fill_between(t[0:30], np.zeros_like(t[0:30]), -np.ones_like(t[0:30]), alpha = .2)
axs[0].fill_between(t[30:70], np.zeros_like(t[30:70]), -np.ones_like(t[30:70]), alpha = .2)
axs[0].fill_between(t[70:110], np.zeros_like(t[70:110]), -np.ones_like(t[70:110]), alpha = .2)
axs[0].fill_between(t[110:135], np.zeros_like(t[110:135]), -np.ones_like(t[110:135]), alpha = .2)
axs[0].text(12, -1, "Initial Stage")


axs[1].plot(t, crop_data[:, 0])
axs[1].fill_between(t[0:30], np.zeros_like(t[0:30]), np.ones_like(t[0:30]), alpha = .2)
axs[1].fill_between(t[30:70], np.zeros_like(t[30:70]), np.ones_like(t[30:70]), alpha = .2)
axs[1].fill_between(t[70:110], np.zeros_like(t[70:110]), np.ones_like(t[70:110]), alpha = .2)
axs[1].fill_between(t[110:135], np.zeros_like(t[110:135]), np.ones_like(t[110:135]), alpha = .2)
axs[1].set_ylabel(r"Coverage percentage [\%]")
axs[1].set_xlabel("Days since plantation")
axs[1].set_title("Coverage")
plt.tight_layout()
plt.savefig("root_depth.png", dpi=300)
plt.show()



#%%
plt.plot(crop_data[:,-2], label=r"$K_{\text{e bound}}$")
plt.plot(crop_data[:,-3], label=r"$K_r$")
plt.plot(crop_data[:,-4], label=r"$K_e$")
plt.plot(crop_data[:,-5], label=r"$K_s$")
fig = plt.gcf() 
fig.set_size_inches(6,3)
plt.xlabel("Days since plantation")
plt.legend()
plt.title("Kr")
plt.tight_layout()
plt.show()

#%%

plt.plot(crop_data[1:,1], label= r"$ET_p$")
plt.plot(crop_data[1:,2], label= r"$ET_0$")
plt.plot(crop_data[1:,4], label= r"$ET_a$")
fig = plt.gcf() 
fig.set_size_inches(7.5,3)
plt.legend()
plt.xlabel("Days since plantation")
plt.ylabel("Water depth [mm]")
plt.legend()
plt.title("Evapotranspirations")
plt.grid()
plt.tight_layout()
plt.savefig("evapotranspirations.png", dpi=300)
plt.show()

#%%

plt.plot(range(1, simu_days), crop_data[1:,4], label= r"$ET_a$", color = "darkgreen")
evaporation = crop_data[1:,-3]*crop_data[1:,4]
transpiration = crop_data[1:,4] - evaporation

plt.fill_between(range(1, simu_days), 0, evaporation, color = "aquamarine",alpha=0.5)
plt.fill_between(range(1, simu_days), evaporation, crop_data[1:,4], color = "limegreen", alpha=0.5)

# Adding text annotations
mid_point = simu_days // 2
plt.text(30, 0.1, 'Evaporation', color='tab:blue', ha='center')
plt.text(80, (crop_data[1:,4][80] + evaporation[80]) / 2, 'Transpiration', color='green', ha='center')


fig = plt.gcf() 
fig.set_size_inches(7.5,3)
plt.legend()
plt.xlabel("Time since plantation [days]")
plt.ylabel("Water loss [mm]")
plt.legend(loc = "upper right")
plt.title("Tomato crop evapotranspiration during a season")
plt.grid()
plt.tight_layout()
plt.savefig("evapotranspiration.png", dpi=300)
plt.show()


#%% Lets plot the soil moisture data
soil_moisture = cultivate_env.get_soil_data()[0]
pd_soil_moisture = pd.DataFrame(soil_moisture)

plt.plot(soil_moisture["layer_4_0"], label="Evp layer")
plt.plot(soil_moisture["layer_3_0"], label="Layer 4")
plt.plot(soil_moisture["layer_2_0"], label="Layer 3")
plt.plot(soil_moisture["layer_1_0"], label="Layer 2")
plt.plot(soil_moisture["layer_0_0"], label="Layer 1")
 
theta_wp = cultivate_env.crops[0].soil.evp_layer.theta_wp
theta_fc = cultivate_env.crops[0].soil.evp_layer.theta_fc
theta_sat = cultivate_env.crops[0].soil.evp_layer.theta_sat
theta_res = cultivate_env.crops[0].soil.evp_layer.theta_res


plt.hlines(theta_wp,0, simu_days, color='gray', linestyles="--")
plt.text(simu_days, theta_wp, r'$\theta_{wp}$', va='top', ha='right', color='black')

plt.hlines(theta_fc,0, simu_days, color='gray', linestyles="--")
plt.text(simu_days, theta_fc, r'$\theta_{fc}$', va='bottom', ha='right', color='black')

plt.hlines(theta_sat, 0, simu_days, color='black', linestyles="--")
#plt.text(simu_days, theta_sat, r'$\theta_{sat}$', va='top', ha='right', color='black')

plt.hlines(theta_res,0, simu_days, color='black', linestyles="--")
plt.text(simu_days, theta_res+0.01, r'$\theta_{res}$', va='bottom', ha='right', color='black')

fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.legend(ncol=2, loc = "lower left")
plt.title("Soil moisture")
plt.tight_layout()
plt.ylabel("Volumetric water content [m3/m3]")
plt.xlabel("Days since plantation")
plt.ylim(0.1, 0.33)
plt.grid()
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


