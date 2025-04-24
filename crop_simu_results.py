#%%
from environments.Cultivates import *
from matplotlib import pyplot as plt
from environments.WMS_policies import ModelBasedIrrigationPolicy, TriggeredIrrigationPolicy
from environments.Data.WMS.WMS_profile import * 

plots_path = "plots/agro_model/"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

weather_data = pd.read_csv("environments/Data/WMS/weather_data.csv")
# Setting up the environment


cultivate_env = Cultivates(crop_params=[potato])
irr_policy = TriggeredIrrigationPolicy(1, 2, 7)

#%%
simu_days = np.sum(potato["stages_duration"])
initial_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()
obs, info = cultivate_env.start(init_weather_info=initial_weather_data)
prev_action = 0
actions = []
for i in range(simu_days):
    daily_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()
    action = irr_policy(obs)[0]
    actions.append(action)
    obs = cultivate_env.step([action], daily_weather_data)

crop_data, soil_data = cultivate_env.crops[0].get_hist_data()
actions = np.array(actions)


#%% let's check after a period of simulation

t = np.cumsum([0] + potato["stages_duration"]) 
v0 = potato["root_depth_max"] +.2
v1 = potato["f_c"][1] + .2
stage_names = ["Initial Stage", "Development Stage", "Mid Season Stage", "Late Stage"]

#%%
simu_time = np.array(range(1, simu_days))
fig, axs = plt.subplots(2,1)
fig.set_size_inches(8.0, 5)


axs[0].scatter(simu_time, crop_data["root_depth"][:-1], label="root depth", s=10, alpha = .5)
axs[0].set_ylabel("Depth [m]")
axs[0].set_title("Root depth")
axs[0].invert_yaxis()

for idx, item in enumerate(t[:-1]):
    a, b = (t[idx], t[idx+1]) 
    axs[0].fill_between(simu_time[a:b], np.zeros_like(simu_time[a:b]), v0*np.ones_like(simu_time[a:b]), alpha = .2)
    axs[1].fill_between(simu_time[a:b], np.zeros_like(simu_time[a:b]), v1*np.ones_like(simu_time[a:b]), alpha = .2)

    axs[0].text((a + b)/2, v0, stage_names[idx], ha = "center")
    axs[1].text((a + b)/2, v1, stage_names[idx], ha = "center", va = "top")

axs[1].scatter(simu_time, crop_data["f_c"][:-1], color = "gray", alpha = .5, s = 5)
axs[1].set_ylim(0, v1)
axs[1].set_ylabel(r"Coverage percentage [\%]")
axs[1].set_xlabel("Days since plantation")
axs[1].set_title("Coverage")

plt.tight_layout()
plt.savefig(plots_path + "root_depth.png", dpi=300)
plt.show() 

#%%
color = "red"
v0 = -.95
v1 = .9
margin = .01
fig, ax = plt.subplots(1,1)
fig.set_size_inches(7.5, 3)
ax.set_ylabel("Depth [m]")
ax.set_xlabel("Time since plantation [days]")
ax.set_title("Root depth evolution")
layers_depth = np.array([0, .15, .35, .55, .75, .95])
layers_name = ["Evp Layer", "Layer 4", "Layer 3", "Layer 2", "Layer 1"]
for idx, item in enumerate(layers_depth[:-1]):
    ax.fill_between(simu_time, layers_depth[idx] + margin, layers_depth[idx+1], alpha  =.15, label = layers_name[idx])   
    #ax.text(t[0]/2, (layers_depth[idx] + layers_depth[idx+1])/2, layers_name[idx], ha = "center", va = "center")
    ax.vlines(t[idx], 0, layers_depth[-1], color=color, linestyles="--")

ax.scatter(simu_time, crop_data["root_depth"][1:], color = "gray", label="Root depth", s=10)

ax.legend(loc = "lower left")

# invert the y axis
ax.invert_yaxis()
ax.set_ylim(None, -0.02)
fig.tight_layout()
fig.savefig(plots_path + "root_depth.png", dpi=300)

#%% 

v0 = -.95
v1 = .9
margin = .01
fig, ax = plt.subplots(1,1)
fig.set_size_inches(7.5, 3)
ax.set_ylabel("Depth [m]")
ax.set_title("Root depth")
layers_depth = np.array([0, .15, .35, .55, .75, .95])
layers_name = ["Evp Layer", "Layer 4", "Layer 3", "Layer 2", "Layer 1"]
for idx, item in enumerate(layers_depth[:-1]):
    ax.vlines(t[idx], 0, 1, color=color, linestyles="--")
    ax.fill_between(simu_time, layers_depth[idx], layers_depth[idx+1] - margin, alpha  =.2)
    if idx != 4:
        ax.text((t[3]+t[4])/2, (layers_depth[idx] + layers_depth[idx+1])/2, layers_name[idx], ha = "center", va = "center")
    else:
        ax.text(t[1]/2, (layers_depth[idx] + layers_depth[idx+1])/2, layers_name[idx], ha = "center", va = "center")

ax.plot(simu_time, crop_data["root_depth"][1:], label="root depth")

# invert the y axis
ax.invert_yaxis()
fig.tight_layout()
ax.set_ylim(1.0, -0.02)

#%%
fig, ax = plt.subplots(1,1)
fig.set_size_inches(7.5, 3)
ax.plot(simu_time, crop_data["f_c"][1:], color = "gray")
ax.vlines(t[0], 0, 1, color=color, linestyles="--")
ax.vlines(t[1], 0, 1, color=color, linestyles="--")
ax.vlines(t[2], 0, 1, color=color, linestyles="--")
ax.vlines(t[3], 0, 1, color=color, linestyles="--")
ax.set_ylabel(r"Coverage percentage [\%]")
ax.set_xlabel("Days since plantation")
ax.set_title("Leaf coverage evolution")

ax.text(t[0]/2, v1, "Initial Stage", ha = "center")
ax.text((t[0]+t[1])/2, v1, "Development Stage", ha = "center")
ax.text((t[1]+t[2])/2, v1, "Mid Season Stage", ha = "center")
ax.text((t[2]+t[3])/2, v1, "Final Stage", ha = "center")
fig.tight_layout()
fig.savefig(plots_path + "coverage.png", dpi=300)




#%%
#plt.plot(crop_data["Ks"], label=r"$K_{\text{e bound}}$")
#plt.plot(crop_data[:,-3], label=r"$K_r$")
#plt.plot(crop_data[:,-4], label=r"$K_e$")
plt.plot(crop_data["K_s"], label=r"$K_s$")
fig = plt.gcf() 
fig.set_size_inches(6,3)
plt.xlabel("Time since plantation [days]")
plt.legend()
plt.title("Crop coefficient evolution")
plt.tight_layout()
plt.show()

#%%

plt.plot(crop_data["ET_p"], label= r"$ET_p$")
plt.plot(crop_data["ET_0"], label= r"$ET_0$")
plt.plot(crop_data["ET_a"], label= r"$ET_a$")
fig = plt.gcf() 
fig.set_size_inches(7.5,3)
plt.legend()
plt.xlabel("Days since plantation")
plt.ylabel("Water depth [mm]")
plt.legend()
plt.title("Evapotranspirations")
plt.grid()
plt.tight_layout()
plt.savefig(plots_path + "evapotranspirations.png", dpi=300)
plt.show()

#%%
crop_evp = crop_data["ET_a"][1:]

plt.plot(simu_time, crop_evp, label= r"$ET_a$", color = "darkgreen")
plt.plot(simu_time, crop_evp, label= r"$ET_a$", color = "darkgreen")
evaporation = crop_data["ET_0"][1:]*crop_data["K_e"][1:]
transpiration = crop_evp - evaporation

plt.fill_between(simu_time, 0, evaporation, color = "aquamarine",alpha=0.5)
plt.fill_between(simu_time, evaporation, crop_evp, color = "limegreen", alpha=0.5)

# Adding text annotations
mid_point = simu_days // 2

plt.text(t[1]/2, 0.1, 'Evaporation', color='tab:blue', ha='center')


plt.text((t[1]+t[2])/2, (crop_evp[(t[1]+t[2])//2] + evaporation[(t[1]+t[2])//2]) / 2, 'Transpiration', color='green', ha='center')


fig = plt.gcf() 
fig.set_size_inches(7.5,3)
plt.legend()
plt.xlabel("Time since plantation [days]")
plt.ylabel("Water loss [mm]")
plt.legend(loc = "upper right")
plt.title("Tomato crop evapotranspiration during a season")
plt.grid()
plt.tight_layout()
plt.savefig(plots_path + "evapotranspiration.png", dpi=300)
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
#plt.text(simu_days, theta_res+0.01, r'$\theta_{res}$', va='bottom', ha='right', color='black')

fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.legend(ncol=2, loc = "lower left")
plt.title("Soil moisture")
plt.tight_layout()
plt.ylabel("Volumetric water content [m3/m3]")
plt.xlabel("Days since plantation")
plt.ylim(0.1, 0.33)
plt.grid()
plt.savefig(plots_path + "soil_moisture.png", dpi=300)

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

#%%

plt.step(simu_time, actions[1:]*1000)
fig = plt.gcf() 
fig.set_size_inches(7.5, 3)
plt.ylabel("Irrigated water [mm]")
plt.xlabel("Day since plantation")
plt.tight_layout()
plt.savefig(plots_path + "irrigation.png", dpi=300)
plt.show()

#%%
fig, axs = plt.subplots(2,1)
axs[0].plot(np.abs(crop_data["avg_h_c"]))

axs[1].plot(np.log(np.abs(crop_data["avg_h_c"]))/375)

#%%

def gen_ideal_curves(crop:dict) -> tuple[np.ndarray, np.ndarray]:
    """returns the ideal root depth and foliage covered curves for a given crop"""
    min_depth = crop["root_depth_init"]
    max_depth = crop["root_depth_max"]
    f_c_max = crop["f_c"][1]
    f_c_min1 = crop["f_c"][0]
    f_c_min2 = crop["f_c"][2]
    stages = crop["stages_duration"]
    t = np.cumsum([0] + stages)
    root_depth = np.zeros(t[-1])
    f_c = np.zeros(t[-1])

    root_depth[0] = min_depth
    f_c[0] = f_c_min1
    for i in range(1, t[-1]):
        if i <= t[1]:
            root_depth[i] = min_depth
            f_c[i] = f_c_min1 
        elif i <= t[2]:
            root_depth[i] = min_depth + (max_depth - min_depth) * (i - t[1]) / (t[2] - t[1])
            f_c[i] = f_c_min1 + (f_c_max - f_c_min1) * (i - t[1]) / (t[2] - t[1])
        elif i <= t[3]:
            f_c[i] = f_c_max
            root_depth[i] = max_depth
        else:
            root_depth[i] = max_depth
            f_c[i] = f_c_max + (f_c_min2 - f_c_max) * (i - t[3]) / (t[4] - t[3])
    return root_depth, f_c
        

#%%
root_depth, f_c = gen_ideal_curves(potato)
plt.plot(root_depth)
plt.plot(f_c)




