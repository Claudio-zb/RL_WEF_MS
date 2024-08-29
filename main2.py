import numpy as np
from matplotlib import pyplot as plt
from environments.WMS_env import *
import pandas as pd

#%%

env = WMS()

env.start()
for i in range(200):
    if i == 54:
        print("here")
    env.step()

hist_data = env.get_hist_data()

#%%
crop_df = pd.DataFrame(hist_data["tomato"][0])
# crop_df.plot()
# plt.show()
soil_df = pd.DataFrame(hist_data["tomato"][1])

#%%

crop_df.plot(y=["root_depth"])
plt.ylabel("root depth [m]")
plt.xlabel("days since plantation")
plt.title("Root depth of tomato plant")
plt.show()

#%%

crop_df.plot(y=["pcrop_evapotranspiration", "crop_evapotranspiration"])
plt.ylabel("Evapotranspiration [mm]")
plt.xlabel("days since plantation")
plt.title("Potential and actual evapotranspiration of tomato plant")
plt.show()
#%%

crop_df.plot(y=["Ks"])
plt.xlabel("days since plantation")
plt.title("Stress coefficient")
plt.show()
# sum of water contents
#%%
sum_w = (hist_data["tomato"][1]["layer_4_4"] -
         hist_data["tomato"][1]["layer_3_4"] +
         0*hist_data["tomato"][1]["layer_2_3"] +
         0*hist_data["tomato"][1]["layer_1_3"] +
         0*hist_data["tomato"][1]["layer_0_3"])

# income and outcome
income = hist_data["tomato"][1]["w_in"]
outcome = hist_data["tomato"][1]["W_out"]


plt.plot(np.sign(sum_w))
plt.legend()
plt.show()

#%%

plt.plot(hist_data["tomato"][1]["layer_4_4"], label="layer_4")
plt.plot(hist_data["tomato"][1]["layer_3_4"], label="layer_3")
plt.plot(hist_data["tomato"][1]["layer_2_4"], label="layer_2")
plt.plot(hist_data["tomato"][1]["layer_1_4"], label="layer_1")
plt.plot(hist_data["tomato"][1]["layer_0_4"], label="evp_layer")
plt.xlabel("days")
plt.ylabel("water content [mm]")
plt.title("Water content in soil layers")
plt.legend()
plt.savefig("soil_layers.png", dpi=300)
plt.show()
#%%

for key in hist_data["tomato"][1]:
    print(key, len(hist_data["tomato"][1]))

