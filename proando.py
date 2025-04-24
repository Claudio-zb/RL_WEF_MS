from environments.Cultivates import * 
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

cultivates_env = Cultivates()
cultivates_env2 = Cultivates()
weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")

obs, doy = cultivates_env.start()
cultivates_env2.start()
for i in range(12):
    obs, doy = cultivates_env.step([0], weather_data.iloc[i].to_dict())
cultivates_env2.set_state(obs, doy)

obs2, doy2 = cultivates_env2.step([0], weather_data.iloc[12].to_dict())
obs, doy = cultivates_env.step([0], weather_data.iloc[12].to_dict())

print(obs, doy)
print(obs2, doy2)

print("yeah whatever")