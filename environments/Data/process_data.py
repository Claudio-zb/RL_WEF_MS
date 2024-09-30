import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("environments/Data/EMS/raw_data/calan_2004_2016.csv", skiprows=41)
data = data.interpolate()
desired_columns = ['Fecha/Hora', 'temp', 'dir']
data = data[desired_columns]

#%% Split data by year

data["Fecha/Hora"] = pd.to_datetime(data["Fecha/Hora"])
data['doy'] = data['Fecha/Hora'].dt.dayofyear
grouped_by_year = data.groupby(data["Fecha/Hora"].dt.year)
a = []
for year, group in grouped_by_year:
    group.set_index('Fecha/Hora', inplace=True)
    resampled_data = group.resample('10T').interpolate()

    a.append(resampled_data)
#%%
for idx, df in enumerate(a):
    df.to_csv(f"environments/Data/EMS/calan_{2004+idx}.csv")
#%%
data = pd.read_csv("environments/Data/EMS/calan_2006.csv")