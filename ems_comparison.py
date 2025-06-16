#%%
from environments.Cultivates import Cultivates
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy
from environments.EMS_policies import RLPumpingPolicy
import numpy as np
from copy import deepcopy
from stable_baselines3 import SAC, TD3, PPO
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
from matplotlib.colors import to_hex


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

year = 2017

irrigation_policy = RLIrrigationPolicy(n_crops=1, 
                                       rl_policy=SAC.load("logs/wms/weights_5/sac/best_model.zip"), 
                                       year=year)
seed = 42
set_of_weights = np.array([[1., 4.0, 1.],
                           [1., 3.0, 1.],
                           [1., 2.0, 1.],
                           [1., 1., 1.],
                           [1., 1., 2.],
                           [1., 1., 3.],
                           [1., 1., 4.], 
                           [1., 2., 2.], 
                           [1., 4., 4.]])
#%%
run = False

if run:

    stats = np.zeros((3, len(set_of_weights), 2))  # [n agents, n weights, n metrics]

    for idx, weights in enumerate(set_of_weights):

        path = f"logs/ems/weights_{idx}/"
        sac_agent = RLPumpingPolicy(1, SAC.load(path + "sac/best_model.zip"), True)
        td3_agent = RLPumpingPolicy(1, TD3.load(path + "td3/best_model.zip"), True)
        ppo_agent = RLPumpingPolicy(1, PPO.load(path + "ppo/best_model.zip"), True)
        agents = [sac_agent, td3_agent, ppo_agent]

        for jdex, name, agent in zip([0, 1, 2], ["sac", "td3", "ppo"], agents):
            simu_env = SimuEnv(irrigation_policy=irrigation_policy, 
                            ems_policy=agent,
                            year=year, seed=seed)
            p_day = simu_env.cultivate_env.crops[0].plantation_day
            simu_env.run(init_doy=p_day, total_days=115-30)

            mg_data, crop_data, soil_data = simu_env.get_simu_data()    
            mg_obs = mg_data["mg_obs"]
            mg_obs_144 = mg_data["end_of_day_samples"]
            v_reqs = crop_data["v_reqs"]
            errors = v_reqs - mg_obs_144[:, 1]  # water requirements vs actual water usage
            stats[jdex, idx, 0] = np.mean(np.abs(errors)) # in percentage
            stats[jdex, idx, 1] = np.sum(np.clip(mg_obs[:,5], -np.inf, 0)) 

            
    np.save("logs/ems/stats.npy", stats) 

# %%

stats = np.load("logs/ems/stats.npy")
stats[:,:,1] = np.abs(stats[:, :, 1]) # bought energy in kWh

min_val = np.inf
index = None
for j, jtem in enumerate(stats):
    for i, item in enumerate(jtem):
        if item[0]*1000 < 10.0:
            if item[1] < min_val:
                min_val = item[1]
                index = (j, i)
            

algorithms = ["SAC", "TD3", "PPO"]
rows = []
for i, weights in enumerate(set_of_weights):
    row = [r"$\bar{\lambda}_1$ = "+ f"{weights[1]:.2f} / " + r"$\bar{\lambda}_2$ = " + f"{weights[2]:.2f}"]
    for j in range(3):
        error = stats[j, i, 0]*1000
        energy = stats[j, i, 1]
        cell = f"{error:.2f} / {energy:.0f}"
        cell_fmt = cell
        if (j, i) == index:
            cell_fmt = f"\\textbf{{{cell}}}"
        if error < 10.0:
            cell_fmt = f"\\cellcolor{{gray!30}}{cell_fmt}"
        row.append(cell_fmt)
    rows.append(row)

header = ["Weights"] + algorithms
latex_table = tabulate(rows, headers=header, tablefmt="latex_raw", stralign="center", numalign="center")
with open("logs/ems/table.tex", "w") as f:
    f.write(latex_table)

# %%
