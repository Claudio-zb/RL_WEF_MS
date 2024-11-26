# This script is going to be used for testing of the different environments
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from environments.EMS_env import MicrogridEnv, NormalizationWrapper, RuleBasedEMS
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines3 import TD3, PPO, SAC

from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from environments.WMS_env import CultivateEnv

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))

def create_wrapped_env(log_file):
    env = CultivateEnv()
    env = Monitor(env, log_file)
    return env

# Create the vectorized environment
def create_callback(alg_name, environment):
    return EvalCallback(environment, best_model_save_path=f'./logs/wms/{alg_name}',
                 log_path=f'./logs/wms/{alg_name}', eval_freq=50,
                 deterministic=True, render=False)
#%%

cultivate_env = create_wrapped_env("./logs/wms/sac_monitor.csv")
model = SAC("MlpPolicy", cultivate_env, verbose=1, gradient_steps=-1)
model.learn(total_timesteps=1_000, callback=create_callback("sac", cultivate_env))

#%% plot the training curves

df = pd.read_csv(f"./logs/wms/sac_monitor.csv", skiprows=1)







