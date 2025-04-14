#%% This script is going to be used for testing of the different environments
import pandas as pd
import time
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm

from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from stable_baselines3 import TD3, PPO, SAC

from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from environments.WMS_env import CultivateEnv, NormalizedWMS, EvalWMS, TestWMS

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma= .075*np.ones(1)) #NormalActionNoise(mean=np.zeros(1), sigma= 1*np.ones(1))

def create_wrapped_env(log_file=None, weights=None):
    if weights is not None:
        env = NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=weights)
    else:
        env = NormalizedWMS(CultivateEnv())
    if log_file is not None:
        env = Monitor(env, log_file)
    return env

def create_eval_env(log_file=None, weights=None):
    if weights is not None:
        env = NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=weights)
    else:
        env = NormalizedWMS(CultivateEnv())
    eval_env = EvalWMS(env)
    if log_file is not None:
        eval_env = Monitor(eval_env, log_file)
    return env

# Create the vectorized environment
def create_callback(alg_name, environment):
    return EvalCallback(environment, 
                        best_model_save_path=f'{path}{alg_name}',
                        log_path=f'{path}{alg_name}', 
                        eval_freq=episode_length*n_envs*4,
                        deterministic=True, render=False)



episode_length = 114
n_envs = 4

total_timesteps = 600_000


#%%
path = ""

vec_env = make_vec_env(lambda: create_wrapped_env(), n_envs=n_envs)
eval_env = create_eval_env(f"{path}ppo/ppo_monitor.csv")


ppo_model: BaseAlgorithm = PPO("MlpPolicy", vec_env, verbose=1, 
                               batch_size=episode_length, 
                               device="cpu", 
                               clip_range=0.18, 
                               n_steps=episode_length*n_envs*4, 
                               learning_rate=0.0001, 
                               gae_lambda=.99,
                               n_epochs=5)
                


start_time = time.time()
ppo_model.learn(total_timesteps=total_timesteps, callback=create_callback("ppo", eval_env))

end_time = time.time()
training_time = end_time - start_time

print(f"Training ppo took {end_time - start_time} seconds")
# %%
