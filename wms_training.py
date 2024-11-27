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
                 log_path=f'./logs/wms/{alg_name}', eval_freq=100,
                 deterministic=True, render=False)
#%%

cultivate_env = create_wrapped_env("./logs/wms/sac_monitor.csv")
train = False
if train:
    model = SAC("MlpPolicy", cultivate_env, verbose=1, gradient_steps=-1)
    model.learn(total_timesteps=10_000, callback=create_callback("sac", cultivate_env))

else:
    model = SAC.load("./logs/wms/sac/best_model.zip")

#%% plot the training curves

df = pd.read_csv(f"./logs/wms/sac_monitor.csv", skiprows=1)

#%% perform the evaluation
x = []
a = []
rews = []
obs, _ = cultivate_env.reset()
x.append(obs)
done = False

while not done:
    #action, _states = model.predict(obs, deterministic=True)
    action = cultivate_env.action_space.sample()
    a.append(action)
    obs, rewards, terminated, truncated, info = cultivate_env.step(action)
    rews.append(rewards)
    x.append(obs)
    done = terminated or truncated
#%%
x = np.array(x)
a = np.array(a)
rews = np.array(rews)

#%%
plt.plot(rews)
plt.title("Episode Reward")
plt.xlabel("Days since plantation")
plt.ylabel("Reward")
plt.show()

#%%
plt.plot(a)
plt.title("Actions")
plt.xlabel("Days since plantation")
plt.ylabel("Irrigation depth [mm]")
plt.show()

#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(x[:,0])
ax[0].set_title("Depletion")
ax[0].set_xlabel("Days since plantation")
ax[0].set_ylabel(r"Depletion [\%]")

ax[1].plot(x[:,1], label = f"mad = {x[:,2]}")
ax[1].set_title("RAW")
plt.tight_layout()
plt.show()









