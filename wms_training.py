# This script is going to be used for testing of the different environments
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from stable_baselines3 import TD3, PPO, SAC

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from environments.WMS_env import CultivateEnv

action_noise = NormalActionNoise(mean=np.zeros(1), sigma= 3*np.ones(1))

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
train = True
if train:
    sac_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/wms/sac_monitor.csv"), n_envs=4, seed=0)
    td3_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/wms/td3_monitor.csv"), n_envs=4, seed=0)
    ppo_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/wms/ppo_monitor.csv"), n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)

    sac_env = create_wrapped_env("./logs/wms/sac_monitor.csv")
    td3_env = create_wrapped_env("./logs/wms/td3_monitor.csv")
    ppo_env = create_wrapped_env("./logs/wms/ppo_monitor.csv")

    td3_model = TD3("MlpPolicy", td3_env, action_noise=action_noise, verbose=1, gradient_steps=-1)
    ppo_model = PPO("MlpPolicy", ppo_env, verbose=1, batch_size=128, normalize_advantage=True,
                    use_sde=True, learning_rate=0.001)
    sac_model = SAC("MlpPolicy", sac_env, verbose=1, gradient_steps=-1)

    models = [ppo_model, sac_model, td3_model]
    envs = [ppo_env, sac_env, td3_env]
    models_name = ["ppo", "sac", "td3"]
    for model, name, env in zip(models, models_name, envs):
        model.learn(total_timesteps=10_000, callback=create_callback(name, env))

#%% plot the training curves
#%% Plotting the training curves
h = 4

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

fig, ax = plt.subplots()
for name in ["sac", "td3", "ppo"]:
    df = pd.read_csv(f"./logs/wms/{name}_monitor.csv", skiprows=1)
    df['moving_avg'] = moving_average(df['r'], 20)
    ax.plot(df.index, df['moving_avg'], label=name.upper(), alpha=0.85)

ax.set_ylim(-200, 150)
ax.set_xlabel(r"Episode")
ax.set_ylabel(r"Mean reward per episode")
ax.set_title(r"\textbf{RL algorithms training curves}")
fig.set_size_inches(1.3*h * 1.618, h*0.75)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("logs/wms/training_curves.png", dpi=300)
plt.show()

#%%
# Load the best models
sac_best_model = SAC.load("./logs/wms/sac/best_model")
td3_best_model = TD3.load("./logs/wms/td3/best_model")
ppo_best_model = PPO.load("./logs/wms/ppo/best_model")
best_models = [sac_best_model, ppo_best_model, td3_best_model]

#%% perform the evaluation of SAC model
cultivate_env = CultivateEnv()
for model in best_models:
    x = []
    a = []
    rews = []
    obs, _ = cultivate_env.reset()
    x.append(obs)
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = action
        #action = cultivate_env.action_space.sample()
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
    ax[0].plot(x[:,1])
    ax[0].plot(x[:,2])
    ax[0].plot(x[:,3])
    ax[0].plot(x[:,4])
    ax[0].set_title("Depletion")
    ax[0].set_xlabel("Days since plantation")
    ax[0].set_ylabel(r"Depletion [\%]")

    ax[1].plot(x[:,6], label = f"mad = {x[:,2]}")
    ax[1].set_title("RAW")
    plt.tight_layout()
    plt.show()









