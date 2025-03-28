#%% This script is going to be used for testing of the different environments
import pandas as pd
import time
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

action_noise = NormalActionNoise(mean=np.zeros(1), sigma= 2*np.ones(1))

def create_wrapped_env(log_file):
    env = CultivateEnv()
    env = Monitor(env, log_file)
    return env

# Create the vectorized environment
def create_callback(alg_name, environment):
    return EvalCallback(environment, best_model_save_path=f'./logs/wms/{alg_name}',
                 log_path=f'./logs/wms/{alg_name}', eval_freq=1_000,
                 deterministic=True, render=False)

train = False
if train:
    sac_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/wms/sac_monitor.csv"), n_envs=4, seed=0)
    #td3_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/wms/td3_monitor.csv"), n_envs=4, seed=0)
    #ppo_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/wms/ppo_monitor.csv"), n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)

    sac_env = create_wrapped_env("./logs/wms/sac_monitor.csv")
    #td3_env = create_wrapped_env("./logs/wms/td3_monitor.csv")
    #ppo_env = create_wrapped_env("./logs/wms/ppo_monitor.csv")

    #td3_model = TD3("MlpPolicy", td3_env, action_noise=action_noise, verbose=1, gradient_steps=-1, batch_size=256,
    #                policy_delay=3)
    #ppo_model = PPO("MlpPolicy", ppo_env, verbose=1, batch_size=256, normalize_advantage=True,
    #                use_sde=True, device="cpu", clip_range=0.18)
    sac_model = SAC("MlpPolicy", sac_env, verbose=1, gradient_steps=-1, batch_size=256, ent_coef=0.05)

    models = [sac_model] #[ppo_model, sac_model, td3_model]
    envs = [sac_env] #[ppo_env, sac_env, td3_env]
    models_name = ["sac"] #["ppo", "sac", "td3"]
    training_times = []
    for model, name, env in zip(models, models_name, envs):
        start_time = time.time()
        model.learn(total_timesteps=250_000, callback=create_callback(name, env))
        end_time = time.time()
        training_time = end_time - start_time
        training_times.append(training_time)
        print(f"Training {name} took {end_time - start_time} seconds")

#%% Plotting the training curves
h = 4

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def moving_std(data, window_size):
    return data.rolling(window=window_size).std()

fig, ax = plt.subplots()
window = 20
for name in ["sac", "td3", "ppo"]:
    df = pd.read_csv(f"./logs/wms/{name}_monitor.csv", skiprows=1)
    df['moving_avg'] = moving_average(df['r'], window)
    df['moving_std'] = moving_std(df['r'],window)
    ax.plot(df.index, df['moving_avg'], label=name.upper(), alpha=0.85)
    ax.fill_between(df.index, df['moving_avg'] - df['moving_std'], df['moving_avg'] + df['moving_std'], alpha=0.3)

ax.set_ylim(0, 80)
ax.set_xlabel(r"Episode", fontsize = 14)
ax.set_ylabel(r"Mean reward per episode", fontsize = 14)
#ax.set_title(r"\textbf{RL algorithms training curves}")
fig.set_size_inches(h * 2, h)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("logs/wms/wms_plot.png", dpi=300)
plt.show()

#%% compute the mean and std of the rewards

for name in ["sac"]: # , "td3", "ppo"]:
    df = pd.read_csv(f"./logs/wms/{name}_monitor.csv", skiprows=1)
    print(f"{name}: mean = {df['r'].mean()}, std = {df['r'].std()}")


#%%
# Load the best models
sac_best_model = SAC.load("./logs/wms/sac/best_model")
#td3_best_model = TD3.load("./logs/wms/td3/best_model")
#ppo_best_model = PPO.load("./logs/wms/ppo/best_model")
best_models = [sac_best_model]#[sac_best_model, ppo_best_model, td3_best_model]
models_name = ["sac"] #["sac", "ppo", "td3"]

#%% perform the evaluation of SAC model
cultivate_env = CultivateEnv()
for model, name in zip(best_models, models_name):
    x = []
    a = []
    rews = []
    obs, _ = cultivate_env.reset(options={"theta": 0.25})
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
    t = np.arange(1, len(x) + 1)
    #%%
    plt.plot(rews)
    plt.title("Episode Reward")
    plt.xlabel("Days since plantation")
    plt.ylabel("Reward")
    plt.savefig(f"logs/wms/sac/{name}_episode_reward.png", dpi=300)
    plt.show()

    #%%
    plt.step(t[:-1], a)
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h*.65)
    plt.title("Daily water requirement")
    plt.xlabel("Days since plantation")
    plt.ylabel("Irrigation depth [mm]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"logs/wms/sac/{name}_irrigation_depth.png", dpi=300)
    plt.show()

    #%%

    fig, ax = plt.subplots(1,1)
    ax.plot(t,x[:,0], label = "evp layer")
    ax.plot(t, x[:,1], label = "4th layer")
    ax.plot(t, x[:,2], label = "3rd layer")
    ax.plot(t, x[:,3], label = "2nd layer")
    ax.plot(t, x[:,4], label = "1st layer")
    ax.axhline(0.3, color = "gray", linestyle="--", alpha = .8)
    #ax[0].axhline((0.3-.13)*.5, color = "gray", linestyle="--", label = "Threshold", alpha = .8)
    y_min = 0.115
    y_max = 0.315
    midpoint = (y_min + y_max) / 2
    distance = y_max - midpoint
    ax.set_ylim(midpoint - distance, midpoint + distance)
    ax.text(10, 0.31, "Field capacity", color='gray', ha='center', va='top', fontsize = 14)
    ax.axhline(0.13, color="gray", linestyle="--", alpha=.8)
    ax.text(10, 0.12, 'Wilting point', color='gray', ha='center', va='bottom', fontsize = 14)
    fig.set_size_inches(h * 2, h)
    ax.legend(loc="upper right")
    #ax.set_title("Soil water content evolution")
    ax.set_xlabel("Days since plantation", fontsize = 17)
    ax.set_ylabel(r"Volumetric water content [$m^3/m^3$]", fontsize=17)
    plt.tight_layout()
    plt.savefig(f"training_results/wms/{name}_soil_water_content.png", dpi=300)
    plt.show()

#%%
    # Create subplots with one axis per soil layer
    num_layers = 5
    fig, axs = plt.subplots(num_layers, 1, sharex=True, figsize=(10, 2 * num_layers))

    # Plot each soil layer on a separate axis
    for i in range(num_layers):
        axs[i].plot(t, x[:, i], label=f'Layer {i + 1}')
        axs[i].axhline(0.3, color='gray', linestyle='--', alpha=0.8, label='Field capacity' if i == 0 else "")
        axs[i].axhline(0.13, color='gray', linestyle='--', alpha=0.8, label='Wilting point' if i == 0 else "")
        j = i
        #axs[i].set_ylabel(f"Layer {5-j} [$m^3/m^3$]")
        axs[i].legend(loc='upper right')
        axs[i].grid()

    # Set common labels
    axs[-1].set_xlabel("Days since plantation")
    fig.text(0.04, 0.5, 'Volumetric water content [$m^3/m^3$]', va='center', rotation='vertical')
    fig.suptitle("Soil Moisture Evolution", fontsize=14)
    fig.set_size_inches(h * 2, h * .3 * num_layers)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(f"logs/wms/sac/soil_moisture_evolution.png", dpi=300)
    plt.show()


    #%%
    fig, ax = plt.subplots(2,1)
    ax[0].step(t[:-1], x[1:,7])
    #ax[0].set_title("Water stress coeffient evolution")
    #ax[0].set_xlabel("Days since plantation")
    ax[0].set_ylabel(r"$K_s$")
    ax[0].grid()
    #fig.set_size_inches(h * 1.618, h)

    ax[1].step(t[:-1], x[1:,6])
    ax[1].invert_yaxis()
    ax[1].set_title("Root length evolution")
    ax[1].set_xlabel("Days since plantation")
    ax[1].set_ylabel(r"Root length [m]")
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(f"logs/wms/sac/{name}_Ks_and_root_length.png", dpi=300)
    plt.show()

    #%%
    fig, ax = plt.subplots(1,1)
    ax.step(t[:-1], x[1:,7])
    #ax.set_title("Water stress coeffient evolution")
    #ax[0].set_xlabel("Days since plantation")
    ax.set_ylabel(r"$K_s$")
    ax.grid()
    fig.set_size_inches(h * 2, h)

    plt.tight_layout()
    plt.savefig(f"training_results/wms/{name}_Ks.png", dpi=300)
    plt.show()

    print(f"{name} relative yield {cultivate_env.cultivates.crops[0].relative_yield}")
#%%

h = 4

for alg_name in ["sac", "td3", "ppo"]:

    data = np.load(f'logs/wms/{alg_name}/evaluations.npz')
    argmax = np.argmax(data['results'].mean(axis=1))
    print(f"Best evaluation for {alg_name} at timestep {data['timesteps'][argmax]} with mean return {data['results'].mean(axis=1)[argmax]} and std {data['results'].std(axis=1)[argmax]}")
    # Extract the arrays
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']

    # Plot the results
    plt.plot(timesteps, results.mean(axis=1), label=f'{alg_name.upper()}')
    plt.fill_between(timesteps, results.mean(axis=1) - results.std(axis=1), results.mean(axis=1) + results.std(axis=1), alpha=0.3)

fig = plt.gcf()
fig.set_size_inches((h * 2, h))
plt.xlabel('Timesteps', fontsize = 17)
plt.ylabel('Mean Return', fontsize = 17)
#plt.title('Evaluation Results Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("logs/wms/wms_eval_curve.png", dpi=300)
plt.show()








