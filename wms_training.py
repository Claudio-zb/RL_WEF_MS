#%% This script is going to be used for testing of the different environments
import pandas as pd
import time
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from RL_algorithms.PPO2 import train as ppo_train

from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from stable_baselines3 import TD3, SAC

from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
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

models_dict = {"td3": TD3, "sac": SAC}
alg_names = ["td3", "sac"]

#train = True
#experimental = True
#path = "logs/wms/" if not experimental else "experimental_logs/wms/"

set_of_weights = np.array([[.9, .9, 1.2],
                           [.95, .95, 1.1],
                           [1.0, 1.0, 1.0], 
                           [1.05, 1.05, 0.9], 
                           [1.1, 1.1, 0.8]])

#%%
train = True
if train: 
    for idx, weights in enumerate(set_of_weights):

        path = f"logs/wms/weights_{idx}/"

        print(f"Training with weights {weights}")

        vec_envs = [make_vec_env(lambda: create_wrapped_env(weights=weights), n_envs=n_envs) for _ in alg_names]
        eval_envs = [create_wrapped_env(f"{path}{name}/{name}_monitor.csv", weights=weights) for name in alg_names]
        
        td3_model: BaseAlgorithm = TD3("MlpPolicy", vec_envs[0], action_noise=action_noise, 
                        verbose=1, batch_size=episode_length*4, train_freq=2, 
                        gradient_steps=2)
        
        sac_model: BaseAlgorithm = SAC("MlpPolicy", vec_envs[1], verbose=1, batch_size=episode_length*4, 
                        ent_coef=0.1, train_freq=2, gradient_steps=2)
        
        models:list[BaseAlgorithm] = [td3_model, sac_model]
        
        training_times = []
        for model, name, eval_env in zip(models, alg_names, eval_envs):
            start_time = time.time()
            model.learn(total_timesteps=60_000, callback=create_callback(name, eval_env))
            end_time = time.time()
            training_time = end_time - start_time
            training_times.append(training_time)
            print(f"Training {name} took {end_time - start_time} seconds")

        ### ppo training
        ppo_env = NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=weights)  
        ppo_train(ppo_env, 
                  max_training_timesteps=60_000,
                  update_freq=144*2,
                  eval_freq=144*2,
                  log_path=path+"ppo",
                  eval_env= EvalWMS(NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=weights)),
                  n_epochs=5)
#%%

# Load the training curves

folders = ["weights_0", 
               "weights_1", 
               "weights_2", 
               "weights_3",
               "weights_4"]

for index, folder in enumerate(folders):
    path = f"logs/wms/{folder}/"
    for alg_name in alg_names:
        val_env = TestWMS(NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=set_of_weights[index]))
        agent = models_dict[alg_name].load(f"{path}{alg_name}/best_model")
        obs, _ = val_env.reset()
        done = False
        actions = []
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            actions.append(action*20)
            obs, rewards, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated

        actions = np.array(actions)
        total_water = np.sum(actions)
        print(f"Total water for {alg_name} with weights {index} is {total_water} m3")
        print(f"Relative yield for {alg_name} with weights {index} is {obs[8]}")

#%%
# plot the training curves 

for index, folder in enumerate(folders):
    plt.figure(figsize=(8, 6))
    path = f"logs/wms/{folder}/"
    for alg_name in alg_names:
        evaluation_data = np.load(f"{path}{alg_name}/evaluations.npz")
        # Extract the arrays
        timesteps = evaluation_data['timesteps']
        results = evaluation_data['results']
        ep_lengths = evaluation_data['ep_lengths']

        # Plot the results
        plt.plot(timesteps, results.mean(axis=1), label=f'{alg_name.upper()}')
        plt.fill_between(timesteps, 
                        results.mean(axis=1) - results.std(axis=1), 
                        results.mean(axis=1) + results.std(axis=1), 
                        alpha=0.3)
    fig = plt.gcf()
    fig.set_size_inches(9, 4)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Mean Return', fontsize=14)
    plt.title('Evaluation Results Over Time', fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{path}_{alg_name}_evaluation_plot.png", dpi=300)
    plt.show()
            


#%%
# Load and plot evaluation data
plt.figure(figsize=(8, 6))
for alg_name in alg_names:
    evaluation_data = np.load(f"{path}{alg_name}/evaluations.npz")

    # Extract the arrays
    timesteps = evaluation_data['timesteps']
    results = evaluation_data['results']
    ep_lengths = evaluation_data['ep_lengths']

    # Plot the results
    plt.plot(timesteps, results.mean(axis=1), label=f'{alg_name.upper()}')
    plt.fill_between(timesteps, 
                    results.mean(axis=1) - results.std(axis=1), 
                    results.mean(axis=1) + results.std(axis=1), 
                    alpha=0.3)
fig = plt.gcf()
fig.set_size_inches(9, 4)
plt.xlabel('Timesteps', fontsize=14)
plt.ylabel('Mean Return', fontsize=14)
plt.title('Evaluation Results Over Time', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{path}evaluation_plot.png", dpi=300)
plt.show()


#%%
# Load the best models
sac_best_model = SAC.load(f"{path}sac/best_model")
td3_best_model = TD3.load(f"{path}td3/best_model")
#ppo_best_model = TD3.load(f"{path}td3/best_model")

best_models = [sac_best_model]#[sac_best_model, ppo_best_model, td3_best_model]
models_name = ["sac"] #["sac", "ppo", "td3"]

#%% perform the evaluation of PPO model
cultivate_env = TestWMS(NormalizedWMS(CultivateEnv(), reward_weigths=weights))
for model, name in zip(best_models, models_name):
    x = []
    a = []
    rews = []
    obs, _ = cultivate_env.reset()
    x.append(obs)
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        #action = cultivate_env.action_space.sample()
        a.append(action)
        obs, rewards, terminated, truncated, info = cultivate_env.step(action)
        rews.append(rewards)
        x.append(obs)
        done = terminated or truncated

    
    x = np.array(x)
    a = np.array(a)
    rews = np.array(rews)
    t = np.arange(1, len(x) + 1)
    #%%
    plt.plot(rews)
    plt.title("Episode Reward")
    plt.xlabel("Days since plantation")
    plt.ylabel("Reward")
    plt.savefig(f"{path}sac/{name}_episode_reward.png", dpi=300)
    plt.show()

    #%%
    h = 4
    plt.step(t[:-1], a)
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h*.65)
    plt.title("Daily water requirement")
    plt.xlabel("Time since plantation [days]")
    plt.ylabel(r"Irrigation volume [$m^3$]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/agro_geo_model/{name}_irrigation_depth.png", dpi=300)
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
    ax.set_xlabel("Time since plantation [Days]", fontsize = 17)
    ax.set_ylabel(r"Volumetric water content [$m^3/m^3$]", fontsize=17)
    plt.tight_layout()
    plt.savefig(f"plots/agro_geo_model/{name}_soil_water_content.png", dpi=300)
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
    axs[-1].set_xlabel("Time since plantation [days]")
    fig.text(0.04, 0.5, 'Volumetric water content [$m^3/m^3$]', va='center', rotation='vertical')
    fig.suptitle("Soil Moisture Evolution", fontsize=14)
    fig.set_size_inches(h * 2, h * .3 * num_layers)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    #plt.savefig(f"logs/agro_geo_model/soil_moisture_evolution.png", dpi=300)
    plt.show()


    #%%
    fig, ax = plt.subplots(2,1)
    ax[0].step(t[:-1], x[1:,7], where='post')
    #ax[0].set_title("Water stress coeffient evolution")
    #ax[0].set_xlabel("Days since plantation")
    ax[0].set_ylabel(r"$K_s$")
    ax[0].set_ylim(0, 1.1)
    ax[0].grid()
    #fig.set_size_inches(h * 1.618, h)

    ax[1].step(t[:-1], x[1:,8], where='post')

    #ax[1].invert_yaxis()
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
    plt.savefig(f"plots/agro_geo_model/{name}_Ks.png", dpi=300)
    plt.show()

    print(f"{name} relative yield {cultivate_env.cultivates.crops[0].relative_yield}")
#%%

h = 4

for alg_name in ["sac", "td3", "ppo"]:

    data = np.load(f'{path}{alg_name}/evaluations.npz')
    print(len(data['results']))
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

#%% 

crop_data = cultivate_env.cultivates.get_hist_data()['potato'][0]

#%%
# Initialize lists to store results for plotting
relative_yields = []
total_water_usage = []
weight_labels = []

# Loop through folders and collect data
for index, folder in enumerate(folders):
    path = f"logs/wms/{folder}/"
    weight_labels.append(f"{np.round(set_of_weights[index, -1], 2)}")
    triad_relative_yields = []
    triad_water_usage = []
    
    for alg_name in alg_names:
        val_env = TestWMS(NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=set_of_weights[index]))
        agent = models_dict[alg_name].load(f"{path}{alg_name}/best_model")
        obs, _ = val_env.reset()
        done = False
        actions = []
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            actions.append(action * 20)
            obs, rewards, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated

        actions = np.array(actions)
        total_water = np.sum(actions)
        triad_water_usage.append(total_water)
        triad_relative_yields.append(obs[8])
        print(f"Total water for {alg_name} with weights {index} is {total_water} m3")
        print(f"Relative yield for {alg_name} with weights {index} is {obs[8]}")
    
    total_water_usage.append(triad_water_usage)
    relative_yields.append(triad_relative_yields)

# Create bar plots
import matplotlib.pyplot as plt
import numpy as np

#%% Plot relative yields
x = np.arange(len(weight_labels))  # Label locations
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
for i, alg_name in enumerate(alg_names):
    ax.bar(x + i * width, [ry[i] for ry in relative_yields], width, label=alg_name.upper())

ax.set_xlabel("Reward Function Weights")
ax.set_ylabel("Relative Yield")
ax.set_title("Relative Yield Comparison by Reward Function Weights")
ax.set_xticks(x + width)
ax.set_xticklabels([fr"$\alpha = {weight}$" for weight in weight_labels], ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("logs/wms/relative_yield_comparison.png", dpi=300)
plt.show()

# Plot water usage
fig, ax = plt.subplots(figsize=(10, 6))
for i, alg_name in enumerate(alg_names):
    ax.bar(x + i * width, [wu[i] for wu in total_water_usage], width, label=alg_name.upper())

ax.set_xlabel("Reward Function Weights")
ax.set_ylabel("Total Water Usage (m³)")
ax.set_title("Water Usage Comparison by Reward Function Weights")
ax.set_xticks(x + width)
ax.set_xticklabels([fr"$\alpha = {weight}$" for weight in weight_labels], ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("logs/wms/water_usage_comparison.png", dpi=300)
plt.show()
