#%% This script is going to be used for testing of the different environments
import pandas as pd
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from RL_algorithms.PPO2 import PPO, ActorCritic, train as ppo_train

from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3 import TD3, SAC

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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
                        eval_freq=episode_length*2,
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

set_of_weights2 = np.array([[0.90, 1.2, 0.90],
                            [0.8, 1.4, 0.8],
                            [0.7, 1.6, 0.7],
                            [0.6, 1.8, 0.6],])

set_of_weights = np.concatenate((set_of_weights, set_of_weights2), axis=0)

indexes = [0,1,2,3,4] #[5, 6, 7, 8]

#%%
train = False
if train: 
    for idx, weights in zip(indexes, set_of_weights2):

        path = f"logs/wms/weights_{idx}/"

        print(f"Training with weights {weights}")

        vec_envs = [make_vec_env(lambda: create_wrapped_env(weights=weights), n_envs=n_envs) for _ in alg_names]
        eval_envs = [create_eval_env(f"{path}{name}/{name}_monitor.csv", weights=weights) for name in alg_names]
        
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
#%%  Compute statistics for the trained models

folders = ["weights_0", 
            "weights_1", 
            "weights_2", 
            "weights_3",
            "weights_4",
            "weights_5", 
            "weights_6", 
            "weights_7", 
            "weights_8"]

weights_dict = {0: [0.9, 1.2, 0.9],
                1: [0.95, 1.1, 0.95],
                2: [1.0, 1.0, 1.0],
                3: [1.05, 0.9, 1.05],
                4: [1.1, 0.8, 1.1],
                5: [0.90, 1.2, 0.90],
                6: [0.8, 1.4, 0.8],
                7: [0.7, 1.6, 0.7],
                8: [0.6, 1.8, 0.6]}

alg_names = ["td3", "sac", "ppo"]

# Initialize lists to store results for plotting

weight_labels = []
alg_yields = []
alg_water_usage = []

for index, folder in enumerate(folders):
    path = f"logs/wms/{folder}/"
    relative_yields = []
    total_water_usage = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg_name in alg_names:
        val_env = TestWMS(NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=weights_dict[index]))
        if alg_name == "ppo":
            agent = ActorCritic(state_dim=val_env.observation_space.shape[0],
                                action_dim=val_env.action_space.shape[0],
                                has_continuous_action_space=True,
                                action_std_init=0.6)
            agent.load_state_dict(torch.load(f"{path}{alg_name}/ppo_CultivateEnv_best.pth"))
        else:
            agent = models_dict[alg_name].load(f"{path}{alg_name}/best_model")
        obs, _ = val_env.reset()
        done = False
        actions = []
        observations = []
        while not done:
            if alg_name == "ppo":
                action = agent.get_action(obs)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            actions.append(action*20)
            obs, rewards, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated
            observations.append(obs)
        
        observations = np.array(observations)
        actions = np.array(actions)
        total_water = np.sum(actions)
        relative_yield = np.exp(np.mean(np.log(observations[:,7] + 1e-10)))

        relative_yields.append(relative_yield)
        total_water_usage.append(total_water)

        print(f"Total water for {alg_name} with weights {index} is {total_water} m3")
        print(f"Relative yield for {alg_name} with weights {index} is {relative_yield}")
        ax.plot(observations[:, 7], label=f"{alg_name} K_s")
    ax.legend()
    plt.show()

    alg_yields.append(relative_yields)
    alg_water_usage.append(total_water_usage)
    weight_labels.append(weights_dict[index])

#%% plot evaluation curves

for index, folder in enumerate(folders):
    path = f"logs/wms/{folder}/"
    fig, ax = plt.subplots(figsize=(8, 4))
    for alg_name in alg_names:
        if alg_name == "ppo":
            data = pd.read_csv(f"{path}{alg_name}/evaluations.csv")
            ax.plot(data["timestep"], data["reward"], label=alg_name.upper())
            ax.fill_between(data["timestep"], data["reward"] - data["std"], data["reward"] + data["std"], alpha=0.2)
        else:
            data = np.load(f"{path}{alg_name}/evaluations.npz") 
            avg_rews = data["results"].mean(axis=1)
            std_rews = data["results"].std(axis=1)
            ax.plot(data["timesteps"], avg_rews, label=alg_name.upper())
            ax.fill_between(data["timesteps"], avg_rews - std_rews, avg_rews + std_rews, alpha=0.2)
    plt.legend()
    plt.grid()
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.tight_layout()
    plt.savefig(f"{path}evaluation_curves.png", dpi=300)
    plt.show()


#%% Plot relative yields
x = np.arange(len(weight_labels))  # Label locations
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
for i, alg_name in enumerate(alg_names):
    ax.bar(x + i * width, [ry[i] for ry in alg_yields], width, label=alg_name.upper())

#ax.set_xlabel("Reward Function Weights")
ax.set_ylabel("Relative Yield")
ax.set_title("Relative Yield Comparison by Reward Function Weights")
ax.set_xticks(x + width)
#ax.set_xticklabels([fr"$\lambda_1 = {weights[0]}$ \\ $\lambda_2 = {weights[1]}$ \\ $\lambda_2 = {weights[2]}$" for weights in weight_labels], 
#                   )
ax.set_xticklabels([fr"$\lambda_1 = {weights[0]}$ \\ $\lambda_2 = {weights[1]}$ \\ $\lambda_2 = {weights[2]}$" for weights in weight_labels], 
                   )
ax.legend()
plt.tight_layout()
plt.savefig("logs/wms/relative_yield_comparison.png", dpi=300)
plt.show()

# Plot water usage
fig, ax = plt.subplots(figsize=(10, 6))
for i, alg_name in enumerate(alg_names):
    ax.bar(x + i * width, [wu[i] for wu in alg_water_usage], width, label=alg_name.upper())

#ax.set_xlabel("Reward Function Weights")
ax.set_ylabel("Total Water Usage (m³)")
ax.set_title("Water Usage Comparison by Reward Function Weights")
ax.set_xticks(x + width)
ax.set_xticklabels([fr"$\lambda_1 = {weights[0]}$ \\ $\lambda_2 = {weights[1]}$ \\ $\lambda_2 = {weights[2]}$" for weights in weight_labels])
ax.legend()
plt.tight_layout()
plt.savefig("logs/wms/water_usage_comparison.png", dpi=300)
plt.show()

# %%
