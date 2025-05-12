#%%
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from environments.EMS_env import MicrogridEnv, NormalizationWrapper
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import time


isExperimental = True

location = "experimental_logs/ems/" if isExperimental else "logs/ems/"

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))

n_envs = 8
def create_wrapped_env(log_file=None):
    env = MicrogridEnv()
    env = NormalizationWrapper(env)
    if log_file is not None:
        env = Monitor(env, log_file)
    return env


# Create the vectorized environment


def create_callback(alg_name, environment):
    return EvalCallback(environment, 
                        best_model_save_path=f'{location + alg_name}',
                        log_path=f'{location + alg_name}', 
                        eval_freq=4*episode_length*16, 
                        deterministic=True, render=False)

episode_length = 144*3
train = False
if train:
    sac_vec_env = make_vec_env(lambda: create_wrapped_env(), n_envs=n_envs, seed=0)
    td3_vec_env = make_vec_env(lambda: create_wrapped_env(), n_envs=n_envs, seed=0)
    ppo_vec_env = make_vec_env(lambda: create_wrapped_env(), n_envs=n_envs, seed=0)

    sac_env = create_wrapped_env(f"{location}sac/sac_monitor.csv")
    td3_env = create_wrapped_env(f"{location}td3/td3_monitor.csv")
    ppo_env = create_wrapped_env(f"{location}ppo/ppo_monitor.csv")

    sac_model = SAC("MlpPolicy", sac_env, verbose=1, 
                    train_freq=10, batch_size=512)
    td3_model = TD3("MlpPolicy", td3_env, action_noise=action_noise, 
                    verbose=1, train_freq=10, batch_size=512, target_policy_noise=0.1)
    ppo_model = PPO("MlpPolicy", ppo_env, verbose=1, 
                    batch_size=episode_length*8, 
                    device="cpu", n_steps=episode_length*n_envs*2, 
                    n_epochs=12,
                    learning_rate=3e-4, ent_coef=0.0, clip_range=0.12)

    models = [sac_model, td3_model, ppo_model]
    envs = [sac_env, td3_env, ppo_env]
    models_name = ["sac", "td3", "ppo"]
    for model, name, env in zip(models, models_name, envs):
        start_time = time.time()
        model.learn(total_timesteps=2_000_000, callback=create_callback(name, env))
        end_time = time.time()

        print(f"Training {name} took {(end_time - start_time)} seconds")

for name in ["sac", "td3", "ppo"]:
    evaluation_data = np.load(f"{location + name}/evaluations.npz")

    # Extract the arrays
    timesteps = evaluation_data['timesteps']
    results = evaluation_data['results']
    ep_lengths = evaluation_data['ep_lengths']

    # Plot the results
    
    plt.plot(timesteps, results.mean(axis=1), label='Mean Return')
    plt.fill_between(timesteps, 
                 results.mean(axis=1) - results.std(axis=1), 
                 results.mean(axis=1) + results.std(axis=1), 
                 alpha=0.3, label='Std Dev')
plt.xlabel('Timesteps', fontsize=14)
plt.ylabel('Mean Return', fontsize=14)
plt.title('Evaluation Results Over Time', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{location}evaluation_plot.png", dpi=300)
plt.show()

#%% Plotting the training curves
h = 4

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def moving_std(data, window_size):
    return data.rolling(window=window_size).std()

window = 1

fig, ax = plt.subplots()
for name in ["sac"]:
    df = pd.read_csv(f"{location + name}/{name}_monitor.csv", skiprows=1)
    df['moving_avg'] = moving_average(df['r'], window)
    #df['moving_std'] = moving_std(df['r'], window)
    ax.plot(df.index, df['moving_avg'], label=name.upper())
    #ax.fill_between(df.index, df['moving_avg'] - df['moving_std'], df['moving_avg'] + df['moving_std'], alpha=0.3)

ax.set_ylim(-50, 250)
ax.set_xlabel(r"Episode", fontsize=14)
ax.set_ylabel(r"Mean reward per episode", fontsize=14)
#ax.set_title(r"\textbf{RL algorithms training curves}")
fig.set_size_inches(h * 2, h)
plt.legend(loc = "lower right")
plt.grid()
plt.tight_layout()
plt.savefig(f"{location}ems_plot.png", dpi=300)
plt.show()

#%% compute mean and std from training curves from episode 200

for name in ["sac", "td3", "ppo"]:
    df = pd.read_csv(f"{location + name}_monitor.csv", skiprows=1)
    print(f"{name}: mean = {df['r'].mean()}, std = {df['r'].std()}")

#%%
# Load the best models
sac_best_model = SAC.load(f"{location}sac/best_model")
td3_best_model = TD3.load(f"{location}td3/best_model")
ppo_best_model = PPO.load(f"{location}ppo/best_model")
best_models = [td3_best_model]  #[sac_best_model, ppo_best_model, td3_best_model]
models_name = ["td3"]  #["sac", "ppo", "td3"]

def eval_policy(observation, rl_model):
    return rl_model.predict(observation, deterministic=True)[0]

simu_days = 3
mg_env = create_wrapped_env()
for alg, name in zip(best_models, models_name):
    policy = lambda observation: eval_policy(observation, alg)
    x = []
    a = []
    rews = []
    t_obs, obs = mg_env.reset()
    x.append(obs["state"])
    for i in range(simu_days * 144):
        action = policy(t_obs)
        a.append(action)
        t_obs, rew, done, _, obs = mg_env.step(action)
        x.append(obs["state"])
        rews.append(rew)
        if done:
            break

    x = np.array(x)
    a = np.array(a)
    rews = np.array(rews)
    t = np.linspace(0, 24*simu_days, len(x) - 1)

    #%%
    h = 4
    t = np.linspace(0, 24*simu_days, len(x) - 1)
    plt.step(t, x[:-1, 0], label=r'$V_{req}^{day}$')
    plt.step(t, x[:-1, 2], label=r'$V_{irr}$')
    plt.xlabel("Time [hr]", fontsize = 14)
    plt.ylabel("Water Volume [m3]", fontsize = 14)
    #plt.title("Irrigated water over two days")
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h*.75)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(f"logs/ems/{name}_irrigated_water.png", dpi=300)
    plt.show()
    #%%
    plt.step(t, a[:, 0], label='Tank recharge')
    plt.step(t, a[:, 1], label='Irrigation')
    plt.xlabel("Time [hr]")
    plt.ylabel("Flow rate [l/s]")
    plt.ylim(0, 1.0)
    plt.title("Valves flow rate over two days")
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h * .75)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"logs/ems/{name}_aperture.png", dpi=300)
    plt.show()

    #%%
    print((x[143, 0] - x[143, 2])/x[143, 0])
    print((x[287, 0] - x[287, 2])/x[287, 0])

    #print((0.03966951699351884+0.03457275545314583)/2)

    #%%
    plt.step(t, x[1:, 3])
    plt.hlines(0.0, 0, 48, color='black', linestyles="--")
    plt.hlines(1.0, 0, 48, color='black', linestyles="--")
    plt.title("Aquifer drawdown")
    plt.xlabel("Time [hr]")
    plt.ylabel("Drawdown [m]")
    #plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h * .75)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"logs/ems/{name}_drawdown.png", dpi=300)
    plt.show()

    #%%
    plt.step(t, x[:-1, -3], label="SOC")
    plt.hlines(20, 0, 24*simu_days, color='black', linestyles="--")
    plt.hlines(100, 0, 24*simu_days, color='black', linestyles="--")
    plt.ylabel("State of charge [kWh]")
    plt.xlabel("Time [hr]")
    plt.legend()
    plt.title("State of charge over two days")
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h * .75)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"logs/ems/{name}_soc.png", dpi=300)
    plt.show()

    #%%

    plt.step(t, x[:-1, -2])
    plt.ylabel("Energy [kWh]")
    plt.xlabel("Time [hr]")
    plt.title("Grid Energy")
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h * .75)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"logs/ems/{name}_res_e.png", dpi=300)
    #plt.legend()

    plt.show()

    #%% Mulit axis figures

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, x[:-1, 0], label='Required water')
    axs[0].plot(t, x[:-1, 2], label='Irrigated water')
    axs[0].set_xlabel("Time [hr]")
    axs[0].set_ylabel("Water Volume [m3]")
    axs[0].set_title("Irrigated water over two days")

    axs[1].plot(t, x[:-1, 3], label="s")
    axs[1].hlines(0.0, 0, 48, color='black', linestyles="--")
    axs[1].hlines(1.0, 0, 48, color='black', linestyles="--")
    axs[1].set_title("Aquifer drawdown")
    axs[1].set_xlabel("Time [hr]")
    axs[1].set_ylabel("Drawdown [m]")

    plt.tight_layout()
    plt.savefig(f"logs/ems/{name}_irrigated_water_drawdown.png", dpi=300)
    plt.show()

    #%%
    fig, axs = plt.subplots(2, 1)
    axs[1].plot(t, x[:-1, -3], label="SOC")
    axs[1].hlines(20, 0, 48, color='black', linestyles="--")
    axs[1].hlines(100, 0, 48, color='black', linestyles="--")
    axs[1].set_ylabel("State of charge [kWh]")
    axs[1].set_xlabel("Time [hr]")
    axs[1].legend()
    axs[1].set_title("State of charge over two days")

    axs[0].plot(t, a[:, 0], label='Tank recharge')
    axs[0].plot(t, a[:, 1], label='Irrigation')
    axs[0].set_xlabel("Time [hr]")
    axs[0].set_ylabel("Flow rate [l/s]")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Valves flow rate over two days")
    plt.tight_layout()
    plt.savefig(f"logs/ems/{name}_soc_aperture.png", dpi=300)
    plt.show()

    #%%

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(t, x[:-1, 0], label='Required water')
    axs[0, 0].plot(t, x[:-1, 2], label='Irrigated water')
    axs[0, 0].set_xlabel("Time [hr]")
    axs[0, 0].set_ylabel("Water Volume [m3]")
    axs[0, 0].legend()
    axs[0, 0].set_title(r"\textbf{Required and irrigated water}")

    axs[1, 0].plot(t, x[:-1, 1], label="s")
    #axs[1, 0].hlines(0.0, 0, simu_days*24, color='black', linestyles="--")
    #axs[1, 0].hlines(1.0, 0, simu_days*24, color='black', linestyles="--")
    axs[1, 0].set_title(r"\textbf{Aquifer drawdown}")
    axs[1, 0].set_xlabel("Time [hr]")
    axs[1, 0].set_ylabel("Drawdown [m]")

    axs[1, 1].plot(t, np.clip(x[:-1, -2], -np.inf, 0), label="Energy")
    axs[1, 1].set_ylabel("Grid Energy [kWh]")
    axs[1, 1].set_xlabel("Time [hr]")
    axs[1, 1].set_title(r"\textbf{Batteries state of energy}")

    #axs[1, 1].plot(t, x[:-1, -3], label="SOC")
    #axs[1, 1].hlines(20, 0, simu_days*24, color='black', linestyles="--")
    #axs[1, 1].hlines(100, 0, simu_days*24, color='black', linestyles="--")
    #axs[1, 1].set_ylabel("State of energy [kWh]")
    #axs[1, 1].set_xlabel("Time [hr]")
    #axs[1, 1].set_title(r"\textbf{Batteries state of energy}")

    axs[0, 1].plot(t, a[:, 0], label='Tank recharge')
    axs[0, 1].plot(t, a[:, 1], label='Irrigation', alpha=.8)
    axs[0, 1].set_xlabel("Time [hr]")
    axs[0, 1].set_ylabel("Flow rate [l/s]")
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_title(r"\textbf{Valves flow rate}")
    axs[0, 1].legend()

    fig.set_size_inches(12, 5.5)
    fig.suptitle(r"\textbf{EMS two days operation}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"logs/ems/{name}_soc_aperture.png", dpi=300)
    plt.show()

    #%%
    plt.plot(t, x[:-1, 1], label='v_tanks')
    plt.xlabel("Time [hr]")
    plt.legend()
    plt.show()

    #%%
    plt.plot(t, x[:-1, -3], label="SOC")
    plt.hlines(20, 0, 24*simu_days, color='black', linestyles="--")
    plt.hlines(100, 0, 24*simu_days, color='black', linestyles="--")
    plt.ylabel("State of charge [kWh]")
    plt.xlabel("Time [hr]")
    plt.title("State of charge over two days")
    fig = plt.gcf()
    fig.set_size_inches(h * 1.618, h * .75)
    plt.tight_layout()
    plt.savefig(f"logs/ems/{name}_soc.png", dpi=300)
    plt.legend()

    plt.show()

    #%% 

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, rews)

    axs[1].plot(t, np.cumsum(rews))
    axs[1].set_xlabel("Time [hr]")
    
# %%
