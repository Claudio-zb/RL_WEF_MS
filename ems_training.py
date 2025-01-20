import pandas as pd
from stable_baselines3.common.monitor import Monitor

from environments.EMS_env import MicrogridEnv, NormalizationWrapper, RuleBasedEMS
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))


def create_wrapped_env(log_file):
    env = MicrogridEnv()
    env = NormalizationWrapper(env)
    env = Monitor(env, log_file)
    return env


# Create the vectorized environment


def create_callback(alg_name, environment):
    return EvalCallback(environment, best_model_save_path=f'./logs/{alg_name}',
                        log_path=f'./logs/ems/{alg_name}', eval_freq=5000,
                        deterministic=True, render=False)


#%%
train = False
if train:
    sac_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/ems/sac_monitor.csv"), n_envs=4, seed=0)
    td3_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/ems/td3_monitor.csv"), n_envs=4, seed=0)
    ppo_vec_env = make_vec_env(lambda: create_wrapped_env("./logs/ems/ppo_monitor.csv"), n_envs=4, seed=0)

    sac_env = create_wrapped_env("./logs/ems/sac_monitor.csv")
    td3_env = create_wrapped_env("./logs/ems/td3_monitor.csv")
    ppo_env = create_wrapped_env("./logs/ems/ppo_monitor.csv")

    sac_model = SAC("MlpPolicy", sac_env, verbose=1, gradient_steps=-1)
    td3_model = TD3("MlpPolicy", td3_env, action_noise=action_noise, verbose=1, gradient_steps=-1)
    ppo_model = PPO("MlpPolicy", ppo_env, verbose=1, batch_size=128)

    models = [sac_model, td3_model, ppo_model]
    envs = [sac_env, td3_env, ppo_env]
    models_name = ["sac", "td3", "ppo"]
    for model, name, env in zip(models, models_name, envs):
        model.learn(total_timesteps=1_000_000, callback=create_callback(name, env))

#%% Plotting the training curves
h = 4

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

fig, ax = plt.subplots()
for name in ["sac", "td3", "ppo"]:
    df = pd.read_csv(f"./logs/ems/{name}_monitor.csv", skiprows=1)
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
plt.savefig("logs/ems/training_curves.png", dpi=300)
plt.show()

#%%
# Load the best models
sac_best_model = SAC.load("./logs/ems/sac/best_model")
td3_best_model = TD3.load("./logs/ems/td3/best_model")
ppo_best_model = PPO.load("./logs/ems/ppo/best_model")
best_models = [sac_best_model, ppo_best_model, td3_best_model]


#%%

def eval_policy(observation, rl_model):
    return rl_model.predict(observation, deterministic=True)[0]


mg_env = create_wrapped_env("./logs/ems/eval_monitor.csv")
for alg in best_models:
    policy = lambda observation: eval_policy(observation, alg)
    x = []
    a = []
    rews = []
    t_obs, obs = mg_env.reset()
    x.append(obs["state"])
    for i in range(2 * 144):
        action = policy(t_obs)
        a.append(action)
        t_obs, rew, done, _, obs = mg_env.step(action)
        x.append(obs["state"])
        rews.append(rew)
        print(obs)
        if done:
            break
    break

#%%
x = np.array(x)
a = np.array(a)
rews = np.array(rews)
t = np.linspace(0, 48, len(x) - 1)

#%%
t = np.linspace(0, 48, len(x) - 1)
plt.plot(t, x[:-1, 0], label='Required water')
plt.plot(t, x[:-1, 2], label='Irrigated water')
plt.xlabel("Time [hr]")
plt.ylabel("Water Volume [m3]")
plt.title("Irrigated water over two days")
fig = plt.gcf()
fig.set_size_inches(h * 1.618, h * .75)
plt.tight_layout()
plt.legend()
plt.savefig("logs/ems/irrigated_water.png", dpi=300)
plt.show()
#%%
plt.plot(t, a[:, 0], label='Tank recharge')
plt.plot(t, a[:, 1], label='Irrigation')
plt.xlabel("Time [hr]")
plt.ylabel("Flow rate [l/s]")
plt.ylim(0, 1.0)
plt.title("Valves flow rate over two days")
fig = plt.gcf()
fig.set_size_inches(h * 1.618, h * .75)
plt.legend()
plt.tight_layout()
plt.savefig("logs/ems/aperture.png", dpi=300)
plt.show()

#%%
print((x[143, 0] - x[143, 2])/x[143, 0])
print((x[287, 0] - x[287, 2])/x[287, 0])

print((0.03966951699351884+0.03457275545314583)/2)

#%%
plt.plot(t, x[:-1, 3], label="s")
plt.hlines(0.0, 0, 48, color='black', linestyles="--")
plt.hlines(1.0, 0, 48, color='black', linestyles="--")
plt.title("Aquifer drawdown")
plt.xlabel("Time [hr]")
plt.ylabel("Drawdown [m]")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(h * 1.618, h * .75)
plt.tight_layout()
plt.show()

#%%

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
plt.savefig("logs/ems/irrigated_water_drawdown.png", dpi=300)
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
plt.savefig("logs/ems/soc_aperture.png", dpi=300)
plt.show()

#%%

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(t, x[:-1, 0], label='Required water')
axs[0, 0].plot(t, x[:-1, 2], label='Irrigated water')
axs[0, 0].set_xlabel("Time [hr]")
axs[0, 0].set_ylabel("Water Volume [m3]")
axs[0, 0].legend()
axs[0, 0].set_title(r"\textbf{Required and irrigated water}")

axs[1, 0].plot(t, x[:-1, 3], label="s")
axs[1, 0].hlines(0.0, 0, 48, color='black', linestyles="--")
axs[1, 0].hlines(1.0, 0, 48, color='black', linestyles="--")
axs[1, 0].set_title(r"\textbf{Aquifer drawdown}")
axs[1, 0].set_xlabel("Time [hr]")
axs[1, 0].set_ylabel("Drawdown [m]")

axs[1, 1].plot(t, x[:-1, -3], label="SOC")
axs[1, 1].hlines(20, 0, 48, color='black', linestyles="--")
axs[1, 1].hlines(100, 0, 48, color='black', linestyles="--")
axs[1, 1].set_ylabel("State of energy [kWh]")
axs[1, 1].set_xlabel("Time [hr]")
axs[1, 1].set_title(r"\textbf{Batteries state of energy}")

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
plt.savefig("logs/ems/soc_aperture.png", dpi=300)
plt.show()

#%%
plt.plot(t, x[:-1, 1], label='v_tanks')
plt.xlabel("Time [hr]")
plt.legend()
plt.show()

#%%
plt.plot(rews)
plt.show()

#%%
plt.plot(t, x[:-1, -3], label="SOC")
plt.hlines(20, 0, 48, color='black', linestyles="--")
plt.hlines(100, 0, 48, color='black', linestyles="--")
plt.ylabel("State of charge [kWh]")
plt.xlabel("Time [hr]")
plt.title("State of charge over two days")
fig = plt.gcf()
fig.set_size_inches(h * 1.618, h * .75)
plt.tight_layout()
plt.savefig("logs/ems/soc.png", dpi=300)
plt.legend()

plt.show()

#%%
