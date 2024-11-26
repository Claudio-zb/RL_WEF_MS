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
ax.set_title(r"RL algorithms training curves")
fig.set_size_inches(h*1.618, h)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#%%
# Load the best models
sac_best_model = SAC.load("./logs/ems/sac/best_model")
td3_best_model = TD3.load("./logs/ems/td3/best_model")
ppo_best_model = PPO.load("./logs/ems/ppo/best_model")
best_models = [ppo_best_model, sac_best_model, td3_best_model]
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

#%%
plt.plot(a[:, 0], label='Q_p')
plt.plot(a[:, 1], label='Q_irr')
plt.legend()
plt.show()
#%%

plt.plot(x[:, 0], label='v_ref')
plt.plot(x[:, 2], label='v')
plt.legend()
plt.show()

#%%

plt.plot(x[:, 1], label='v_tanks')
plt.legend()
plt.show()

#%%
plt.plot(rews)
plt.show()

#%%
plt.plot(x[:,3], label = "s")
plt.show()

#%%
plt.plot(x[:, -3], label='SOC')
plt.legend()
plt.show()

#%%
