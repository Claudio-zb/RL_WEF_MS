import pandas as pd
from stable_baselines3.common.monitor import Monitor

from environments.EMS_env import MicrogridEnv, NormalizationWrapper, RuleBasedEMS

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

def create_wrapped_env():
    env = MicrogridEnv()
    env = NormalizationWrapper(env)
    env = Monitor(env, "./logs/")
    return env

# Create the vectorized environment
mg_vec_env = make_vec_env(create_wrapped_env, n_envs=4, seed=0)
mg_env = create_wrapped_env()

eval_callback = EvalCallback(mg_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)

#%%
train = True
if train:
    model = SAC("MlpPolicy", mg_vec_env, verbose=1, gradient_steps=-1)
    model.learn(total_timesteps=10_000, callback=eval_callback)
    model.save("sac_microgrid")

#%%

# Load the best model
best_model = SAC.load("./logs/best_model")

# Plotting the training curves
results_dir = './logs/'
df = pd.read_csv(results_dir + 'monitor.csv', skiprows=1)
df.plot(y='r', title='Training Curve')
plt.show()
#%%
from stable_baselines3.common import results_plotter

# Helper from the library
results_plotter.plot_results(
    [results_dir], 10_000, results_plotter.X_TIMESTEPS, "TD3 LunarLander"
)
#%%

def policy(observation):
    return model.predict(observation, deterministic=True)[0]

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

#%%
ems = RuleBasedEMS(1, policy)
ems.get_action([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
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
