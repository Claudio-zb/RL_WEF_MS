from environments.basic_env import basic_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#%%
# The noise objects for TD3
env = basic_env()
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10_000, log_interval=10)
model.save("simple_env")

s, _ = env.reset()
responses=[]
rewards = []
actions = []
for i in range(100):
    action = model.predict(s)[0]
    tuple = env.step(action)
    s = tuple[0]
    responses.append(s[0])
    rewards.append(tuple[1])
    actions.append(action[0])
    print(tuple)
    print("================")
    print(action)
