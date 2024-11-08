from environments.EMS_env import MicrogridEnv, NormalizationWrapper
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt

env = MicrogridEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(10000)
actions1 = []
actions2 = []
h = 288
responses=np.zeros((env.observation_space.low.shape[-1], 288))
actions = np.zeros((2,288))
rewards = np.zeros(288)
s, _ = env.reset()
for i in range(h):
    if i == 143:
        print("a")
    action = model.predict(s)[0]
    s, rew, d, w, info = env.step(action)
    responses[:,i] = s
    rewards[i] = rew
    actions[:,i] = action
    actions1.append(action[0])
    actions2.append(action[1])

plt.plot(responses[0,:-1], label='')
plt.plot(responses[1,:-1], label='')
