from environments.EMS_env import MicrogridEnv, NormalizationWrapper
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt

from main2 import policy

mg_env = MicrogridEnv()

model = PPO("MlpPolicy", mg_env, verbose=1)
model.learn(200_000)


#%%
policy = lambda x: model.predict(x, deterministic=True)[0]
x = []
obs, _ = mg_env.reset()
x.append(obs)
for i in range(50):
    action = policy(obs)
    obs, rew, done, _, _ = mg_env.step(action)
    x.append(obs)
    print(obs)
    if done:
        break

#%%
x = np.array(x)
import matplotlib.pyplot as plt
plt.plot(x[:, 0], label='v_ref')
plt.plot(x[:, 1], label='v')
plt.legend()
plt.show()
