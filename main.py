from environments.EMS_env import MicrogridEnv, NormalizationWrapper
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt

mg_env = MicrogridEnv()

#%%
train = True
if train:
    model = PPO("MlpPolicy", mg_env, verbose=1)
    model.learn(200_000)
    model.save("ppo.pth")


#%%
model = PPO.load("ppo.pth")
policy = lambda x: model.predict(x, deterministic=True)[0]
x = []
a = []
obs, _ = mg_env.reset()
x.append(obs)
for i in range(50):
    action = policy(obs)
    a.append(action)
    obs, rew, done, _, _ = mg_env.step(action)
    x.append(obs)
    print(obs)
    if done:
        break

#%%
import matplotlib.pyplot as plt
x = np.array(x)
a = np.array(a)

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
