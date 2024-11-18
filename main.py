from environments.EMS_env import MicrogridEnv, NormalizationWrapper, RuleBasedEMS

import numpy as np
from stable_baselines3 import TD3, PPO, SAC
import matplotlib.pyplot as plt

mg_env = MicrogridEnv()
mg_env = NormalizationWrapper(mg_env)

#%%
train = True
if train:
    model = SAC("MlpPolicy", mg_env, verbose=1, gradient_steps=2)
    model.learn(500_000)
    model.save("td3.pth")

#%%
model = SAC.load("td3.pth")

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
