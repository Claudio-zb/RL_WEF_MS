# Main file to run the RL algorithms

from RL_algorithms.DQN import DQN
from environments.EMS_env import EMS_env
from json import load
import pandas as pd
import os
from datetime import datetime
import torch
import numpy as np

alg_name = "DQN"
"""
if __name__ == '__main__':
    try:
        with open('./training_options/dqn_quad_opts.json', 'r') as file:
            options = load(file)
        print(options)
        print("Options loaded")
    except Exception as e:
        print(e)

    penalties = [0, 1, 3]

    for penalty in penalties:

        environment = EMS_env()
        rl_model = DQN(environment, options=None)
        results, policy = rl_model.learn(3500)

        env_id = options["env_id"]
        # Get the current date and time
        current_datetime = datetime.now()
        date_time_str = current_datetime.strftime("%Y%m%d_%H%M%S")
        directory_name = f"{alg_name}_{date_time_str}"
        path = "training_results/ems"  # Change this to your desired path
        full_path = os.path.join(path, directory_name)
        os.makedirs(full_path)

        torch.save(policy, full_path + r"\policy.pt")
        df = pd.DataFrame(results)
        df.to_csv(full_path + r"\training_results.csv")
        """

def rwd_fun(s, a, s_next, e_penal=0):
    next_error = s[0] - s_next[4]
    current_error = s[0] - s[4]
    delta_error = np.abs(next_error) - np.abs(current_error)
    delta_Irr = (a[0] - s[3])/100
    delta_Q_p = (a[1] - s[5])/100

    if s[-1] != 143:
        reward = (np.exp(-1 * next_error ** 2 / (s[0] + 1e-5) ** 2 -
                  delta_Irr**2 - delta_Q_p**2) +
                  2*np.exp(-3 * next_error ** 2 / (s[0] + 1e-5) ** 2 -
                  2*delta_Irr**2 - 2*delta_Q_p**2) -
                  0.5 * np.sign(np.min([delta_error, 0])) -
                  3.0 * np.sign(np.max([delta_error, 0])))
        if s[10] != 0:
            reward = reward - e_penal
    else:
        reward = (np.exp(-1 * next_error ** 2 / (s[0] + 1e-5) ** 2 -
                  delta_Irr**2 - delta_Q_p**2) +
                  2*np.exp(-3 * next_error ** 2 / (s[0] + 1e-5) ** 2) -
                  2*delta_Irr**2 - 2*delta_Q_p**2)
        if s_next[10] != 0:
            reward = reward - e_penal

    return np.array([reward])

path1 = 'training_results/ems/DQN_20240309_063357'
path2 = 'training_results/ems/DQN_20240309_102806'
path3 = 'training_results/ems/DQN_20240309_142633'

training_results1 = pd.read_csv(path1 + '/training_results.csv')
training_results2 = pd.read_csv(path2 + '/training_results.csv')
training_results3 = pd.read_csv(path3 + '/training_results.csv')

model1 = torch.load(path1 + '/policy.pt')
model2 = torch.load(path2 + '/policy.pt')
model3 = torch.load(path3 + '/policy.pt')

rwd_fun_1 = lambda s, a, s_next: rwd_fun(s, a, s_next, 0)
rwd_fun_2 = lambda s, a, s_next: rwd_fun(s, a, s_next, 1)
rwd_fun_3 = lambda s, a, s_next: rwd_fun(s, a, s_next, 2)

mdl = EMS_env()
results = mdl.compare_policies([model1, model2, model3], 288, [rwd_fun_1, rwd_fun_2, rwd_fun_3])

