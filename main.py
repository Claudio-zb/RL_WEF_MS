# Main file to run the RL algorithms

from RL_algorithms.DQN import DQN
from environments.EMS_env import EMS_env
from json import load
import pandas as pd
import os
from datetime import datetime
import torch
from sklearn.externals import joblib

alg_name = "DQN"

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
        results, policy, scaler = rl_model.learn(3500)

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
        joblib.dump(scaler, full_path + r"\scaler.pkl")