# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from RL_algorithms.PPO import PPO
from RL_algorithms.DQN import DQN
from environments.Quad_env import Quad_env
from json import load
import pandas as pd
import os
from datetime import datetime
import torch

alg_name = "DQN"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        with open('./training_options/dqn_quad_opts.json', 'r') as file:
            options = load(file)
        print(options)
        print("Options loaded")
    except Exception as e:
        print(e)

    # Get the current date and time
    current_datetime = datetime.now()
    date_time_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    directory_name = f"{alg_name}_{date_time_str}"
    path = "./training_results/quad_env"  # Change this to your desired path
    full_path = os.path.join(path, directory_name)
    os.makedirs(full_path)

    env_id = options["env_id"]

    environment = Quad_env()
    rl_model = DQN(environment, options=None)
    results, policy = rl_model.learn(1000)
    torch.save(policy, full_path + r"\policy.pt")
    df = pd.DataFrame(results)
    df.to_csv(full_path + r"\training_results.csv")
