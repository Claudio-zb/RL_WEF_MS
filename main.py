# Main file to run the RL algorithms

from RL_algorithms.DQN import DQN
from environments.EMS_env import EMS_env
from json import load
import pandas as pd
import os
from datetime import datetime
import torch
import joblib
from utils_functions.funcionesEMS import follow_ref_rew_1, follow_ref_rew_2
from matplotlib.animation import FuncAnimation

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from time import sleep, perf_counter
import threading

# Create the main window
root = tk.Tk()
root.title("My Tkinter GUI")

# Create a label
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

# Create a Matplotlib figure
env = EMS_env(rwd_function=follow_ref_rew_2)
dqn_alg = DQN(env, options=None)
fig = dqn_alg.get_training_fig()


# Create a canvas and add the figure to it
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

# Add the canvas to the window
canvas.get_tk_widget().pack()

# Create a button to start the DQN training
def start_button_callback():
    print("Starting DQN training...")
    thread = threading.Thread(target=dqn_alg.learn(2000))
    thread.start()

start_button = tk.Button(root, text="Start DQN training", command=start_button_callback)
start_button.pack()

# Run the application
root.mainloop()
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

    penalties = [follow_ref_rew_2, follow_ref_rew_1]

    for penalty in penalties:

        environment = EMS_env(rwd_function=penalty)
        rl_model = DQN(environment, options=None)
        results, policy, scaler = rl_model.learn(2000)

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
        joblib.dump(scaler, full_path + r"\scaler.pkl")"""