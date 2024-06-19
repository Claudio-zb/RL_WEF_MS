import tkinter as tk
from environments.custom_env import CustomEnv
from RL_algorithms.RL_algorithm import RL_algorithm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from time import sleep, perf_counter
import threading
import queue
import os
from datetime import datetime
import torch
import pandas as pd
import joblib

import numpy as np
import matplotlib.pyplot as plt

class TrainerUI:
    '''Class to train the RL agent and plot the results in real time using tkinter'''
    def __init__(self, env:CustomEnv, alg:RL_algorithm):
        
        self.env = env
        self.alg = alg

        self.root = tk.Tk()
        self.root.title("RL Trainer")

        top = tk.Toplevel(self.root)
        top.title("Environment Sample")


        # Create a figure and axes
        self.fig = Figure()
        self.axs = [self.fig.add_subplot(3, 1, i+1) for i in range(3)]
        self.lines = []
        # lines for the mean rewards 
        self.lines.append(self.axs[0].plot([], [], label="Mean rewards")[0])
        self.lines.append(self.axs[0].plot([], [], label="Mean rewards (last 10 episodes)")[0])

        # lines for the Q values of the networks
        self.lines.append(self.axs[1].plot([], [], label="Q policy")[0])
        self.lines.append(self.axs[1].plot([], [], label="Q target")[0])

        # lines for the epsilon values
        self.lines.append(self.axs[2].plot([], [], label="Epsilon")[0])

        # Create a canvas and add the figure to it
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        self.env_fig = self.env.get_figure()

        # Create a canvas for the environment sample
        self.env_canvas = FigureCanvasTkAgg(self.env_fig, master=top)
        self.env_canvas.draw()
        self.env_canvas.get_tk_widget().pack()


        # Add the canvas to the window
        self.canvas.get_tk_widget().pack()

        # Create a button to start the plot update
        start_button = tk.Button(self.root, text="Start Training", command=self.start_plot)
        start_button.pack()

        # Create a button to pause the plot update
        pause_button = tk.Button(self.root, text="Pause Training", command=self.pause_plot)
        pause_button.pack()

        # Initialize the job object
        self.job = None
        self.i_episode = 0
        self.stop_training = False

        # structures with the historic statistics
        self.n_eps = 200
        self.stats = []
        self.q = queue.Queue()
        self.mean_rewards = []
        self.std_rewards = []
        self.values_q_net = []
        self.values_target_net = []
        self.epsilon = []

        self.full_path = ""

    def update_plot(self) -> None:        
        '''Update the plot with the new data from the queue'''
        try:
            while True:
                self.update_stats()
        except queue.Empty:
            pass
        
        # Plot the mean rewards
        self.lines[0].set_data(range(len(self.mean_rewards)), self.mean_rewards)
        # last 10 episodes mean rewards
        if len(self.mean_rewards) >= 10:
            avg_rew = np.convolve(self.mean_rewards, np.ones(10)/10, mode='valid')
            self.lines[1].set_data(range(10, len(self.mean_rewards)+1), avg_rew)

        # plot the Q values of both networks  
        self.lines[2].set_data(range(len(self.values_q_net)), self.values_q_net)
        self.lines[3].set_data(range(len(self.values_target_net)), self.values_target_net) 

        # plot the epsilon values

        self.lines[4].set_data(range(len(self.epsilon)), self.epsilon)
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()
        self.canvas.flush_events()
        self.job = self.root.after(100, self.update_plot)

    def start_plot(self) -> None:
        '''Start the plot update process in a separate thread'''
        def target():
            
            current_datetime = datetime.now()
            date_time_str = current_datetime.strftime("%Y%m%d_%H%M%S")
            directory_name = f"{self.alg.__class__.__name__}_{date_time_str}"
            path = "training_results/ems"  # Change this to your desired path
            self.full_path = os.path.join(path, directory_name)
            os.makedirs(self.full_path)
            
            for ax in self.axs:
                ax.legend()
            while True:
                if self.i_episode % 10 == 0: # Show a sample of the environment every 10 episodes
                    self.env.show_sample(self.alg.get_policy())
                    for ax in self.env_fig.get_axes():
                        ax.relim()
                        ax.autoscale_view()
                    self.env_canvas.draw_idle()
                    self.env_canvas.flush_events()

                if self.i_episode % 50 == 0:
                    self.save_model(self.i_episode)
                if self.i_episode == 5000 or self.stop_training:
                    self.save_results()
                    break
                self.q.put(self.alg.one_ep_training(self.i_episode))
                self.i_episode += 1
                
        thread = threading.Thread(target=target)
        thread.start()
        self.job = self.root.after(100, self.update_plot)  # update every 100 ms

    def pause_plot(self):
        if self.job is not None:
            self.stop_training = True
            self.root.after_cancel(self.job)
            self.job = None
        self.save_results()

    def run(self):
        # Run the application
        self.root.mainloop()
    
    def save_results(self):
        "Saves the results of the training in a csv file and the final model in a .pt file"

        stats = self.stats_to_df()
        torch.save(self.alg.get_policy(), self.full_path + f"\\policy_final.pt")
        stats.to_csv(self.full_path + r"\training_results.csv")

    def save_model(self, i_episode:int):
        "Saves the current policy model in a .pt file"
        torch.save(self.alg.get_policy(), self.full_path + f"\\policy_ep_{i_episode}.pt")


    def update_stats(self):
        """Update the statistics with the new data from the queue"""

        stats = self.q.get_nowait()
        self.mean_rewards.append(stats[0])
        self.std_rewards.append(stats[1])
        self.epsilon.append(stats[2])
        self.values_q_net.append(stats[3])
        self.values_target_net.append(stats[4])

    def stats_to_df(self) -> pd.DataFrame:
        """Convert the statistics to a pandas DataFrame"""
     
        df = pd.DataFrame({"mean_rewards": self.mean_rewards, 
                           "std_rewards": self.std_rewards, 
                           "values_q_net": self.values_q_net, 
                           "values_target_net": self.values_target_net})
        return df
    

class Trainer():
    """Class to train a RL agent over a custom environment"""

    def __init__(self, env: CustomEnv, rl_algorithm: RL_algorithm):
        self.env = env
        self.rl_algorithm = rl_algorithm

    def train(self, n_eps: int = 200):
        """Train the RL agent over the environment for n_eps episodes"""
        stats = []
        for i in range(n_eps):
            stats.append(self.rl_algorithm.one_ep_training(i))
        return stats

