import tkinter as tk
from environments.custom_env import Custom_env
from RL_algorithms.RL_algorithm import RL_algorithm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from time import sleep, perf_counter
import threading
import queue

import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    '''Class to train the RL agent and plot the results in real time using tkinter'''
    def __init__(self, env:Custom_env, alg:RL_algorithm):
        
        self.env = env
        self.alg = alg

        self.root = tk.Tk()
        self.root.title("RL Trainer")

        # Create a figure and axes
        self.fig = Figure()
        self.axs = [self.fig.add_subplot(3, 1, i+1) for i in range(3)]
        self.lines = [ax.plot([], [])[0] for ax in self.axs]

        # Create a canvas and add the figure to it
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        #self.canvas.get_tk_widget().pack()
        self.canvas.draw()

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

        # strcutures with the historic statistics
        self.n_eps = 200
        self.stats = []
        self.q = queue.Queue()
        self.mean_rewards = []
        self.std_rewards = []
        self.values_q_net = []
        self.values_target_net = []

    def update_plot(self) -> None:        
        '''Update the plot with the new data from the queue'''
        try:
            while True:
                stats = self.q.get_nowait()
                self.mean_rewards.append(stats[0])
                self.std_rewards.append(stats[1])
                self.values_q_net.append(stats[2])
                self.values_target_net.append(stats[3])

        except queue.Empty:
            pass
        for idx, ax in enumerate(self.axs):
            self.lines[idx].set_data(range(len(self.mean_rewards)), [self.mean_rewards, self.std_rewards, self.values_q_net, self.values_target_net][idx])
            ax.relim()
            ax.autoscale_view()
            
        self.canvas.draw_idle()
        self.job = self.root.after(100, self.update_plot)

    def start_plot(self) -> None:
        '''Start the plot update process in a separate thread'''
        def target():
            while True:
                if self.i_episode % 10 == 0:
                    self.env.show_sample(self.alg.policy_net, self.alg.scaler)
                if self.i_episode == 100:
                    break
                self.q.put(self.alg.one_ep_training(self.i_episode))
                self.i_episode += 1
                
        thread = threading.Thread(target=target)
        thread.start()
        self.job = self.root.after(100, self.update_plot)  # update every 100 ms

    def pause_plot(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None

    def run(self):
        # Run the application
        self.root.mainloop()