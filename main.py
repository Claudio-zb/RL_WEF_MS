# Main file to run the RL algorithms
#%%
from RL_algorithms.DQN import DQN
from environments.EMS_env import EMS_env
from environments.GH_env import GH_env
from json import load
import pandas as pd
from datetime import datetime
from utils_functions.funcionesEMS import follow_ref_rew_1, follow_ref_rew_2
from matplotlib.animation import FuncAnimation
 
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from time import sleep, perf_counter
from RL_algorithms.Trainer import TrainerUI
import numpy as np 

from environments.GH_env import*

gh_env = GH_env()

x = gh_env.reset()
x1 = gh_env.step(np.array([0.0]))


"""
#%%
env = EMS_env()
dqn = DQN(env)

trainer = TrainerUI(env, dqn)

trainer.run()
"""