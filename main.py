# Main file to run the RL algorithms
#%%
from RL_algorithms.DQN import DQN
from RL_algorithms.TD3 import TD3
from environments.EMS_env import EMS_env, EMS_env2
from environments.CEMS_env import ContinousEMSEnv
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
from environments.Quad_env import Quad_env

from environments.GH_env import*
from environments.CEMS_env import*

env = Quad_env()#ContinousEMSEnv()
rl_algorithm = TD3(env)

trainer = TrainerUI(env, rl_algorithm)

trainer.run()