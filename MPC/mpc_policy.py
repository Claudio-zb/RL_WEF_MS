#%%
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from environments.WMS_policy import IrrigationPolicy
from environments.WMS_env import Cultivates
import pyswarms as ps

#%%
class MPCPolicy():
    def __init__(self, model: Cultivates, horizon: int = 10):
        self.model = Cultivates()
        self.horizon = horizon
    
    def get_action(self, obs):
        # Set-up hyperparameters
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}

        # Call instance of PSO
        optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)

        # Perform optimization
        cost, pos = optimizer.optimize(fx.sphere, iters=1000)
        return act
    
    def cost_function(self):
        return

#%%
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}

# Call instance of PSO
optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)

# Perform optimization
cost, pos = optimizer.optimize(fx.sphere, iters=1000)