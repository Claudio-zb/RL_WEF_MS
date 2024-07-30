from typing import Any, Union, Tuple, Callable, List, Iterable
from environments.EMS_constants import *

import copy
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import torch

from utils_functions.funcionesEMS import *
from gymnasium import spaces
from environments.custom_env import Custom_env

class WMS_env(Custom_env):
    
    def __init__(self, action_low, action_high, continuous: bool = False):
        """First implementation made for only one crop"""

        super().__init__(action_low, action_high, continuous)

        self.T = 1
        self.S = 1
        self.s = 1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[Any | dict[str, Any]]:
        
        self.t = 0
        self.s = 1

        pass
    
    def step(self, action: np.ndarray) -> tuple[ndarray, ndarray, bool, bool, dict[str, Any]]:
        pass
    