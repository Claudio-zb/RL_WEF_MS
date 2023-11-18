
from environments.EMS_env import EMS_env
import numpy as np
ems_env = EMS_env()
ems_env.reset()
ems_env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

