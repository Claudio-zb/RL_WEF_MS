#%%
from environments.utils.predict_utils import Predictor, EarlyStopping
from environments.Data.EMS.EMS_constants import *
import numpy as np
from environments.utils.funcionesEMS import *
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
from scipy.special import exp1

alpha = r_wells**2*S/(4*T*600) # m^2/s
def drawdown(k: int, dQ: Iterable):  # drawdown of the well
    """
    Computes the drawdown of the well according the theis equation
    :param k: time instant
    :param dQ: delta flow rate [m3/s]
    :return: drawdown
    """
    assert k == len(dQ), "The length of dQ should match the number of temporal k points"
    
    l = np.arange(1, k + 1)
    arg = (r_wells ** 2 * S) / (4 * T * (k - l + 1) * 600)
    sum_ = 0.0
    for i in range(k):
        sum_ += dQ[i] * exp1(arg[i])
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val

a = exp1(alpha)


#%%
n_coeffs = 20
coefficients = np.array([np.exp(-alpha*t)/(t+1) for t in range(0, n_coeffs)])
coefficients =  coefficients[::-1]
simu_steps = 144*10
Q = [0]*(n_coeffs) + [1]
response = np.zeros(simu_steps)
response2 = np.zeros(simu_steps) 
for k in range(1,simu_steps):
    Q1 = np.array(Q)
    response2[k] = drawdown(k, np.diff(Q1)[-k:])
    response[k] = np.dot(coefficients, Q1[-n_coeffs-1:-1]) + Q1[-1]*a
    Q = Q + [random.uniform(0, 1)]
plt.plot(response/5)
plt.plot(response2/1000)

fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.title("Drawdown of the well")
plt.grid()

# %%
