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
r_wells = 0.4  # m
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
# %%

def drawdown2(s_k, k, q):
    #w = exp1(alpha/k) 
    return s_k + q/(4*np.pi*T)*(exp1(alpha/k+1) - exp1(alpha/(k)) )

#%%
n_coeffs = 20
coefficients = np.array([np.exp(-alpha*t)/(t+1) for t in range(0, n_coeffs)])
coefficients =  coefficients[::-1]
simu_steps = 144*10
Q = [0]*(n_coeffs) + [1]
response = np.zeros(simu_steps)
response2 = np.zeros(simu_steps) 
response3 = np.zeros(simu_steps)
for k in range(1,simu_steps):
    Q1 = np.array(Q)
    response2[k] = drawdown(k, np.diff(Q1)[-k:])
    response[k] = np.dot(coefficients, Q1[-n_coeffs-1:-1]) + Q1[-1]*a
    Q = Q + [random.uniform(0, 1)]
plt.plot(response[0:144]/5)
plt.plot(response2[0:144]/1000)

fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.title("Drawdown of the well")
plt.grid()

# %%

class WellModel(torch.nn.Module):
    def __init__(self):
        super(WellModel, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

    def forward(self, s_k, q_k):
        return self.alpha * s_k + self.beta * q_k
    
#%% Prepare training data
Q1_data = []
response2_data = []
for k in range(1, simu_steps):
    Q1_data.append(Q1[-n_coeffs-1:-1].copy())
    response2_data.append(response2[k])

Q1_tensor = torch.tensor(Q1_data, dtype=torch.float32)
response2_tensor = torch.tensor(response2_data, dtype=torch.float32).unsqueeze(1)

# Simple dataset and dataloader
class WellDataset(Dataset):
    def __init__(self, Q_data, s_data):
        self.Q_data = Q_data
        self.s_data = s_data

    def __len__(self):
        return len(self.Q_data)

    def __getitem__(self, idx):
        # s_k: previous drawdown, q_k: current Q
        s_k = torch.sum(self.Q_data[idx])  # or use a more meaningful feature
        q_k = self.Q_data[idx][-1]
        return torch.tensor([s_k], dtype=torch.float32), torch.tensor([q_k], dtype=torch.float32), self.s_data[idx]

dataset = WellDataset(Q1_tensor, response2_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = WellModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10000
for epoch in range(epochs):
    total_loss = 0
    for s_k, q_k, target in dataloader:
        optimizer.zero_grad()
        output = model(s_k, q_k)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
# %%

Q = [0]*(n_coeffs) + [1]
response = np.zeros(simu_steps)
response2 = np.zeros(simu_steps) 
for k in range(1,simu_steps):
    Q1 = np.array(Q)
    response2[k] = drawdown(k, np.diff(Q1)[-k:])
    with torch.no_grad():
        s_k = torch.sum(torch.tensor(Q1[-n_coeffs-1:-1], dtype=torch.float32))
        q_k = torch.tensor(Q1[-1], dtype=torch.float32)
        response[k] = model(s_k, q_k).item()
    Q = Q + [random.uniform(0, 1)]
plt.plot(response[0:144]/5)
plt.plot(response2[0:144]/1000)

fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.title("Drawdown of the well")
plt.grid()

# %%
