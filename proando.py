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
 
def drawdown(k: int, Q: Iterable):  # drawdown of the well
    """
    Computes the drawdown of the well according the theis equation
    :param k: time instant
    :param dQ: delta flow rate [l/s2]
    :return: drawdown
    """

    assert k == len(Q), "The length of dQ should match the number of temporal k points"
    
    l = np.arange(1, k + 1)
    arg = (r_wells ** 2 * S) / (4 * T * (k - l + 1) * 600) 
    arg0 = (r_wells ** 2 * S) / (4 * T * k * 600) if k > 0 else np.inf
    arg = np.concatenate(([arg0], arg))
    dW = np.diff(exp1(arg))
    sum_ = 0.0
    for i in range(k):
        sum_ += Q[i] * dW[i] * 1e-3
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val

a = exp1(alpha)

class AprbsHandler:
    def __init__(self, max_amplitude=1.0, min_amplitude=0.0, t0=1000):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.t0 = t0
        self.current_amplitude = np.random.uniform(min_amplitude, max_amplitude)
        self.current_time = 0

    def reset(self):
        self.current_time = 0
        self.current_amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)

    def __call__(self):
        self.current_time += 1
        if self.current_time <= self.t0:
            return self.current_amplitude
        else:
            self.reset()
            return self.current_amplitude
    def to_zero(self):
        """Set the current amplitude to zero."""
        self.current_amplitude = 0.0
        self.current_time = 0
        
#%%

aprbs = AprbsHandler(max_amplitude=1.0, min_amplitude=0.0, t0=20)
u = [aprbs() for _ in range(100)]
plt.plot(u)

#%% Prepare training data
simu_steps = 144*100
aprbs = AprbsHandler(max_amplitude=1.0, min_amplitude=0.0, t0=20)
Q1_data = np.zeros(simu_steps)
response_data = np.zeros_like(Q1_data)

for k in range(1, simu_steps):
    Q1_data[k] = aprbs()
    response_data[k] = drawdown(k, Q1_data[:k])
    if k%144 == 0:
        aprbs.to_zero()  # Reset the APRBS handler every 144 steps
        print(f"{k//144}")
plt.plot(response_data[0:144*2])
plt.plot(Q1_data[0:144*2])

# %%

class WellModel(torch.nn.Module):
    def __init__(self, n_q = 1):
        super(WellModel, self).__init__()
        self.fc = torch.nn.Linear(1 + n_q, 1, bias=True)  # s_k and q_k as inputs
    def forward(self, s_k, q_k):
        input = torch.cat((s_k, q_k), dim=-1)  # Concatenate s_k and q_k
        dS = self.fc(input)  # Linear transformation
        next_s = s_k + dS  # Update drawdown
        return next_s  # Output is the predicted drawdown 
    

#%% Simple dataset and dataloader
class WellDataset(Dataset):
    def __init__(self, Q_data, s_data, n_q=1):
        
        self.n_q = n_q

        self.Q_data = Q_data[0:-n_q]
        self.s_data = s_data[1:-n_q]
        self.target = s_data[n_q:]

    def __len__(self):
        return len(self.Q_data)-self.n_q

    def __getitem__(self, idx):
        # s_k: previous drawdown, q_k: current Q
        s_k = self.s_data[idx]  # or use a more meaningful feature
        q_k = self.Q_data[idx:idx + self.n_q]  # current flow rate
        return torch.tensor([s_k], dtype=torch.float32), torch.tensor(q_k, dtype=torch.float32), torch.tensor([self.target[idx]], dtype=torch.float32)


#%% Model, loss, optimizer
n_q = 2
dataset = WellDataset(Q1_data, response_data, n_q=n_q)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
model = WellModel(n_q=n_q)
criterion = torch.nn.HuberLoss(delta=.1)  # Huber loss is robust to outliers
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for s_k, q_k, target in dataloader:
        optimizer.zero_grad()
        output = model(s_k, q_k)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
# %%

response = np.zeros(simu_steps)
response2 = np.zeros(simu_steps) 
Q = [0.0]*(n_q-1)
s_k = 0.0
for k in range(1,simu_steps//50):
    Q = Q + [aprbs()]
    Q1 = np.array(Q)
    response2[k] = drawdown(k, Q1[n_q-1:])
    if k % 100 == 0:
        pass#s_k = response2[k-1]  # Use the last known drawdown as the initial condition
    with torch.no_grad():
        q_k = torch.tensor(Q1[k-1:k+n_q-1], dtype=torch.float32)
        s_k = model(torch.tensor([s_k], dtype=torch.float32), q_k).item()
        response[k] = s_k
#%%    
plt.plot(np.diff(response[0:144*2]))
plt.plot(np.diff(response2[0:144*2]))

fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.title("Drawdown of the well")
plt.grid()

# %%
