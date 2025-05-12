#%%
from environments.utils.predict_utils import Predictor, EarlyStopping
from environments.Data.EMS.EMS_constants import *
import numpy as np
from environments.utils.funcionesEMS import *
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random

temperature = np.concatenate((get_temperatura("ver")[0:81*144], get_temperatura("inv")[0:81*144]), axis=0)
radiation = np.concatenate((get_rad("ver")[0:81*144], get_rad("inv")[0:81*144]), axis=0)

p_pv = solar_power(radiation, temperature)

train_length = 60*144 #60 days
val_length = 21*144 #21 days

mean, std = np.mean(p_pv), np.std(p_pv)
p_pv_norm = torch.tensor((p_pv - mean) / std, dtype=torch.float32, device="cuda")

temperature_train = np.concatenate((get_temperatura("ver")[0:train_length], get_temperatura("inv")[0:train_length]), axis=0)
radiation_train = np.concatenate((get_rad("ver")[0:train_length], get_rad("inv")[0:train_length]), axis=0)
p_pv_train = torch.tensor((solar_power(radiation_train, temperature_train) - mean)/std, dtype=torch.float32)

temperature_val = np.concatenate((get_temperatura("ver")[train_length:], get_temperatura("inv")[train_length:]), axis=0)
radiation_val = np.concatenate((get_rad("ver")[train_length:], get_rad("inv")[train_length:]), axis=0)
p_pv_val = torch.tensor((solar_power(radiation_val, temperature_val) - mean)/std, dtype=torch.float32)

#%%
class PVDataSet(Dataset):
    def __init__(self, data:torch.Tensor, seq_len=144):
        self.data:torch.Tensor = data
        self.seq_len = seq_len
        self.n_days = len(data) // seq_len
    
    def __getitem__(self, index):
        #pick a random day
        j = random.randint(0, self.n_days - 3)
        #moment of day 
        k = index % 144
        # pick a random coefficient
        lambda_ = random.uniform(0, .1)

        day_val = self.data[index:index+self.seq_len]
        next_day_val = self.data[index+self.seq_len:index+2*self.seq_len]

        other_day_val = self.data[144*j+k: 144*j+k+self.seq_len]
        other_next_day_val = self.data[144*j+k+self.seq_len: 144*j+k+2*self.seq_len]

        x = day_val*(1-lambda_) + other_day_val*(lambda_)
        y = next_day_val*(1-lambda_) + other_next_day_val*(lambda_)
        return x.unsqueeze(0), y.unsqueeze(0)
    
    def __len__(self):
        return len(self.data) - 2 * self.seq_len

#%%

train_dataset = PVDataSet(p_pv_train, seq_len=144)
val_dataset = PVDataSet(p_pv_train, seq_len=144)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for x,y in val_loader:
    print(x.shape, y.shape)
    break
#%%

# configure the training loop
mdl = Predictor(n_features=1, n_hidden=64, pred_steps=144, n_layers=2, mean=mean, std=std, device="cuda")
optimizer = torch.optim.Adam(mdl.parameters(), lr=0.01)
#
# optimizer = torch.optim.SGD(mdl.parameters(), lr=0.001)
loss_fn = torch.nn.HuberLoss()
early_stopping = EarlyStopping(patience=10, tolerance=1e-4)


# Initialize early stopping
early_stopping = EarlyStopping(patience=5, tolerance=1e-4)
#%% train the model
mdl.train()
for epoch in range(100):
    stop_training = False
    for i, (x, y) in enumerate(train_loader):
        x = x.to("cuda")
        y = y.to("cuda")
        optimizer.zero_grad()
        y_pred = mdl(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            with torch.no_grad():
                val_loss = 0
                for x_val, y_val in val_loader:
                    x_val = x_val.to("cuda")
                    y_val = y_val.to("cuda")
                    y_pred_val = mdl(x_val)
                    val_loss += loss_fn(y_pred_val, y_val).item()
                val_loss /= len(val_loader)

                print(f"Epoch {epoch}, Batch {i}, Validation Loss: {val_loss}")
                if early_stopping(val_loss):
                    print("Early stopping triggered.")
                    stop_training = True
                    break
    if stop_training:
        break
#%%
mdl.save("predictive_models/pv_model.pt")
#%%
x, y = train_dataset[random.randint(0,999)]

plt.plot(x[0].to("cpu").numpy().flatten(), label="Input")
plt.plot(y[0].to("cpu").numpy().flatten(), label="True")
#%%

mdl = Predictor(n_features=1, pred_steps=144, n_hidden=64, n_layers=2, mean=mean, std=std, device="cuda")
mdl.load("predictive_models/pv_model.pt")
mdl.eval()
#%%
x, y = train_dataset[144]
a = mdl.predict(x.to("cuda"), 144, isNormalized=True).to("cpu").numpy().flatten()
plt.plot(a, label="Predicted")

plt.plot(y.to("cpu").numpy().flatten(), label="True")

#%%
plt.plot(x[2].to("cpu").numpy().flatten(), label="Input")