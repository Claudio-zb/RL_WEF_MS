#%%
from environments.utils.predict_utils import Predictor, EarlyStopping, PVDataSet, load_model
from environments.Data.EMS.EMS_constants import *
import numpy as np
from environments.utils.funcionesEMS import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
n_pred = 144

pv_inv = solar_power(get_rad("inv")[0:81*144], get_temperatura("inv")[0:81*144])
pv_ver = solar_power(get_rad("ver")[0:81*144], get_temperatura("ver")[0:81*144])

p_pv = np.concatenate((pv_inv, pv_ver), axis=0)
mean, std = np.mean(p_pv), np.std(p_pv)

train_length = 60*144 #60 days
val_length = 21*144 #21 days


p_pv_train = torch.tensor(np.concatenate((pv_inv[:train_length], pv_ver[:train_length]) - mean)/std, dtype=torch.float32)
p_pv_val = torch.tensor(np.concatenate((pv_inv[train_length:], pv_ver[train_length:]) - mean)/std, dtype=torch.float32)

#%%

train_dataset = PVDataSet(p_pv_train, x_len=288, pred_steps=n_pred)
val_dataset = PVDataSet(p_pv_val, x_len=288, pred_steps=n_pred)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

for x,y in train_loader:
    print(x.shape, y.shape)
    break

#%%

#%%

# configure the training loop
mdl = Predictor(n_features=1, n_hidden=64, pred_steps=n_pred, n_layers=1, mean=mean, std=std, device="cuda")

optimizer = torch.optim.RAdam(mdl.parameters(), lr=0.001)
#
# optimizer = torch.optim.SGD(mdl.parameters(), lr=0.001)
loss_fn = torch.nn.HuberLoss()
early_stopping = EarlyStopping(patience=10, tolerance=1e-5)

#%% train the model
train = False
if train:
    for epoch in range(100):
        mdl.train()
        stop_training = False
        for i, (x, y) in enumerate(train_loader):
            x = x.to("cuda")
            y = y.to("cuda")
            y_pred = mdl.forward(x, None, n_pred)
            y_pred = y_pred
            train_loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss = 0
        if epoch % 2 == 0:
            mdl.eval()
            with torch.no_grad():
                val_loss = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to("cuda")
                    y_val = y_val.to("cuda")
                    y_pred_val = mdl(x_val, None, n_pred)
                    val_loss.append(loss_fn(y_pred_val, y_val).item())
                val_loss = np.mean(val_loss)
                print(f"Epoch {epoch}, Batch {i}, Validation Loss: {val_loss}")
                if early_stopping(val_loss):
                    print("Early stopping triggered.")
                    stop_training = True
                    break
        if stop_training:
            break
mdl.save(mdl.state_dict(), "predictive_models/pv_model.pt")
#%%
test_dataset = PVDataSet(p_pv_val, x_len=144, pred_steps=144)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
x, y = test_dataset[0]
plt.plot(x.to("cpu").numpy().flatten(), label="Input")
plt.plot(y.to("cpu").numpy().flatten(), label="True")
#%%

mdl = load_model("predictive_models/pv_model.pt", device="cuda")
mdl.eval()
#%%
for x, y in test_loader:
    x = x.to("cuda")
    y = y.to("cuda")
    break
a = mdl.forward(x.to("cuda")[0:1], None, 288+144).to("cpu").detach().numpy().flatten()[144:]
plt.plot(a, label="Predicted")

plt.plot(y[0:1].to("cpu").numpy().flatten(), label="True")

#%%
plt.plot(x.to("cpu").numpy().flatten(), label="Input")
# %%
