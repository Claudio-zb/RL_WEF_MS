#%%
from environments.WMS_policy import TriggeredIrrigationPolicy
from environments.WMS_env import Cultivates
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from predictive_models.utils import*

#%% Data obtention
obtain_data = True # Set to True

if obtain_data:
    n_episodes = 500
    simu_days = 114
    observations = []
    exogenous_variables = {"irrigation": [], "evapotranspiration": [], "precipitation": []}
    endogenous_variables = {"theta_evp": [], "theta_4": [], "theta_3": [], "theta_2": [], "theta_1": [],
                            "root_depth": []}

    irr_freq = np.random.randint(1, 5)
    irr_volume = np.random.rand()*20
    irr_policy = TriggeredIrrigationPolicy(1, irr_freq, irr_volume)

    for n in range(n_episodes):
        cultivate_env = Cultivates()
        weather_data = pd.read_csv("environments/Data/WMS/weather_data.csv")

        obs, info = cultivate_env.start(weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict())

        for i in range(simu_days):
            daily_weather_data = weather_data.loc[weather_data["doy"] == cultivate_env.doy].iloc[0].to_dict()
            action = irr_policy(obs)
            # record the exogenous variables
            exogenous_variables["irrigation"].append(action[0])
            exogenous_variables["evapotranspiration"].append(daily_weather_data["ET_0"])
            exogenous_variables["precipitation"].append(daily_weather_data["precipitation"])
            obs = cultivate_env.step(action, daily_weather_data)
            # record the endogenous variables
            endogenous_variables["theta_evp"].append(cultivate_env.crops[0].soil.evp_layer.theta)
            endogenous_variables["theta_4"].append(cultivate_env.crops[0].soil.layers[3].theta)
            endogenous_variables["theta_3"].append(cultivate_env.crops[0].soil.layers[2].theta)
            endogenous_variables["theta_2"].append(cultivate_env.crops[0].soil.layers[1].theta)
            endogenous_variables["theta_1"].append(cultivate_env.crops[0].soil.layers[0].theta)
            endogenous_variables["root_depth"].append(cultivate_env.crops[0].root_depth)

        crop_data = cultivate_env.crops[0].hist_data
        crop_data = np.array(crop_data)
        soil_data = pd.DataFrame(cultivate_env.get_soil_data()[0])


    exogenous_variables = pd.DataFrame(exogenous_variables)
    endogenous_variables = pd.DataFrame(endogenous_variables)

    exogenous_variables.to_csv("environments/Data/WMS/exogenous_variables.csv", index=False)
    endogenous_variables.to_csv("environments/Data/WMS/endogenous_variables.csv", index=False)

#%%  Example usage

features = ["theta_3", "theta_1", "root_depth"] #
features = ['theta_evp', 'theta_4', 'theta_3', 'theta_2', 'theta_1', 'root_depth']

#%%

for feature in features:
    # Extract the target variable
    sm_train_dataset = SoilMoistureDataset("exogenous_variables.csv", "endogenous_variables.csv",
                                        feature, dataset_type='train')
    sm_train_dataloader = torch.utils.data.DataLoader(sm_train_dataset, batch_size=256, shuffle=True)

    sm_val_dataset = SoilMoistureDataset("exogenous_variables.csv", "endogenous_variables.csv",
                                      feature, dataset_type='val')
    sm_val_dataloader = torch.utils.data.DataLoader(sm_val_dataset, batch_size=128, shuffle=True)

    # Train the model
    criterion = nn.MSELoss()

    input_size, output_size = sm_train_dataset.get_shapes()  # Adjust as needed
    X_mean, X_std, y_mean, y_std = sm_train_dataset.get_scalers()

    model = MLP(input_size, output_size, X_mean, X_std, y_mean, y_std)

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    n_epochs = 700
    n_batch = 135
    for epoch in range(n_epochs):
        model.train()
        loss = 0
        for X, y in sm_train_dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss += criterion(output, y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in sm_val_dataloader:
                    output = model(X)
                    val_loss += criterion(output, y)

            val_loss = val_loss.item()/len(sm_val_dataloader)

            print(f'Epoch {epoch}, Loss {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"predictive_models/soil_moisture/{feature}.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

    torch.save(model, f"predictive_models/soil_moisture/{feature}.pth")

#%%

for feature in features:
    preds = []

    sm_test_dataset = SoilMoistureDataset("exogenous_variables.csv", "endogenous_variables.csv",
                                            feature, dataset_type='train')
    sm_test_dataloader = torch.utils.data.DataLoader(sm_test_dataset, batch_size=256, shuffle=True)

    X = sm_test_dataset.X*sm_test_dataset.X_std + sm_test_dataset.X_mean
    y = sm_test_dataset.y*sm_test_dataset.y_std + sm_test_dataset.y_mean

    model = torch.load(f"predictive_models/soil_moisture/{feature}.pth")

    with torch.no_grad():
        for i in range(135):
            pred = model.predict(X[i])
            # print(pred)
            preds.append(pred)


    plt.plot(preds, label='Predictions')
    plt.plot(y[:135], label='True values')
    plt.title(f"Predictions for {feature}")
    plt.show()
#%%