import torch
import torch.nn as nn
import pandas as pd
from typing import List
import numpy as np
from torch.utils.data import Dataset


class MLP(nn.Module):
    def __init__(self, input_size, output_size,
                 X_mean=0, X_std=1,
                 y_mean=0, y_std=1):
        super(MLP, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )
        self.X_mean = X_mean
        self.y_mean = y_mean
        self.X_std = X_std
        self.y_std = y_std


    def forward(self, x):
        x = self.sequence(x)
        return x
    def predict(self, x) -> float:
        """

        """
        with torch.no_grad():
            x = (x - self.X_mean) / self.X_std
            prediction = self.forward(x)*self.y_std + self.y_mean
            return prediction.item()

class NN_soil_mdl:
    """
    This class is a wrapper for the soil moisture model. It takes the predictions of the soil moisture in each layer
    """
    def __init__(self, soil_models: List[MLP], root_depth_model: MLP):
        """
        :param: soil_models: List of MLP models for each soil layer. It starts with evp layer and then continues with
        the numbered layers in decreasing order
        :param: root_depth_model: MLP model for the root depth
        """
        self.soil_models = soil_models
        self.root_depth_model = root_depth_model
    def predict(self, x:torch.Tensor):
        soil_preds = []
        for i, model in enumerate(self.soil_models):
            pred = model.predict(x)
            soil_preds.append(pred)
        soil_preds = np.array(soil_preds)
        root_depth = self.root_depth_model.predict(x)
        root_depth = np.clip(abs(root_depth), 0, 1.1*0.8)
        active_layers = 1 + int(root_depth // 0.15)
        last_layer = root_depth % 0.15
        ponderator = np.zeros_like(soil_preds)
        for i in range(active_layers-1):
            ponderator[i] = .15
        ponderator[active_layers-1] = last_layer
        ponderator = ponderator / np.sum(ponderator)
        theta_a = np.dot(np.abs(soil_preds), ponderator)

        return theta_a


# Dataset and Datloders definition


class SoilMoistureDataset(Dataset):
    def __init__(self, exogenous_file: str, endogenous_file: str, target_feature: str,
                 dataset_type: str = 'train', train_ratio: float = 0.7, val_ratio: float = 0.15):

        self.exogenous_data = pd.read_csv(exogenous_file)
        self.endogenous_data = pd.read_csv(endogenous_file)

        # Combine exogenous and endogenous variables into a single dataset
        self.data = pd.concat([self.exogenous_data, self.endogenous_data], axis=1).iloc[:-1]

        # Normalize the data
        self.X = torch.tensor(self.data.values, dtype=torch.float32)
        self.X_mean = self.X.mean(dim=0)
        self.X_std = self.X.std(dim=0)
        self.X = (self.X - self.X_mean) / self.X_std

        y = self.endogenous_data[target_feature]
        y = y.iloc[1:]
        y = torch.tensor(y.values, dtype=torch.float32)
        self.y_mean = y.mean()
        self.y_std = y.std()
        y_tensor = (y - self.y_mean) / self.y_std
        y_tensor = y_tensor.unsqueeze(-1)
        self.y = y_tensor

        # Split the data into training, validation, and test sets
        total_samples = len(self.X)
        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(val_ratio * total_samples)

        if dataset_type == 'train':
            self.X = self.X[:train_end]
            self.y = self.y[:train_end]
        elif dataset_type == 'val':
            self.X = self.X[train_end:val_end]
            self.y = self.y[train_end:val_end]
        elif dataset_type == 'test':
            self.X = self.X[val_end:]
            self.y = self.y[val_end:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_scalers(self):
        return self.X_mean, self.X_std, self.y_mean, self.y_std

    def get_shapes(self):
        return self.X.shape[-1], self.y.shape[-1]
