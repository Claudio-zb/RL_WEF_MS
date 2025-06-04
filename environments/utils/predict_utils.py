import torch
from torch.utils.data import Dataset
import numpy as np
import random

# Early stopping configuration
class EarlyStopping:
    def __init__(self, patience=5, tolerance=1e-4):
        self.patience = patience
        self.tolerance = tolerance
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss)-> bool:
        if val_loss < self.best_loss - self.tolerance:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
class ETDataset(Dataset):
    def __init__(self, data:torch.Tensor, x_len:int = 7, pred_steps:int = 7):
        self.data:torch.Tensor = data
        self.pred_steps:int = pred_steps
        self.x_len:int = x_len
        self.n_weeks = len(data) // pred_steps
    
    def __getitem__(self, index):
        """Pick a random day and return the input and output sequences.
        It uses mixed-up signals for data augmentation."""
        
        j = random.randint(0, self.n_weeks - 4) # pick a random day 
        k = index % 7 # moment of day 
        jdex = 7*j+k # pick a random coefficient
        
        lambda_ = random.uniform(0, .2)

        week_val = self.data[index:index+self.x_len]
        next_week_val = self.data[index+1: index+self.x_len+self.pred_steps]

        other_week_val = self.data[jdex:jdex+self.x_len]
        other_next_week_val = self.data[jdex+1: jdex+self.x_len+self.pred_steps]

        x = week_val*(1-lambda_) + other_week_val*(lambda_)
        y = next_week_val*(1-lambda_) + other_next_week_val*(lambda_)
        return x.unsqueeze(-1), y.unsqueeze(-1) # (t_steps, n_features)
    
    def __len__(self):
        return len(self.data) - self.x_len - self.pred_steps
    
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

    

    
class PVDataSet(Dataset):
    def __init__(self, data:torch.Tensor, x_len:int=288, pred_steps:int=144):
        self.data:torch.Tensor = data
        self.pred_steps:int = pred_steps
        self.x_len:int = x_len
        self.n_days = len(data) // pred_steps
    
    def __getitem__(self, index):
        """Pick a random day and return the input and output sequences.
        It uses mixed-up signals for data augmentation."""
        
        j = random.randint(0, self.n_days - 4) # pick a random day 
        k = index % 144 # moment of day 
        jdex = 144*j+k # pick a random coefficient
        
        lambda_ = random.uniform(0, .2)

        day_val = self.data[index:index+self.x_len]
        next_day_val = self.data[index+1: index+self.x_len+self.pred_steps]

        other_day_val = self.data[jdex:jdex+self.x_len]
        other_next_day_val = self.data[jdex+1: jdex+self.x_len+self.pred_steps]

        x = day_val*(1-lambda_) + other_day_val*(lambda_)
        y = next_day_val*(1-lambda_) + other_next_day_val*(lambda_)
        return x.unsqueeze(-1), y.unsqueeze(-1) # (t_steps, n_features)
    
    def __len__(self):
        return len(self.data) - self.x_len - self.pred_steps
    
class PDDataSet(Dataset):
    def __init__(self, data:torch.Tensor, x_len:int=288, pred_steps:int=144):
        self.data:torch.Tensor = data
        self.pred_steps:int = pred_steps
        self.x_len:int = x_len
        self.n_days = len(data) // pred_steps
    
    def __getitem__(self, index):
        """Pick a random day and return the input and output sequences.
        It uses mixed-up signals for data augmentation."""
        
        j = random.randint(0, self.n_days - 4) # pick a random day 
        k = index % 24 # moment of day 
        jdex = 24*j+k # pick a random coefficient
        
        lambda_ = random.uniform(0, .2)

        day_val = self.data[index:index+self.x_len]
        next_day_val = self.data[index+1: index+self.x_len+self.pred_steps]

        other_day_val = self.data[jdex:jdex+self.x_len]
        other_next_day_val = self.data[jdex+1: jdex+self.x_len+self.pred_steps]

        x = day_val*(1-lambda_) + other_day_val*(lambda_)
        y = next_day_val*(1-lambda_) + other_next_day_val*(lambda_)
        return x.unsqueeze(-1), y.unsqueeze(-1) # (t_steps, n_features)
    
    def __len__(self):
        return len(self.data) - self.x_len - self.pred_steps


class Predictor(torch.nn.Module):

    def __init__(self, n_features, n_hidden, n_layers, mean, std, pred_steps, device="cpu"):
        super(Predictor, self).__init__()
        self.device = device
        self.n_hidden:int = n_hidden
        self.n_layers:int = n_layers
        self.n_features:int = n_features
        self.pred_steps:int = pred_steps

        self.lstm = torch.nn.LSTMCell(input_size=n_features, hidden_size=n_hidden, device=device)
        self.mlp = torch.nn.Linear(n_hidden, n_features, device=device)
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)
        self.hidden = None

    def forward(self, x:torch.Tensor, hidden:torch.Tensor = None, n_steps=1) -> torch.Tensor:
        self.hidden = hidden
        batch_size, t_steps, n_features = x.shape
        y = torch.zeros((batch_size, t_steps, n_features,), device=x.device)
        for i in range(t_steps):
            self.hidden = self.lstm.forward(x[:,i,:], self.hidden)
            y[:,i,:] = self.mlp(self.hidden[0])
        output = self.mlp(self.hidden[0])
        for j in range(n_steps-1):
            self.hidden = self.lstm.forward(output, self.hidden)
            output = self.mlp(self.hidden[0])
            y = torch.cat((y, output.unsqueeze(-1)), dim=1)
        return y

    def predict(self, x:torch.Tensor, n_steps:int, mode = "train") -> torch.Tensor:
        """
        Predict the next n_steps values of the input sequence x.
        """    
        self.hidden = None

        if mode == "train": 
            for i in range(n_steps):
                x = self.forward(x, self.hidden)

        if mode == "eval":
            with torch.no_grad():
                for i in range(n_steps):
                    x = self.forward(x, self.hidden)        
        return x

    def to(self, device):
        self.device = device
        self.lstm = self.lstm.to(device)
        self.mlp = self.mlp.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return
    
    def save(self, file_path: str):
        """
        Save the model parameters, mean, and std to a file.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'mean': self.mean.cpu(),
            'std': self.std.cpu(),
            'n_features': self.n_features,
            'n_hidden': self.n_hidden,
            'n_layers':self.n_layers,
            'pred_steps': self.pred_steps
        }, file_path)

    def load(self, params_dict):
        """
        Load the model parameters, mean, and std from a file.
        """
        self.load_state_dict(params_dict)

def load_model(file_path: str, device="cpu"):
    """
    Load the model parameters, mean, and std from a file.
    """
    checkpoint = torch.load(file_path, map_location=device)
    model = Predictor(checkpoint['n_features'], checkpoint['n_hidden'], checkpoint['n_layers'], checkpoint['mean'], checkpoint['std'], checkpoint['pred_steps'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

class Forecaster():
    """Wrapper for the Predictor model to handle input normalization and prediction."""
    def __init__(self, model:Predictor):
        self.model = model
        model.to("cpu")
        self.mean = model.mean.clone()
        self.std = model.std.clone()

    def predict(self, pv_array:np.ndarray, n_steps:int):
        pv_tensor = torch.tensor(pv_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pv_tensor = (pv_tensor - self.mean) / self.std
            prediction = self.model.forward(pv_tensor, None, n_steps)
            prediction = torch.clamp_min(prediction * self.std + self.mean, 0.0)
        return prediction.detach().numpy().flatten()[-n_steps:]
        

