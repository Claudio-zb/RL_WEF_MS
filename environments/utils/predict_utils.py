import torch
import numpy as np

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

class Predictor(torch.nn.Module):

    def __init__(self, n_features, n_hidden, n_layers, mean, std, pred_steps, device="cpu"):
        super(Predictor, self).__init__()
        self.device = device
        self.n_hidden:int = n_hidden
        self.n_layers:int = n_layers
        self.n_features:int = n_features
        self.pred_steps:int = pred_steps

        self.lstm = torch.nn.LSTMCell(n_features, n_hidden, n_layers, device)
        self.mlp = torch.nn.Linear(n_hidden, n_features, device=device)
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)

    #def forward(self, x:torch.Tensor):
    #    h_and_c = None
    #    batch_size, n_features, t_steps = x.shape
    #    y = torch.zeros((batch_size, n_features, t_steps), device=x.device)
    #    for i in range(t_steps):
    #        h_and_c = self.lstm.forward(x[:,:,i], h_and_c)
    #        y[:,:,i] = self.mlp(h_and_c[0])
    #    return y

    def forward(self, x:torch.Tensor):
        self.pred_steps = 144
        h_and_c = None
        batch_size, n_features, t_steps = x.shape
        y = torch.zeros((batch_size, n_features, t_steps), device=x.device)
        for i in range(t_steps):
            h_and_c = self.lstm.forward(x[:,:,i], h_and_c)
        y[:,:,0] = self.mlp(h_and_c[0])
        for i in range(self.pred_steps-1):
            h_and_c = self.lstm.forward(y[:,:,i].clone(), h_and_c)
            y[:,:,i+1] = self.mlp(h_and_c[0])

        return y
    
    def predict(self, x:torch.Tensor, n_steps:int, isNormalized:bool = False) -> torch.Tensor:
        """
        Predict the next n_steps values of the input sequence x.
        """
        assert x.dim() == 2, "Input x must be a 2D tensor." 
        
        x = x if isNormalized else (x - self.mean) / self.std
            
        with torch.no_grad():
            h_and_c = None
            n_features, t_steps = x.shape
            y = torch.zeros((n_features, n_steps), device=x.device)
            for i in range(t_steps):
                h_and_c = self.lstm.forward(x[:,i], h_and_c)
            y[:,0] = self.mlp(h_and_c[0])
            for i in range(n_steps-1):
                h_and_c = self.lstm.forward(y[:,i], h_and_c)
                y[:,i+1] = self.mlp(h_and_c[0])

        y = y if isNormalized else y * self.std + self.mean

        return y

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
            'n_features': self.n_hidden,
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

