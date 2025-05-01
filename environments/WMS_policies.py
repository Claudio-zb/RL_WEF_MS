import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod, ABCMeta   
from predictive_models.utils import NN_soil_mdl, MLP
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
import pyswarms as ps
import copy
import pandas as pd
from environments.Cultivates import Cultivates
from typing import Any


def obs_dict_2_obs_array(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Turns an observation dictionary into a flattened array. 
    It works for centralized WMS agents. It can be used for single crop RL agents as well."""
    return np.array([obs[crop_name] for crop_name in obs.keys()]).flatten()


class Policy(metaclass = ABCMeta):
    """Abstract Class for policies"""

    @abstractmethod
    def get_action(self, observation, disturbances):
        pass


class IrrigationPolicy(Policy, metaclass = ABCMeta):
    def __init__(self, n_crops: int, year:int):
        self.n_crops:int = n_crops
        self.year:int = year

    @abstractmethod
    def get_action(self, obs_dict: dict[str, np.ndarray], disturbances: np.ndarray, doy:int) -> np.ndarray[float]:

        """ Returns the irrigation action [m] for each crop in the setup. 
        :param obs_dict: dictionary with the observations for each crop
        :param disturbances: array with the disturbances for each crop
        :param doy: day of the year
        :return: irrigation action [m] for each crop in the setup"""
        pass

class PPOIrrigationPolicy(IrrigationPolicy):
    pass


class RLIrrigationPolicy(IrrigationPolicy):
    """Irrigation manager implemented by RL agent"""
    def __init__(self, n_crops: int, rl_policy: BaseAlgorithm, year:int, 
                 days_ahead:int = 1, isNormalized:float=True):
        
        super().__init__(n_crops, year)
        self.rl_policy:BaseAlgorithm = rl_policy
        self.isNormalized:float = isNormalized
        self.count:int = 1
        self.relative_yield:float = 1.0
        self.days_ahead: int = 1 
        self.weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
        self.days_ahead = days_ahead
    
    def get_action(self, obs: Dict[str, np.ndarray], disturbances:np.ndarray, doy) -> np.ndarray:
        
        index = int(self.weather_data.loc[(self.weather_data["year"] == self.year) & (self.weather_data["doy"] == doy)].index.values[0])
        
        precipitations = self.weather_data.iloc[index:index+self.days_ahead]["precipitation"].values
        obs_array = obs_dict_2_obs_array(obs)
        obs_array_ = np.zeros(11*self.n_crops + self.days_ahead) 
        obs_array_[:10*self.n_crops] = obs_array  # assign the observations (length = 10)
        obs_array_[8] = 0.0 if obs_array_[8] < 3 else 1.0 # normalize the drought indicator
        obs_array_[9] = obs_array_[9] / 114 # normalize the time component
        self.relative_yield = self.relative_yield*obs_array[7]
        obs_array_[10] = self.relative_yield**(1/self.count)
        
        obs_array_[11:] = np.abs(precipitations*(1.0 + np.random.randn(self.days_ahead)*0.1)) # add noise to the precipitation predictions
        action = self.rl_policy.predict(obs_array_, deterministic=True)[0]*20.0/1000
        self.count += 1
        return action


class RuleBasedIrrigationPolicy(IrrigationPolicy):
    """Class for Irrigation Policy Based on Predictions for the soil moisture"""
    def __init__(self, n_crops, neural_model: NN_soil_mdl, root_length_model: MLP, theta_fc: float = 0.3, theta_wp: float = 0.13, mad: float = .5):
        """
        : param n_crops: number of crops
        : param neural_model: neural network model for theta_a
        : param root_length_model: neural network model for root length
        """
        super().__init__(n_crops)
        self.theta_model:NN_soil_mdl = neural_model
        self.root_length_model: MLP = root_length_model
        self.theta_fc:float = theta_fc
        self.theta_wp:float = theta_wp
        self.mad:float = mad
        self.prev_irrigation:float = 0.0

    def get_action(self, obs: np.ndarray, evapotranspiration, expected_water) -> np.ndarray:
        """
        :param obs: observation array
        :param prev_irrigation: previous irrigation  [m3]
        :param evapotranspiration: evapotranspiration [m3]
        :param expected_water: expected water [m3]
        """
        if len(obs) > 6:
            obs_ = np.concatenate((obs[:5], obs[-3:-2]))
        else:
            obs_ = obs  
        obs = np.concatenate((np.array([self.prev_irrigation, evapotranspiration, expected_water]), obs_))
        theta_a = self.theta_model.predict(torch.tensor(obs, dtype=torch.float32))-.1
        root_length = self.root_length_model.predict(torch.tensor(obs, dtype=torch.float32)) 
        threshold = self.theta_fc - self.mad*(self.theta_fc - self.theta_wp)
        action = 0.0
        if theta_a < threshold:
            action = (self.theta_fc - theta_a)*abs(root_length) - expected_water

        self.prev_irrigation = action
        return action 
    
def rule_based_policy() -> RuleBasedIrrigationPolicy:
    """
    Creates a rule-based irrigation policy
    """
    theta_models = [torch.load(f"predictive_models/soil_moisture/theta_{5-j}.pth", weights_only=False) for j in range(1, 5)]
    theta_models = [torch.load("predictive_models/soil_moisture/theta_evp.pth", weights_only=False)] + theta_models
    root_length_model = torch.load("predictive_models/soil_moisture/root_depth.pth", weights_only=False)
    theta_a_mdl = NN_soil_mdl(theta_models, root_length_model)

    return RuleBasedIrrigationPolicy(n_crops=1, neural_model=theta_a_mdl, root_length_model=root_length_model)

class RBIrrigationPolicy(IrrigationPolicy):
    def __init__(self, n_crops, model: Cultivates, year:int):
        super().__init__(n_crops, year)
        self.weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
        self.model:Cultivates = model
        self.index:int = int(self.weather_data.loc[(self.weather_data["year"] == self.year) & (self.weather_data["doy"] == self.model.crops[0].plantation_day)].index.values[0])
        self.days_since_plantation:int = 0
        
    
    def get_action(self, obs, disturbances, doy) -> np.ndarray:

        disturbances_dict = self.weather_data.iloc[self.index + self.days_since_plantation].to_dict()
        
        self.model.set_state(obs, doy)
        obs_dict, _ = self.model.step([0.0], disturbances_dict)
        expected_water = self.weather_data.iloc[self.index + self.days_since_plantation:self.index + self.days_since_plantation + 3]["precipitation"].values
        # add 10% of uncertainty to expected water
        expected_water = np.abs(expected_water*(1.0 + np.random.randn(len(expected_water))*0.1))

        expected_water = sum(expected_water)*0.001 # [mm] -> [m]
        actions = np.zeros(self.n_crops)
        
        for idx, crop in enumerate(self.model.crops):
            obs_array = obs_dict[crop.crop_name]
            n_layers = len(crop.soil.get_reversed_layers())
            theta_a = np.sum([obs_array[i]*crop.soil.get_reversed_layers()[i].depth for i in range(n_layers)])/crop.soil.get_depth()
            threshold = crop.soil.get_theta_fc() - crop.MAD*(crop.soil.get_theta_fc() - crop.soil.get_theta_wp())
            if theta_a < threshold:
                action = (crop.soil.get_theta_fc() - theta_a)*abs(crop.root_depth) - expected_water
            else:
                action = 0.0

            actions[idx] = action
        
        self.days_since_plantation += 1

        return actions
            
        



class ScheduledIrrigationPolicy(IrrigationPolicy):
    """Class that defines policies with a fixed irrigation schedule"""
    def __init__(self, n_crops: int, frequency: int, irr_amount: float = 5.0):
        """
        :param n_crops: number of crops
        :param frequency: frequency of irrigation
        :param irr_amount: amount of irrigation [mm]
        """
        super().__init__(n_crops)
        self.days_count: int = 1
        self.irr_amount = irr_amount
        self.frequency = frequency

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        irrigation = np.zeros(self.n_crops)
        if self.days_count >= self.frequency:
            irrigation = np.ones(self.n_crops)*self.irr_amount*.001 # [mm] -> [m]
            self.days_count = 0
        self.days_count += 1
        return irrigation
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        irrigation = self.__call__(obs)  
        return irrigation

class MPCIrrigationPolicy(IrrigationPolicy):
    """Class that defines policies via Model Predictive Control"""
    def __init__(self, n_crops:int, model:Cultivates, year:int, horizon: int = 7, 
                 reward_weights:np.ndarray = np.array([1.0, 1.0, 1.0])):

        super().__init__(n_crops, year)

        self.horizon = horizon
        self.weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
        self.previous_solution:np.ndarray = np.zeros(self.horizon)
        self.model:Cultivates = model
        self.et_model: Predictor = Predictor(n_features=1, n_hidden=10, n_layers=1, mean=0.0, std=1.0)
        self.et_model:Predictor = torch.load("predictive_models/et_model_2.pth", weights_only=False)
        self.et_model.to("cpu")
        self.reward_weights = reward_weights
        self.first_index:int = None
        self.days_count:int = 0
    
    def get_action(self, obs:np.ndarray, disturbances:np.ndarray, doy:int) -> np.ndarray[np.floating]:
        if self.first_index is None:
            self.first_index = int(self.weather_data.loc[(self.weather_data["year"] == self.year) & (self.weather_data["doy"] == doy)].index.values[0])
        timestamp = self.first_index + self.days_count
        self.model.set_state(obs, int(obs["potato"][-1]))
        # Set-up hyperparameters
        init_position_ = np.concatenate((self.previous_solution[1:], np.zeros(1)))
        init_position = np.random.uniform(low=0, high=0.02, size=(10*self.horizon, self.horizon)) 
        init_position[0, :] = init_position_
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
        # Create bounds
        max_bound = .02 * np.ones(self.horizon)
        min_bound = np.zeros(self.horizon) 
        bounds = (min_bound, max_bound)
        pred_disturbances = []
        et_regresors = self.weather_data.iloc[timestamp-7:timestamp]["ET_0"].values
        et_predictions = self.et_model.predict(torch.tensor(et_regresors, dtype=torch.float32).unsqueeze(0), 
                                               n_steps=self.horizon, 
                                               isNormalized=False).to("cpu").numpy().flatten()
        
        precipitation_preds = self.weather_data.iloc[timestamp:timestamp+self.horizon]["precipitation"].values
        precipitation_preds[1:5] = np.abs(np.random.normal(precipitation_preds[0:5], 0.1*precipitation_preds[0:5]))
        precipitation_preds[5:] = np.abs(np.random.normal(precipitation_preds[5:], 0.2*precipitation_preds[5:]))
        precipitation_preds = np.abs(precipitation_preds)
        for j in range(self.horizon):
            weather_dict = {"ET_0":et_predictions[j],
                            "precipitation": precipitation_preds[j],
                            "doy": self.weather_data.iloc[timestamp+j]["doy"]} 
            pred_disturbances.append(weather_dict)

        # Call instance of PSO
        optimizer = ps.single.LocalBestPSO(n_particles=10*self.horizon, 
                                           dimensions=self.horizon, init_pos=init_position, 
                                           options=options, bounds=bounds)

        cost_fun = lambda x: self.cost_function(obs, x, pred_disturbances=pred_disturbances, 
                                                   weights=self.reward_weights)

        # Perform optimization
        cost, action = optimizer.optimize(cost_fun, iters=20, n_processes=None)
        self.previous_solution = action
        self.days_count += 1
        return [action[0]]  # [m]
    
    def cost_function(self, obs: dict[str, np.ndarray], actions: np.ndarray, pred_disturbances: List[dict],
                      weights:np.ndarray = np.array([1.0, 1.0, 1,0])) -> float:
        """Cost function for the PSO"""
        cost = np.zeros(actions.shape[0])
        prev_obs = obs["potato"]
        self.model.set_state(obs, pred_disturbances[0]["doy"])
        for particle, action in enumerate(actions):
            pso_model = copy.deepcopy(self.model)
            for crop in self.model.crops:
                for idx in range(self.horizon):
                    obs_dict, _ = pso_model.step([action[idx]], pred_disturbances[idx])
                    obs_array = obs_dict["potato"]
                    Ks = obs_array[7]
                    delta_Ks = Ks - prev_obs[7] 
                    cost[particle] += -weights[0]*Ks**2 + weights[1]*(delta_Ks)**2 + weights[2]*(action[idx]*1000/20)**2
        return cost
    
    
    def get_predicted_disturbances(self, measured_disturbances: np.ndarray):
        pass

###### Classes for the observation handlers   

class ObservationHandler(metaclass = ABCMeta):
    """Abstract Class for observation handlers"""
    @abstractmethod
    def get_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pass

class Predictor(torch.nn.Module):

    def __init__(self, n_features, n_hidden, n_layers, mean, std, device="cpu"):
        super(Predictor, self).__init__()
        self.device = device
        self.lstm = torch.nn.LSTMCell(n_features, n_hidden, n_layers, device)
        self.mlp = torch.nn.Linear(n_hidden, n_features, device=device)
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)

    def forward(self, x:torch.Tensor):
        h_and_c = None
        batch_size, n_features, t_steps = x.shape
        y = torch.zeros((batch_size, n_features, t_steps), device=x.device)
        for i in range(t_steps):
            h_and_c = self.lstm.forward(x[:,:,i], h_and_c)
            y[:,:,i] = self.mlp(h_and_c[0])
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
    
    def load_model_parameters(self, state_dict):
        self.load_state_dict(state_dict)

    def to(self, device):
        self.device = device
        self.lstm.to(device)
        self.mlp.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
