import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod, ABCMeta    
from predictive_models.utils import NN_soil_mdl, MLP
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
import pyswarms as ps
import copy
import pandas as pd


def obs_dict_2_obs_array(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Turns an observation dictionary into a flattened array"""
    return np.array([obs[crop_name] for crop_name in obs.keys()]).flatten()


class Policy(metaclass = ABCMeta):
    """Abstract Class for policies"""

    @abstractmethod
    def get_action(self, observation, disturbances):
        pass


class IrrigationPolicy(Policy, metaclass = ABCMeta):
    def __init__(self, n_crops: int):
        self.n_crops = n_crops

    @abstractmethod
    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pass


class RLIrrigationPolicy(IrrigationPolicy):
    """Irrigation manager implemented by RL agent"""
    def __init__(self, n_crops: int, rl_policy: BaseAlgorithm):
        super().__init__(n_crops)
        self.rl_policy:BaseAlgorithm = rl_policy

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        obs_array = obs_dict_2_obs_array(obs)
        action = self.rl_policy.predict(obs_array, deterministic=True)
        return action
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        action = self.__call__(obs)  
        return action/1000.0


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
            irrigation = np.ones(self.n_crops)*self.irr_amount*.001 # [mm] -> [m3]
            self.days_count = 0
        self.days_count += 1
        return irrigation
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        irrigation = self.__call__(obs)  
        return irrigation

class MPCIrrigationPolicy(IrrigationPolicy):
    """Class that defines policies via Model Predictive Control"""
    def __init__(self, n_crops:int = 1, horizon: int = 10):
        super().__init__(n_crops)
        self.horizon = horizon
        self.weather_data = pd.read_csv("environments/Data/WMS/extracted_data.csv")
        self.previous_solution:np.ndarray = np.zeros(self.horizon)
    
    def get_action(self, obs, model, disturbances) -> List[np.floating]:
        obs = obs["potato"]
        self.model = model
        # Set-up hyperparameters
        init_position_ = np.concatenate((self.previous_solution[1:], np.zeros(1)))
        init_position = np.random.uniform(low=0, high=0.02, size=(10*self.horizon, self.horizon)) 
        init_position[0, :] = init_position_
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
        # Create bounds
        max_bound = .02 * np.ones(self.horizon)
        min_bound = np.zeros(self.horizon) 
        bounds = (min_bound, max_bound)
        doy = disturbances["doy"]
        pred_disturbances = []

        for j in range(self.horizon):
            weather_data = self.weather_data.loc[self.weather_data["doy"] == doy+j]
            pred_disturbances.append(weather_data.iloc[0].to_dict())

        # Call instance of PSO
        optimizer = ps.single.LocalBestPSO(n_particles=10*self.horizon, 
                                           dimensions=self.horizon, init_pos=init_position, 
                                           options=options, bounds=bounds)

        cost_fun = lambda x: self.cost_function(obs, x, pred_disturbances=pred_disturbances)

        # Perform optimization
        cost, action = optimizer.optimize(cost_fun, iters=50, n_processes=None)
        self.previous_solution = action
        return [action[0]]  # [m]
    
    def cost_function(self, obs: np.ndarray, actions: np.ndarray, pred_disturbances: List[dict]) -> float:
        cost = np.zeros(actions.shape[0])
        for particle, action in enumerate(actions):
            pso_model = copy.deepcopy(self.model)
            for crop in self.model.crops:
                for idx in range(self.horizon):
                    obs_dict = pso_model.step([action[idx]], pred_disturbances[idx])
                    obs_array = obs_dict["potato"]
                    Ks = obs_array[-1]
                    cost[particle] += -Ks + action[idx]
        return cost

###### Classes for the observation handlers   

class ObservationHandler(metaclass = ABCMeta):
    """Abstract Class for observation handlers"""
    @abstractmethod
    def get_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pass