import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from predictive_models.utils import NN_soil_mdl, MLP
import torch


def obs_dict_2_obs_array(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Turns an observation dictionary into a flattened array"""
    return np.array([obs[crop_name] for crop_name in obs.keys()]).flatten()


class IrrigationPolicy(ABC):
    def __init__(self, n_crops: int):
        self.n_crops = n_crops

    @abstractmethod
    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pass


class LearnedIrrigationPolicy(IrrigationPolicy):
    def __init__(self, n_crops: int, rl_policy):
        super().__init__(n_crops)
        self.n_crops = n_crops
        self.rl_policy = rl_policy

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        obs_array = obs_dict_2_obs_array(obs)
        action = self.rl_policy.predict(obs_array, deterministic=True)[0]
        return action


class ModelBasedIrrigationPolicy:
    def __init__(self, n_crops, neural_model: NN_soil_mdl, root_length_model: MLP):
        """
        : param n_crops: number of crops
        : param neural_model: neural network model for theta_a
        : param root_length_model: neural network model for root length
        """
        self.theta_model:NN_soil_mdl = neural_model
        self.root_length_model: MLP = root_length_model
        self.theta_fc = 0.3
        self.theta_wp = 0.13
        self.mad = .5

    def get_action(self, obs: np.ndarray, prev_irrigation, evapotranspiration, expected_water) -> np.ndarray:
        """
        :param obs: observation array
        :param prev_irrigation: previous irrigation  [m3]
        :param evapotranspiration: evapotranspiration [m3]
        :param expected_water: expected water [m3]
        """
        obs = np.concatenate((np.array([prev_irrigation, evapotranspiration, expected_water]), obs))
        theta_a = self.theta_model.predict(torch.tensor(obs, dtype=torch.float32))
        root_length = self.root_length_model.predict(torch.tensor(obs, dtype=torch.float32))
        threshold = self.theta_fc - self.mad*(self.theta_fc - self.theta_wp)
        action = 0.0
        if theta_a < threshold:
            action = (self.theta_fc - theta_a)*abs(root_length) - expected_water
        return action


class TriggeredIrrigationPolicy(IrrigationPolicy):
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
