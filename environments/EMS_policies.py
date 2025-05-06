import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union
from stable_baselines3.common.base_class import BaseAlgorithm
from environments.utils.funcionesEMS import *

class PumpingPolicy(ABC):
    """
    Abstract class for pumping policies.
    """

    def __init__(self, n_crops=1, ):
        self.n_crops = n_crops

    @abstractmethod
    def get_action(self, water_reqs:np.ndarray, observation:np.ndarray) -> np.ndarray:
        """
        Get the action for the given observation.
        :param observation: The observation of the environment.
        :return: The action to be taken.
        """
        pass
    


class RBPumpingPolicy(PumpingPolicy):
    """
    Rule-based pumping policy.
    """
    def __init__(self, n_crops=1, v_tank_max:float=5.0):
        super().__init__(n_crops)
        self.v_tank_max:float = v_tank_max

    def get_action(self, water_reqs:np.ndarray, observation:np.ndarray) -> np.ndarray:
        
        v_tanks = np.array(observation[0:self.n_crops])
        v_irrs = np.array(observation[self.n_crops:2*self.n_crops])
        assert len(water_reqs) == self.n_crops, "Water requirements and number of crops do not match"
        
        q_irrs = np.clip((water_reqs - v_irrs)*1000/3600, 0, 1)
        q_ps = np.array([q_irrs[idx] if v_tank < self.v_tank_max else 0.0 for idx, v_tank in enumerate(v_tanks)]) 
        return np.concatenate((q_ps, q_irrs)) 
    
class RLPumpingPolicy(PumpingPolicy):
    """Class for the Q_p, Q_irr action space agent"""
    def __init__(self, n_crops: int, rl_policy: BaseAlgorithm, isNormalized: bool = False):
        super().__init__(n_crops)
        self.policy = rl_policy
        self.residual: float = 0.0
        if isNormalized:
            self.transform = generate_t_matrix(n_crops)
        else:
            self.transform = np.eye(4 * n_crops + 5)

    def get_action(self,
                   state: tuple[list[float], list[float], list[float], float, int],
                   v_reqs: Union[list[float], np.ndarray],
                   disturbances) -> tuple[float, list[list]]:
        """Computes the action since the current observation
        disturbances: Tuple of (p_fv, p_load)
        return pbat, [[Q_p, Qirr]]"""
        p_fv = disturbances[0]
        p_load = disturbances[1]
        n_crops = self.n_crops

        v_tanks, v_irrs, drawdowns = state[0], state[1], state[2]

        flattened_state = np.concatenate((v_reqs, v_tanks, v_irrs, drawdowns,
                                          np.array([p_fv, p_load, state[3], self.residual, state[4]])))
        transformed_state = np.matmul(self.transform, flattened_state)
        actions = self.policy.predict(transformed_state, deterministic=True)[0]
        Q_p = actions[0:n_crops]
        P_q_ps = get_p_q_p(Q_p, h_p_const)
        Q_irr = actions[n_crops:]

        for i in range(n_crops):
            if v_reqs[i] < v_irrs[i]:
                Q_irr[i] = 0.0
        
        pumps = [[Q_p[i], Q_irr[i]] for i in range(n_crops)]

        SoE = state[3]
        Pbat, _, self.residual = manage_batteries(SoE, p_fv, p_load, P_q_ps)
        return Pbat, pumps

        





