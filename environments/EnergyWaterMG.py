import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from environments.utils.funcionesEMS import *
import torch

from scipy.special import exp1

class EnergyWaterMG:
    def __init__(self, n_crops: int = 1):

        # setting up the environment
        self.n_crops: int = n_crops
        self.v_tanks_min: list[float] = [Vt_min] * n_crops
        self.v_tanks_max: list[float] = [Vt_max] * n_crops

        # ss variables
        self.v_tanks: np.ndarray[np.float32] = np.ones(n_crops) * (Vt_max + Vt_min) / 2
        self.v_irrs: np.ndarray[np.float32] = np.zeros(n_crops)
        self.h_0: np.ndarray[np.float32] = 50* np.ones(n_crops)  # initial height of the water in the tank (5 meters)
        self.drawdowns: np.ndarray[np.float32] = np.zeros(n_crops)
        self.soe: float = SoE_max
        self.p_pumps: np.ndarray[np.float32] = np.zeros(n_crops)  # power of the pumps [kW]

        # drawdown relevant variables
        self.prev_Qps: np.ndarray[np.float32] = np.zeros(n_crops)
        self.prev_Qirrs: np.ndarray[np.float32] = np.zeros(n_crops)
        self.dQs: list[np.ndarray] = [np.array([0])] * n_crops
        self.e_residual: float = 0.0  # residual energy [kWh]

        # 10-minutes counter
        self.k: int = 0

    def next_step(self, actions: np.ndarray, disturbances:np.ndarray) -> np.ndarray[np.float32, int]:
        """
        note: the pbat action is computed from an external policy
        :param actions: array of actions [q_p, ..., q_irr]
        :param disturbances: array of disturbances [p_fv, p_d]
        :return: Tuple of (v_tanks, v_irrs, drawdowns, soe, k)
        """
        assert len(actions) == 2*self.n_crops
        if (self.k % 144) == 0:  # The time at s is 00:00 i.e. a new day is starting
            for idx in range(self.n_crops):
                self.v_irrs[idx] = 0.0


        # unpacking the actions
        if isinstance(actions, np.ndarray):
            q_ps = np.clip(actions[:self.n_crops], 0, Q_p_max)  # [l/s]
            q_irrs = np.clip(actions[self.n_crops:], 0, I_max)  # [l/s]

        else:
            q_ps = actions[:self.n_crops]  # [l/s]
            q_irrs = actions[self.n_crops:] # [l/s]

        # loop over the crops
        for idx, v_tank in enumerate(self.v_tanks):

            if abs(self.drawdowns[idx]) > 1.0:  # If the drawdown is too high, we cannot extract water
                q_ps[idx] = 0.0

            if self.v_tanks[idx] <= Vt_min:  # If the tank is empty, there is no irrigation
                q_irrs[idx] = 0.0

            if self.v_tanks[idx] + q_ps[idx]*600/1000 > Vt_max:  # If the tank is full, there is no pumping
                q_ps[idx] = (Vt_max - self.v_tanks[idx])*1000/600

            self.v_tanks[idx] = np.clip(v_tank + (q_ps[idx] - q_irrs[idx]) * 600 / 1000, self.v_tanks_min[idx],
                                        self.v_tanks_max[idx])
            
            # volume_to_extract = q_irrs[idx]*600/1000
            # volume_available = np.max([self.v_tanks[idx] - self.v_tanks_min[idx], 0])

            # if volume_to_extract > volume_available:
            #     q_irrs[idx] = volume_available*1000/600
            
            self.v_irrs[idx] = np.clip(self.v_irrs[idx] + q_irrs[idx] * 600 / 1000, 0, np.inf)
            self.drawdowns[idx] = drawdown(self.k + 1, self.dQs[idx] / 1e3)
            self.dQs[idx] = np.append(self.dQs[idx], q_ps[idx] - self.prev_Qps[idx])
            self.prev_Qps[idx] = q_ps[idx]
            self.prev_Qirrs[idx] = q_irrs[idx]

            self.p_pumps[idx] = get_p_q_p(q_ps[idx], self.h_0[idx] + self.drawdowns[idx])  # [kW]
        p_bat, next_soe, self.e_residual = manage_batteries(self.soe, disturbances[0], disturbances[1], self.p_pumps)
        delta_SoE = np.max([p_bat, 0]) * n_c * (dt / 3600) + np.min([p_bat, 0]) / n_d * (dt / 3600)  # [kWh]
        self.soe = np.clip(self.soe + delta_SoE, SoE_min, SoE_max)
        assert np.isclose(self.soe, next_soe) 

        self.k += 1

        return self.get_observation()
    
    def get_observation(self) -> np.ndarray[np.float32, int]:
        """ Get the observation of the environment
        Returns: Array of v_tanks, v_irrs, drawdowns, p_pumps, soe, e_residual, k"""

        observation = np.concatenate((self.v_tanks, self.v_irrs, self.drawdowns, self.p_pumps, [self.soe, self.e_residual, self.k % 144]))
        return observation

    def set_state(self, observation: np.ndarray[np.float32], dQs:list[np.ndarray[np.float32]]=None) -> None:
        """ Set the state of the environment """
        self.v_tanks = observation[:self.n_crops]
        self.v_irrs = observation[self.n_crops:2*self.n_crops]
        self.drawdowns = observation[2*self.n_crops:3*self.n_crops]
        self.soe = observation[4*self.n_crops]
        self.k = int(observation[6*self.n_crops])
        if dQs is not None:
            self.dQs = dQs
        else:
            self.dQs = [np.array([0])] * self.n_crops

    def start(self) -> np.ndarray[np.float32, int]:
        """ Start the model in a certain day of year
        Returns: tuple of (v_tanks, v_irrs, drawdowns, soe, k)"""
        self.k = 0
        self.v_irrs = np.zeros(self.n_crops)
        self.prev_Qps = np.zeros(self.n_crops)
        self.dQs = [np.array([0])] * self.n_crops
        return self.get_observation()

    def get_state(self) -> tuple[
        np.ndarray[np.float32], np.ndarray[np.float32], np.ndarray[np.float32], float, float]:
        """ Get the state of the environment 
        Returns: Tuple of (v_tanks, v_irrs, drawdowns, soe, k)"""

        return self.v_tanks, self.v_irrs, self.drawdowns, self.soe, self.k


def obs_to_array(obs: tuple[list[float], list[float], list[float], float, int]) -> tuple[np.ndarray, int]:
    v_tanks, v_irrs, drawdowns, soe, k = obs
    n = len(v_tanks)  # number of crops
    return np.array(v_tanks + v_irrs + drawdowns + [soe, k]), n

def mg_tuple2array(mg_tuple):
    v_tanks, v_irrs, drawdowns, soe, k = mg_tuple
    mg_array = np.concatenate((v_tanks, v_irrs, drawdowns, [soe, k]))
    return mg_array


def map_action(policy_output: Union[torch.Tensor, np.ndarray]) -> np.ndarray[np.float32]:
    if torch.is_tensor(policy_output):
        return policy_output.detach().cpu().numpy()
    else:
        return policy_output
    
def drawdown(k: int, dQ: Union[np.ndarray, list]):  # drawdown of the well
    """
    Computes the drawdown of the well according the theis equation
    :param k: time instant
    :param dQ: delta flow rate [m3/s2]
    :return: drawdown
    """
    assert k == len(dQ), "The length of dQ should match the number of temporal k points"
    if type(dQ) == list:
        dQ = np.array(dQ)
    l = np.arange(1, k + 1)
    arg = (r_wells ** 2 * S) / (4 * T * (k - l + 1) * 600)
    sum_ = np.dot(dQ, exp1(arg))
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val

def drawdown2(k: int, Q: Iterable):  # drawdown of the well
    """
    Computes the drawdown of the well according the theis equation
    :param k: time instant
    :param Q: flow rate [l/s]
    :return: drawdown
    """

    assert k == len(Q), "The length of dQ should match the number of temporal k points"
    
    l = np.arange(1, k + 1)
    arg = (r_wells ** 2 * S) / (4 * T * (k - l + 1) * 600) 
    arg0 = (r_wells ** 2 * S) / (4 * T * k * 600) if k > 0 else np.inf
    arg = np.concatenate(([arg0], arg))
    dW = np.diff(exp1(arg))
    sum_ = 0.0
    for i in range(k):
        sum_ += Q[i] * dW[i] * 1e-3
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val

