import pandas as pd
import numpy as np
from typing import Union

PATH = r'C:\Users\wenap\PycharmProjects\PPO_project\Data\EMS'


def get_demand() -> np.ndarray:
    """
    Read the demand data from the csv file and returns it as a numpy array
    :return: Demand data as a numpy array
    """

    hourly_demand = pd.read_csv(PATH + r'\consumption.csv', sep=',', decimal='.', index_col=0)
    hourly_demand = hourly_demand.values
    s_hourly_demand = hourly_demand.shape
    demand = np.zeros((s_hourly_demand[0], s_hourly_demand[1] * 6))
    dt = 600
    for i in range(s_hourly_demand[0]):
        for j in range(s_hourly_demand[1]):
            demand[i, j * 6:(j + 1) * 6] = hourly_demand[i, j] * dt / 3600
    demand = demand.flatten()
    return demand


def get_temperatura(season: str = 'inv') -> np.ndarray:
    """
    Read the temperature data from the csv file and returns it as a numpy array

    :param season: 'ver' for summer and 'inv' for winter
    :return: Temperature data as a numpy array
    """
    if season == 'ver':
        pass
    else:
        pass
    temperatura = pd.read_csv(PATH + r'\data_temp.csv')
    temperatura = temperatura.interpolate().values.flatten()
    return temperatura


def get_rad(season: str = 'inv') -> np.ndarray:
    """
    Read the radiation data from the csv file and returns it as a numpy array

    :param season: 'ver' for summer and 'inv' for winter
    :return: Radiation data as a numpy array

    """
    if season == 'ver':
        pass
    else:
        pass
    rad = pd.read_csv(PATH + r'\data_rad.csv')
    rad = rad.values.flatten()
    return rad


def get_ref() -> np.ndarray:
    """Read the references data from the csv file and returns it as a numpy array"""
    refs = pd.read_csv(PATH + r'\v_refs.csv')
    refs = refs.values.flatten()
    return refs


def get_reward(E_residual,
               Irr_prev,
               Irr,
               V_Irr_prev,
               V_Irr,
               V_ref,
               Qp_prev, Qp,
               Pbat_prev, Pbat) -> float:
    """
    Computes the reward for the current state of the system
    :param E_residual: Difference between the energy produced and the energy consumed
    :param SoE_penalty: Penalty for trying to move to an unfeasible SoE state
    :param Irr_prev: Previous irrigation [%] in the first day
    :param Irr: Irrigation [%] in the first day
    :param V_Irr_prev: Previous water volume fulfilled the first day
    :param V_Irr: Water volume fulfilled the first day
    :param V_ref: Water volume demand the first day
    :param Qp_prev: Previous power of the pump [%]
    :param Qp: Power of the pump [%]
    :param Pbat_prev: Previous power of the battery [%]
    :param Pbat: Power of the battery [%]
    :return: reward r(t)
    """
    I_max = 100  # 1 / 1000
    I_min = 0
    Q_max = 100  # 1 / 1000
    Q_min = 0

    d_Irr = Irr - Irr_prev
    d_Qp = Qp - Qp_prev
    d_Pbat = Pbat - Pbat_prev

    if E_residual > 0:
        E_sell = E_residual
        E_buy = 0
    else:
        E_sell = 0
        E_buy = -E_residual

    economic_component = (25 * E_sell - 100 * E_buy) / 100
    actuator_penalty = (1/50)*(1e1 * d_Irr**2 + 1e1 * d_Qp ** 2 + 1 * d_Pbat ** 2)
    follow_demand_reward = 200*np.exp(-0.1 * (V_Irr - V_ref)**2)

    constraints_penalty = 0

    # no surpassing demand conditions

    if V_ref < V_Irr_prev < V_Irr:
        constraints_penalty = constraints_penalty + 1e2

    # no surpassing actuator conditions

    if Irr < I_min or I_max < Irr:
        constraints_penalty = constraints_penalty + 1e2

    if Qp < Q_min or Qp > Q_max:
        constraints_penalty = constraints_penalty + 1e2

    reward = economic_component - follow_demand_reward - constraints_penalty - actuator_penalty

    return reward


def solar_power(rad: Union[float, np.ndarray], temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the solar power in kW given the radiation in W/m2 and the temperature in C.

    :parameter rad: Radiation in W/m2
    :parameter temp: Temperature in C
    :return: Solar power in kW

    """
    Pn = 90 * 600 / 3600
    a_fv = -.0045
    Tn = 25
    T_cell = temp + rad / 800 * (Tn - 20)
    return Pn * rad / 1000. * (1 + a_fv * (T_cell - Tn))
