import pandas as pd
import numpy as np
from typing import Union

PATH = r'C:\Users\wenap\PycharmProjects\PPO_project\Data\EMS'

delay = 6 * 6  # 6 hours in 10 minutes intervals


def get_demand() -> np.ndarray:
    """
    Read the demand data from the csv file and returns it as a numpy array
    :return: Demand data as a numpy array [kWh]
    """

    hourly_demand = np.genfromtxt("./Data/EMS/consumption.csv", delimiter=',')
    s_hourly_demand = hourly_demand.shape
    demand = np.zeros((s_hourly_demand[0], s_hourly_demand[1] * 6))
    dt = 600
    for i in range(s_hourly_demand[0]):
        for j in range(s_hourly_demand[1]):
            demand[i, j * 6:(j + 1) * 6] = hourly_demand[i, j]  # * dt / 3600
    demand = demand.flatten()
    return demand


def get_temperatura(season: str = 'ver') -> np.ndarray:
    """
    Read the temperature data from the csv file and returns it as a numpy array

    :param season: 'ver' for summer and 'inv' for winter
    :return: Temperature data as a numpy array
    """
    if season == 'ver':
        file_path = "./Data/EMS/data_temp_ver.csv"
    else:
        file_path = "./Data/EMS/data_temp_inv.csv"
    temperatura = pd.read_csv(file_path)
    temperatura = temperatura.interpolate().values.flatten()
    return temperatura[delay:]


def get_rad(season: str = 'ver') -> np.ndarray:
    """
    Read the radiation data from the csv file and returns it as a numpy array

    :param season: 'ver' for summer and 'inv' for winter
    :return: Radiation data as a numpy array

    """
    if season == 'ver':
        file_path = "./Data/EMS/data_rad_ver.csv"
    else:
        file_path = "./Data/EMS/data_rad_inv.csv"

    rad = pd.read_csv(file_path)
    rad = rad.values.flatten()
    return rad[delay:]


def get_ref() -> np.ndarray:
    """Read the references data from the csv file and returns it as a numpy array"""
    refs = pd.read_csv('./Data/EMS/v_refs.csv')
    refs = refs.values.flatten()
    return refs


def solar_power(rad: Union[float, np.ndarray], temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the solar power in kW given the radiation in W/m2 and the temperature in C.

    :parameter rad: Radiation in W/m2
    :parameter temp: Temperature in °C
    :return: Solar power in kW

    """
    Pn = 90  # 90 * (600 / 3600)
    a_fv = -.0045
    Tn = 25
    T_cell = temp + rad / 800 * (Tn - 20)
    return (Pn * rad / 1000.) * (1 + a_fv * (T_cell - Tn))


def get_reward(E_surplus,
               E_deficit,
               Irr_prev,
               Irr,
               V_Irr,
               V_ref,
               Qp_prev, Qp) -> float:
    """
    Computes the reward for the current state of the system
    :param E_surplus: Energy surplus [kWh]
    :param E_deficit: Energy deficit [kWh]
    :param Irr_prev: Previous irrigation [%] in the first day
    :param Irr: Irrigation [%] in the first day
    :param V_Irr_prev: Previous water volume fulfilled the first day [m3]
    :param V_Irr: Water volume fulfilled the first day [m3]
    :param V_ref: Water volume demand the first day [m3]
    :param Qp_prev: Previous power of the pump [%]
    :param Qp: Power of the pump [%]
    :param Pbat_prev: Previous power of the battery [%]
    :param Pbat: Power of the battery [%]
    :return: reward r(t)
    """

    d_Irr = (Irr - Irr_prev)/100  # [l/s]
    d_Qp = (Qp - Qp_prev)/100  # [l/s]

    economic_component = 25 * E_surplus - 100 * E_deficit
    w_ns = np.max([0.0, V_ref - V_Irr])  # Pending demand
    w_ex = np.max([0.0, -(V_ref - V_Irr)])  # Excedent

    reward = (30*np.exp(-0.05 * (w_ns**2 + 5*w_ex**2 + d_Irr**2 + d_Qp**2))
              + 20*np.exp(-0.1 * (V_ref - V_Irr)**2)
              + economic_component/10)
    return reward
