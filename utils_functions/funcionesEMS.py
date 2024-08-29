import pandas as pd
from typing import Union, Tuple
from environments.Data.EMS.EMS_constants import*

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
    for i in range(s_hourly_demand[0]):
        for j in range(s_hourly_demand[1]):
            demand[i, j * 6:(j + 1) * 6] = hourly_demand[i, j]
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
    Pn = 90
    a_fv = -.0045
    Tn = 25
    T_cell = temp + rad / 800 * (Tn - 20)
    return (Pn * rad / 1000.) * (1 + a_fv * (T_cell - Tn))


def follow_ref_rew_1(s, a, s_next) -> np.ndarray:
    """
    Reward function for the follow reference task
    :param s: current state
    :param a: action
    :param s_next: next state
    :return: reward
    """
    reward = 0

    if np.abs(s[0] - s[1]) > np.abs(s[0] - s_next[1]):
        reward = reward + 1

    elif s[1] > s[0] and s_next[1] > s[1]:
            reward = reward - 2

    tank_reward = 0 #(s_next[3])*.8 if s_next[3] <= 5 else 0

    reward = reward + tank_reward

     # unfeasible action penalty
    if s[3] <= 1 and a[0] > 0:
        reward = reward - 5

    if s[3] >= 5 and a[1] > 0:
        reward = reward - 5

    return np.array([reward], dtype=np.float32)


def manage_batteries(SoE: float,
                    P_fv: float,
                    P_demanded: float,
                    P_pump: float) -> Tuple[float, float, float]:
        """
        Choose the power to re/discharge the batteries and computes the next SoE
        :param SoE:
        :param P_fv:
        :param P_demanded:
        :param P_pump:
        :return: Pbat, next_SoE, E_residual
        """
        E_surplus = 0.0
        E_deficit = 0.0

        P_residual = P_fv - P_demanded - P_pump
        if not -Pbat_max <= P_residual <= Pbat_max:  # The surplus is out of the power bounds of the battery
            Pbat = np.clip(P_residual, -Pbat_max, Pbat_max)  # positive for surplus, negative for deficit
            P_not_used = P_residual - Pbat  # positive for surplus, negative for deficit. In case of negative value is P not available

        else:
            P_not_used = 0
            Pbat = P_residual

        delta_SoE = np.max([Pbat, 0]) * n_c * (dt / 3600) + np.min([Pbat, 0]) / n_d * (dt / 3600) # [kWh]
        next_SoE = SoE + delta_SoE

        if SoE_min <= next_SoE <= SoE_max:  # The recharge is done immediately
            Pbat = Pbat
        else:  # The re/discharge is done but there is a surplus/deficit of energy
            E_surplus = next_SoE - SoE_max if next_SoE > SoE_max else 0
            E_deficit = next_SoE - SoE_min if next_SoE < SoE_min else 0

            next_SoE = np.clip(next_SoE, SoE_min, SoE_max)
            if delta_SoE > 0:
                Pbat = np.max([SoE_max - SoE, 0]) / (dt / 3600) / n_c
            else:
                Pbat = np.min([SoE_min - SoE, 0]) * n_d / (dt / 3600)
            #Pbat = np.max([SoE_max - SoE, 0]) / (dt / 3600) / n_c + np.min([SoE_min - SoE, 0]) * n_d / (dt / 3600)

        E_surplus = E_surplus + P_not_used * (dt / 3600) if P_not_used > 0 else E_surplus
        E_deficit = E_deficit + P_not_used / n_d * (dt / 3600) if P_not_used < 0 else E_deficit

        E_residual = E_surplus + E_deficit 

        return Pbat, next_SoE, E_residual