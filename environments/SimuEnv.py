from environments.WMS_env import CultivateEnv
from environments.EMS_env import EnergyWaterMG
from typing import Callable
import pandas as pd
import numpy as np
from environments.utils.funcionesEMS import *


class SimuEnv:
    def __init__(self, irrigation_policy: Callable, ems_policy: Callable):

        self.microgrid_env: EnergyWaterMG = EnergyWaterMG()
        self.cultivate_env: CultivateEnv = CultivateEnv()

        self.irrigation_policy: Callable = irrigation_policy
        self.ems_policy: Callable = ems_policy

        self.daily_weather_data: pd.DataFrame = pd.read_csv("environments/Data/WMS/weather_data.csv")
        self.ten_min_weather_data: pd.DataFrame = pd.read_csv("environments/Data/EMS/calan_2006.csv")
        self.ten_min_demand: np.ndarray = get_demand()
        self.days_since_started: int = 0
        self.doy: int = 0
        self.last_simulation_data: dict = {}

    def start(self, doy: int):
        self.doy = doy
        self.days_since_started = 1
        return self.days_since_started

    def run(self, init_doy: int, total_days: int):

        self.start(init_doy)

        cultivate_obs, cultivate_info = self.cultivate_env.start(self.doy)
        mg_obs, mg_info = None, None
        done = False
        while not done:

            # Get the action from the policies
            weather_data = self.update_daily_weather(self.doy)
            v_reqs = self.irrigation_policy(cultivate_obs)

            if mg_info is None:
                mg_obs = self.microgrid_env.start(self.doy, v_reqs)
            for i in range(144):
                action = self.ems_policy(mg_obs)
                disturbances = self.get_disturbances(0, i)
                mg_obs = self.microgrid_env.next_step((1, action), disturbances)

            v_irrs = mg_obs[1]
            cultivate_obs = self.cultivate_env.step(v_irrs, weather_data)

            self.doy = np.clip((self.doy + 1) % 365, 1, 365)
            self.days_since_started += 1

            if self.days_since_started >= total_days:
                done = True
        return

    def update_daily_weather(self, doy: int):

        data = self.daily_weather_data.loc[self.daily_weather_data["doy"] == doy]
        precipitation = data["precipitation"].values[0]
        try:
            ET0 = data["ET0"].iloc[0]
        except KeyError:
            ET0 = np.nan
        if np.isnan(ET0):
            wind_speed = data["w_speed"].iloc[0]
            max_temperature = data["t_max"].iloc[0]
            min_temperature = data["t_min"].iloc[0]
            RH_max_temperature = data["RH_tmax"].iloc[0]
            RH_min_temperature = data["RH_tmin"].iloc[0]
            solar_radiation = data["rad"].iloc[0]
            ET0 = data["ET0"].iloc[0]
        else:
            wind_speed = np.nan
            max_temperature = np.nan
            min_temperature = np.nan
            RH_max_temperature = np.nan
            RH_min_temperature = np.nan
            solar_radiation = np.nan

        weather = {"precipitation": precipitation, "ET0": ET0, "wind_speed": wind_speed,
                   "max_temperature": max_temperature, "min_temperature": min_temperature,
                   "RH_max_temperature": RH_max_temperature, "RH_min_temperature": RH_min_temperature,
                   "solar_radiation": solar_radiation}

        return weather

    def get_disturbances(self, doy, d_instant) -> np.ndarray:
        """
        vo sai
        """
        temp_and_rad = self.ten_min_weather_data.iloc[doy + d_instant][['temp', 'dir']].values
        p_pv = solar_power(temp_and_rad[1], temp_and_rad[0])
        p_d = self.ten_min_demand[(144 * doy + d_instant) % len(self.ten_min_demand)]
        disturbances = np.array([p_pv, p_d])
        return disturbances

    def get_simu_data(self):
        return self.last_simulation_data


def isDone(mg_obs, cultivate_obs) -> bool:
    return False
