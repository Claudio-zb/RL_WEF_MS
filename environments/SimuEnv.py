from environments.WMS_env import CultivateEnv
from environments.EMS_env import MicroGridEnv
from typing import Callable
import pandas as pd
import numpy as np
from environments.utils.funcionesEMS import *


class SimuEnv:
    def __init__(self, irrigation_policy: Callable, ems_policy: Callable):
        self.microgrid_env = MicroGridEnv()
        self.cultivate_env = CultivateEnv()
        self.irrigation_policy = irrigation_policy
        self.ems_policy = ems_policy
        self.weather_data: pd.DataFrame = pd.read_csv("environments/Data/WMS/weather_data.csv")
        self.days_since_plantation = 0

    def start(self, doy: int):
        cultivate_obs, cultivate_info = self.cultivate_env.reset(doy)
        self.days_since_plantation = 1

        return None, cultivate_obs

    def run(self):
        doy = 1
        mg_obs, cultivate_obs = self.start(doy)
        done = False
        while not done:
            # Get the action from the policies
            weather_data = self.update_daily_weather(doy)

            irrigation = self.irrigation_policy(cultivate_obs)
            cultivate_obs = self.cultivate_env.step(irrigation, weather_data)
            self.days_since_plantation += 1
            done = self.days_since_plantation >= 200

        return mg_obs, cultivate_obs

    def update_daily_weather(self, doy: int):

        data = self.weather_data.loc[self.weather_data["doy"] == doy]
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


def isDone(mg_obs, cultivate_obs) -> bool:
    return False
