from WMS_env import CultivateEnv
from EMS_env import ContinousEMSEnv
from typing import Callable
import pandas as pd
import numpy as np


class SimuEnv:
    def __init__(self, irrigation_policy: Callable, ems_policy: Callable):
        self.microgrid_env = ContinousEMSEnv()
        self.cultivate_env = CultivateEnv()
        self.irrigation_policy = irrigation_policy
        self.ems_policy = ems_policy
        self.weather_data: pd.DataFrame = pd.read_csv("environments/weather_data.csv")

    def start(self, doy: int):
        cultivate_obs, cultivate_info = self.cultivate_env.reset(doy)
        mg_obs, mg_info = self.microgrid_env.reset()

        return mg_obs, cultivate_obs

    def run(self):
        doy = 1
        mg_obs, cultivate_obs = self.start(doy)
        done = False
        while not done:
            # Get the action from the policies
            weather_data = self.update_daily_weather(doy)
            mg_action = self.ems_policy(mg_obs)

            irrigations = self.irrigation_policy(cultivate_obs)

            # high speed dynamics
            self.microgrid_env.set_Vreq(irrigations)
            for t in range(0,144):


            mg_obs, mg_reward, mg_done, mg_info = self.microgrid_env.step(mg_action)
            cultivate_obs, cultivate_reward, cultivate_done, cultivate_info = self.cultivate_env.step(cultivate_action)

            done = isDone(mg_obs, cultivate_obs)

        return mg_obs, cultivate_obs

    def update_daily_weather(self, doy: int):

        data = self.weather_data.loc[self.weather_data["doy"] == doy]
        precipitation = data["precipitation"].values[0]
        try:
            ET0 = data["ET0"].iloc[0]
        except KeyError:
            ET0 = np.nan
        wind_speed = data["w_speed"].iloc[0]
        max_temperature = data["t_max"].iloc[0]
        min_temperature = data["t_min"].iloc[0]
        RH_max_temperature = data["RH_tmax"].iloc[0]
        RH_min_temperature = data["RH_tmin"].iloc[0]
        solar_radiation = data["rad"].iloc[0]
        ET0 = data["ET0"].iloc[0]

        weather = {"precipitation": precipitation, "ET0": ET0, "wind_speed": wind_speed,
                   "max_temperature": max_temperature, "min_temperature": min_temperature,
                   "RH_max_temperature": RH_max_temperature, "RH_min_temperature": RH_min_temperature,
                   "solar_radiation": solar_radiation}

        return weather


def isDone(mg_obs, cultivate_obs) -> bool:
    return False
