from environments.WMS_env import Cultivates
from environments.EMS_env import EnergyWaterMG, RuleBasedEMS
from typing import Callable
import pandas as pd
import numpy as np
from environments.utils.funcionesEMS import *
import copy
from matplotlib import pyplot as plt


class SimuEnv:
    def __init__(self, irrigation_policy: Callable, ems_policy: RuleBasedEMS):

        self.microgrid_env: EnergyWaterMG = EnergyWaterMG()
        self.cultivate_env: Cultivates = Cultivates()

        self.irrigation_policy: Callable = irrigation_policy
        self.ems_policy: RuleBasedEMS = ems_policy

        self.daily_weather_data: pd.DataFrame = pd.read_csv("environments/Data/WMS/weather_data.csv")
        self.ten_min_weather_data: pd.DataFrame = pd.read_csv("environments/Data/EMS/calan_2006.csv")
        self.ten_min_demand: np.ndarray = get_demand()
        self.days_since_started: int = 0
        self.doy: int = 0
        self.last_simulation_data: dict = {}
        self.soil_data: list = []

    def start(self, doy: int):
        self.doy = doy
        self.days_since_started = 1
        self.last_simulation_data = {}
        return self.days_since_started

    def run(self, init_doy: int, total_days: int):

        self.start(init_doy)

        cultivate_obs_hist, mg_obs_hist = [], []

        cultivate_obs, cultivate_info = self.cultivate_env.start()
        mg_obs = None
        prev_mg_obs = None
        done = False
        v_reqs, v_irrs = None, None
        v_reqs_hist, v_irrs_hist = [], []
        while not done:

            # Get the action from the policies
            weather_data = self.update_daily_weather(self.doy)
            v_reqs = self.irrigation_policy(cultivate_obs)*1000
            v_reqs_hist.append(v_reqs)

            if mg_obs is None:
                mg_obs = self.microgrid_env.start(self.doy)
            observations = []
            for i in range(144):
                prev_mg_obs = copy.deepcopy(mg_obs)
                disturbances = self.get_disturbances(0, i)
                action = self.ems_policy.get_action(mg_obs, v_reqs, disturbances)
                mg_obs = self.microgrid_env.next_step(action)
                observations.append([mg_obs[0][0], mg_obs[1][0], mg_obs[2][0], mg_obs[3], mg_obs[4]])
            observations = np.array(observations)
            v_irrs = prev_mg_obs[1]
            v_irrs_hist.append(v_irrs)
            mg_obs_hist.append(prev_mg_obs)

            cultivate_obs = self.cultivate_env.step(v_irrs, weather_data)
            cultivate_obs_hist.append(copy.deepcopy(cultivate_obs))

            self.doy = np.clip((self.doy + 1) % 365, 1, 365)
            self.days_since_started += 1

            if self.days_since_started >= total_days:
                done = True
        self.soil_data = [crop.soil.get_hist_data() for crop in self.cultivate_env.crops]
        # self.crop_data = [crop.ge for crop in self.cultivate_env.crops]
        self.last_simulation_data = {"cultivate_obs": cultivate_obs_hist,
                                     "mg_obs": mg_obs_hist,
                                     "wms_actions": v_reqs_hist,
                                     "ems_actions": v_irrs_hist}
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

    def get_disturbances(self, doy: int, day_instant: int) -> np.ndarray:
        """
        vo sai
        """
        temp_and_rad = self.ten_min_weather_data.iloc[doy + day_instant][['temp', 'dir']].values
        p_pv = solar_power(temp_and_rad[1], temp_and_rad[0])
        p_d = self.ten_min_demand[(144 * doy + day_instant) % len(self.ten_min_demand)]
        disturbances = np.array([p_pv, p_d])
        return disturbances

    def get_simu_data(self):
        return self.last_simulation_data

    def get_soil_data(self):
        return self.soil_data

def isDone(mg_obs, cultivate_obs) -> bool:
    return False
