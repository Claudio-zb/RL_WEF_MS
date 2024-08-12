from typing import Any, Union, Tuple, Dict, Callable, List, Iterable

import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt

from gymnasium import spaces
import gymnasium as gym

class WMS_env(gym.Env):

    def __init__(self):
        """First implementation made for only one crop"""
        self.days_since_plantation: int = 0
        self.doy: int = 0
        self.plantation_day: int = 0  # doy
        self.harvest_day: int = 0
        # observations
        # state
        self.depletion: float = 0  # [mm]

        # observed disturbances
        # # weather
        self.weather_data: pd.DataFrame = pd.read_csv("environments/processed_data.csv")
        self.precipitation: float = 0
        self.wind_speed: float = 0
        self.max_temperature: float = 0
        self.min_temperature: float = 0
        self.RH_max_temperature: float = 0
        self.RH_min_temperature: float = 0
        self.solar_radiation: float = 0

        self.crop_evapotranspiration: float = 0  # [mm]
        self.potential_crop_evapotranspiration: float = 0  # [mm]
        self.ref_evapotranspiration: float = 0  # [mm]

        # # crop variables
        self.root_depth: float = 0  # [m]
        self.Kc: float = 1.0
        self.Ks: float = 1.0

        # parameters
        self.layers = []
        self.layers.append({'depth': 0.3, 'theta_fc': 0.3, 'theta_wp': 0.1, 'Ks': 1.0})
        self.crop_parameters = tomato
        self.geological_parameters = jose_painecura
        self.AWC = jose_painecura["field_capacity"] - jose_painecura["wilting_point"]
        self.MAD = self.crop_parameters["MAD"]
        self.TAW = 1
        self.per_depletion = 50
    def reset(self, *, seed: int = None, options=None) -> Tuple[Any, Dict[str, Any]]:
        if options is None:
            options = {}
        self.days_since_plantation = 1
        self.doy = self.crop_parameters["plantation_day"]  # lets see
        self.root_depth = self.get_root_depth(self.days_since_plantation)

        self.update_climate_data(self.doy)
        self.ref_evapotranspiration = self.daily_ET0()
        self.Kc = self.get_Kc(self.days_since_plantation)
        self.potential_crop_evapotranspiration = self.ref_evapotranspiration * self.Kc

        self.TAW = self.AWC * self.root_depth
        self.Ks = self.update_Ks()
        self.crop_evapotranspiration = self.potential_crop_evapotranspiration * self.Ks
        self.depletion = self.per_depletion * self.TAW / 100

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[ndarray, ndarray, bool, bool, Dict[str, Any]]:

        self.days_since_plantation += 1
        self.doy = max(1, (self.doy + 1) % 365)
        self.TAW = self.AWC * self.root_depth
        self.depletion = np.clip(self.depletion - self.precipitation + self.potential_crop_evapotranspiration/1000, 0, self.TAW)

        self.root_depth = self.get_root_depth(self.days_since_plantation)

        self.update_climate_data(self.doy)
        self.ref_evapotranspiration = self.daily_ET0()
        self.Kc = self.get_Kc(self.days_since_plantation)
        self.potential_crop_evapotranspiration = self.ref_evapotranspiration * self.Kc

        self.Ks = self.update_Ks()
        self.crop_evapotranspiration = self.potential_crop_evapotranspiration * self.Ks




        return self._get_observation(), {}

    def _get_observation(self) -> ndarray:
        return np.array([self.root_depth,
                         self.potential_crop_evapotranspiration,
                         self.ref_evapotranspiration,
                         self.Kc,
                         self.crop_evapotranspiration,
                         self.depletion,
                         self.TAW,
                         self.Ks])

    def get_root_depth(self, t: int) -> float:
        """Root depth as a function of time
        params
        t: time since plantation day [days]
        """

        root_depth_init = self.crop_parameters["root_depth_init"]
        root_depth_max = self.crop_parameters["root_depth_max"]

        stages = self.crop_parameters["stages_duration"]
        t0 = stages[0]
        t1 = stages[0] + stages[1]

        if t < t0:
            return root_depth_init
        elif t < t1:
            return root_depth_init + (root_depth_max - root_depth_init) * (t - t0) / (t1 - t0)
        else:
            return root_depth_max

    def get_Kc(self, t: int) -> float:
        """Crop coefficient as a function of time
        params
        t: time since plantation day [days]
        """
        stages = self.crop_parameters["stages_duration"]
        t0 = stages[0]
        t1 = stages[0] + stages[1]
        t2 = stages[0] + stages[1] + stages[2]
        t3 = stages[0] + stages[1] + stages[2] + stages[3]

        Kc = self.crop_parameters["Kc"]
        if t < t0:
            return Kc[0]
        elif t < t1:
            return Kc[1]
        elif t < t2:
            return Kc[2]
        elif t < t3:
            return Kc[3]
        else:
            return 0

    def daily_ET0(self):
        params = self.geological_parameters
        I_s = self.solar_radiation * 3.6 / 1000  # [kWh] -> [MJ]

        phi = np.pi * params["latitude"] / 180
        dr = 1 + 0.033 * np.cos(2 * np.pi * self.doy / 365)
        gamma = 0.409 * np.sin(2 * np.pi * self.doy / 365 - 1.39)
        w_s = np.arccos(-np.tan(phi) * np.tan(gamma))
        R_a = (24 * 60 / np.pi) * 0.082 * dr * (
                w_s * np.sin(gamma) * np.sin(phi) + np.cos(phi) * np.cos(gamma) * np.sin(w_s))
        R_SO = (0.75 + 2e-5 * params["elevation"]) * R_a
        albedo = .23
        Rn_SO = I_s * (1 - albedo)  # change alpha_inv for actual albedo value
        e_s_min = e_s(self.min_temperature)
        e_s_max = e_s(self.max_temperature)
        e_s_average = (e_s(self.min_temperature) + e_s(self.max_temperature)) * .5
        e_a = (self.RH_min_temperature * e_s_min + self.RH_max_temperature * e_s_max) / 100 / 2
        sigma = 4.901 * 1e-9
        RnL = sigma * ((self.min_temperature + 273) ** 4 + (self.max_temperature + 273) ** 4) / 2 * (
                0.34 - 0.14 * np.sqrt(e_a)) * (
                      1.35 * (I_s / R_SO) - 0.35)
        R_n = Rn_SO - RnL
        P = 101.3 * ((293 - 0.0065 * 1_524) / 293) ** 5.26
        psi_const = 0.000665 * P
        Tmean = (self.min_temperature + self.max_temperature) / 2
        delta = 2_504 * np.exp(17.27 * Tmean / (Tmean + 237.3)) / (Tmean + 237.3) ** 2
        num = 0.408 * delta * R_n + psi_const * 900 / (Tmean + 273) * self.wind_speed * (e_s_average - e_a)
        den = delta + psi_const * (1 + 0.34 * self.wind_speed)
        return num / den
        pass

    def update_climate_data(self, doy: int):
        data = self.weather_data.loc[self.weather_data["doy"] == doy]
        #self.precipitation = data["precipitation"]
        self.wind_speed = data["w_speed"].iloc[0]
        self.max_temperature = data["t_max"].iloc[0]
        self.min_temperature = data["t_min"].iloc[0]
        self.RH_max_temperature = data["RH_tmax"].iloc[0]
        self.RH_min_temperature = data["RH_tmin"].iloc[0]
        self.solar_radiation = data["rad"].iloc[0]

        return

    def update_Ks(self):
        """Update Ks as a function of the depletion"""

        if self.depletion <= self.MAD*self.TAW:
            Ks = 1.0
        else:
            Ks = (self.TAW - self.depletion) / (self.TAW - self.MAD*self.TAW)
        return np.clip(Ks, 0, 1)


# parameters of cultives
tomato = {"plantation_day": 295,  # [doy]
          "stages_duration": [30, 40, 40, 25],  # [days]
          "Kc": [0.6, 0.75, 1.15, 0.6],
          "MAD": 0.5,
          "root_depth_init": 0.2*0.8,  # [m]
          "root_depth_max": 1.1*0.8,  # [m]
          "height_max": 0.6,  # [m]
          "production_max": 86910,  # [kg/ha]
          "price": 220  # [$/kg]
          }

# geographical parameters

jose_painecura = {"latitude": -38.67948359212895,
                  "longitude": -73.47621873681696,  # [deg]
                  "elevation": 0,  # [m],
                  "field_capacity": 0.25,
                  "wilting_point": 0.1,
                  }


def e_s(temp: float) -> float:
    """
    partial pressure
    """
    return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
