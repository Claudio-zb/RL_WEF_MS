from environments.WMS_policies import IrrigationPolicy 
from environments.Cultivates import Cultivates
from environments.EMS_policies import PumpingPolicy
from environments.EMS_env import EnergyWaterMG
from typing import Callable
import pandas as pd
import numpy as np
from environments.utils.funcionesEMS import *
import copy
import random


class SimuEnv:
    def __init__(self, irrigation_policy: Callable, ems_policy: PumpingPolicy):

        self.microgrid_env: EnergyWaterMG = EnergyWaterMG()
        self.cultivate_env: Cultivates = Cultivates()

        self.irrigation_policy: IrrigationPolicy = irrigation_policy
        self.ems_policy: PumpingPolicy = ems_policy

        self.global_weather_data: pd.DataFrame = pd.read_csv("environments/Data/WMS/extracted_data.csv", index_col=None)
        self.daily_weather_data: pd.DataFrame = None
        
        self.ten_min_weather_data: np.ndarray = solar_power(get_rad("ver"), get_temperatura("ver"))
        self.ten_min_demand: np.ndarray = get_demand()
        
        self.pv_daily_profile: np.ndarray = None
        self.prev_pv_daily_profile: np.ndarray = None
        self.pd_daily_profile: np.ndarray = None

        self.days_since_started: int = 0
        self.doy: int = 0
        self.year: int = None
        self.last_simulation_data: dict = {}
        self.soil_data: list[dict] = []

        self.surface_area: float = 1000  # [m2]
        self.update_10_min_weather()
        self.update_10_min_weather()

    def start(self, doy: int):
        self.year = 2010
        self.daily_weather_data = self.global_weather_data[self.global_weather_data["year"] == self.year].copy()
        self.doy = doy
        self.days_since_started = 1
        self.last_simulation_data = {}
        return self.days_since_started

    def run(self, init_doy: int, total_days: int):

        self.start(init_doy)

        cultivate_obs_hist, end_of_day_samples = [], []
        daily_weather_data = self.daily_weather_data[self.daily_weather_data["doy"] == self.doy].iloc[0].to_dict()

        cultivate_obs, doy = self.cultivate_env.start()
        mg_obs, prev_mg_obs = None, None
        done = False
        v_reqs, v_irrs = None, None
        v_reqs_hist, v_irrs_hist = [], []
        observations = []
        mg_dis_hist = []
        actions = []
        mg_obs = self.microgrid_env.start()
        while not done:

            # Get the action from the policies
            weather_data = self.update_daily_weather(self.doy)
            wms_disturbances = np.array([weather_data["precipitation"], weather_data["ET0"]])
            mm_reqs = self.irrigation_policy.get_action(cultivate_obs, wms_disturbances, doy)  # water requirement [m]
            
            v_reqs = mm_reqs*self.surface_area  # water requirement [m3]
            v_reqs_hist.append(v_reqs[0])
            
            self.update_10_min_weather()

            for i in range(144):
                prev_mg_obs = copy.deepcopy(mg_obs)
                disturbances = self.get_disturbances(self.doy, i)
                mg_dis_hist.append(disturbances)
                action = self.ems_policy.get_action(v_reqs, mg_obs, disturbances)
                mg_obs = self.microgrid_env.next_step(action, disturbances)
                observations.append(mg_obs)
                actions.append(action)
   
            _prev_mg_obs = np.array(prev_mg_obs)

            v_irrs = [prev_mg_obs[1]]
            v_irrs_hist.append(v_irrs)
            end_of_day_samples.append(_prev_mg_obs)

            cultivate_obs, doy = self.cultivate_env.step(v_irrs, weather_data)
            cultivate_obs_hist.append(copy.deepcopy(cultivate_obs))

            self.doy = np.clip((self.doy + 1) % 365, 1, 365)
            self.days_since_started += 1

            if self.days_since_started >= total_days:
                done = True
        end_of_day_samples = np.array(end_of_day_samples)
        actions = np.array(actions)
        q_ps = actions[:,0]
        q_is = actions[:,1]
        self.soil_data = [crop.soil.get_hist_data() for crop in self.cultivate_env.crops]
        # self.crop_data = [crop.ge for crop in self.cultivate_env.crops]
        self.last_simulation_data = {"cultivate_obs": cultivate_obs_hist,
                                     "mg_obs": observations, #mg_obs_hist,
                                     "mg_dis": mg_dis_hist,
                                     "wms_actions": np.array(v_reqs_hist),
                                     "end_of_day_samples": end_of_day_samples,
                                     "qp_actions": q_ps,
                                     "qi_actions": q_is}
        return

    def update_daily_weather(self, doy: int) -> dict:

        #np.random.seed(seed)
        #random.seed(seed)

        data = self.daily_weather_data.loc[self.daily_weather_data["doy"] == doy]
        precipitation = data["precipitation"].values[0]
        try:
            ET0 = data["ET_0"].iloc[0]
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
        Get the P_pv and P_res disturbances for the microgrid.
        """
        p_pv = self.pv_daily_profile[day_instant]  # solar power [kW]
        p_d = self.pd_daily_profile[day_instant]  # demand [kW]
        disturbances = np.array([p_pv, p_d])
        return disturbances

    def get_simu_data(self):
        return self.last_simulation_data

    def get_soil_data(self):
        return self.soil_data
    
    def update_10_min_weather(self):
        """Pick a weather data time series for an entire day.
        It uses mixed-up signals for data augmentation."""
        
        n_days = self.ten_min_weather_data.shape[0] // 144

        j = random.randint(0, n_days - 1) # pick a random day  
        jdex = 144*j # index of the start of a daily profile
        
        i = random.randint(0, n_days-1)
        index = 144*i

        lambda_ = random.uniform(0, .2)

        pv_day_val = self.ten_min_weather_data[jdex:jdex+144]
        other_pv_day_val = self.ten_min_weather_data[index:index+144]

        pd_day_val = self.ten_min_demand[jdex:jdex+144]
        other_pd_day_val = self.ten_min_demand[index:index+144]

        self.pv_daily_profile = pv_day_val*(1-lambda_) + other_pv_day_val*(lambda_)
        self.prev_pv_daily_profile = self.pv_daily_profile
        
        self.pd_daily_profile = pd_day_val*(1-lambda_) + other_pd_day_val*(lambda_)


def isDone(mg_obs, cultivate_obs) -> bool:
    return False
