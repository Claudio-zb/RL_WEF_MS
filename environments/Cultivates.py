from typing import Any, Union
import numpy as np

from environments.Data.WMS.WMS_profile import *

class Cultivates:
    """Class able to wrap multiple crops under the same meteorological variables to simulate a farm"""

    def __init__(self, crop_params:list[dict] = [potato], geological_params: dict = jose_painecura):
        self.crops: list[Crop] = [crop_from_dict(crop_param) for crop_param in crop_params]
        # the simulation will start in the first plantation day
        self.doy: int = min([crop.plantation_day for crop in self.crops])

        self.wind_speed: float = 0.0
        self.max_temperature: float = 0.0
        self.min_temperature: float = 0.0
        self.RH_max_temperature: float = 0.0
        self.RH_min_temperature: float = 0.0
        self.solar_radiation: float = 0.0
        self.precipitation: float = 0.0
        self.geological_parameters = geological_params
        self.ET0: float = 0.0


    def start(self) -> tuple[dict[str, Any], int]:
        """
        Starts the simulation of crops
        :return: a dictionary with the initial state of the crops and the day of year
        """
        self.doy = min([crop.plantation_day for crop in self.crops])
        #self.set_climate_data(init_weather_info)
        for crop in self.crops:
            crop.reset()
            if crop.plantation_day == self.doy:
                crop.start()
        return self.get_obs(), self.doy

    def set_climate_data(self, climate_data: dict):
        """ Set the climate data for the current day
        """

        try:
            self.solar_radiation = climate_data["solar_radiation"]
            self.max_temperature = climate_data["max_temperature"]
            self.min_temperature = climate_data["min_temperature"]
            self.RH_max_temperature = climate_data["RH_max_temperature"]
            self.RH_min_temperature = climate_data["RH_min_temperature"]
            self.wind_speed = climate_data["wind_speed"]
            self.ET0 = self.get_ET0()
        except KeyError:
            self.ET0 = climate_data["ET_0"]

        assert isinstance(self.ET0, float) or isinstance(self.ET0, np.floating)
        
        self.precipitation = climate_data["precipitation"]/1000  # [mm] -> [m]

        return
    
    def set_state(self, state: dict, doy: int):
        """Set the state of the crops"""
        self.doy = int(doy)
        for crop in self.crops:
            crop._is_active = True if state[crop.crop_name][-1] > 0 else False # is active if days since plantations >= 1
            crop.set_state(state[crop.crop_name], self.doy)
        
        
        
    def step(self, irrigations: Union[list, np.ndarray], weather_data: dict) -> tuple[dict, int]:
        """Performance a new step in the simulation, given an action-disturbance pair
        param: irrigations: the amount of water [m] going in by the evaporation layer
        param: climate_data: a dictionary with the daily weather data
        returns: a dictionary with the current state of active crops """

        self.set_climate_data(weather_data)
        for idx, crop in enumerate(self.crops):
            if crop.plantation_day == self.doy:
                crop.start()
            if crop.is_active():
                infiltrated_water, runoff_water = self.compute_infiltration(irrigations[idx], self.precipitation)
                crop.step(self.ET0, infiltrated_water)
        self.doy = max(1, (self.doy + 1) % 365) 
        return self.get_obs(), self.doy

    def get_hist_data(self):
        """
        Collects the historic data from the
        """
        hist_data = {}
        for crop in self.crops:
            hist_data[crop.crop_name] = crop.get_hist_data()
        return hist_data

    def get_ET0(self) -> float:
        """
        Returns daily reference evapotranspiration
        """
        if self.ET0 is not np.nan:
            return self.ET0
        else:
            return self.estimate_ET0()

    def estimate_ET0(self) -> float:
        """
        Computes an estimation for daily evapotranspiration from weather data [mm/day]
        """
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

    @staticmethod
    def compute_infiltration(irrigation: float, precipitation: float) -> tuple[float, float]:
        """
        :param irrigation: [m3]
        :param precipitation: [mm]
        :return: infiltration [m3], runoff [m3]
        """
        irrigation = irrigation
        assert isinstance(irrigation, (int, float, np.floating))
        precipitation = precipitation/1000  # [mm] -> [m3]
        return irrigation + precipitation, 0.0


    def get_soil_data(self):
        soil_data = [crop.soil.get_hist_data() for crop in self.crops]
        return soil_data

    def get_obs(self) -> dict[str, np.ndarray]:
        """Get the current state of every crop."""
        obs = {}
        for crop in self.crops:
            obs[crop.crop_name] = crop.get_obs()
        return obs

    def set_theta(self, theta: float):
        for crop in self.crops:
            crop.soil.set_theta(theta)

class Crop:

    def __init__(self, crop_name, plantation_day, stages_duration, Kcb,
                 MAD, root_depth_init, root_depth_max, price, f_c_list, Ky_list):
        """First implementation made for only one crop"""

        self.crop_name = crop_name
        self.days_since_plantation: int = 0
        self.doy: int = 0
        self.days_of_water_stress: int = 0
        self.plantation_day: int = plantation_day
        self.harvest_day: int = 0
        self.MAD: float = MAD
        self.price: float = price
        self.root_depth_init: float = root_depth_init
        self.root_depth_max: float = root_depth_max

        self.stages_duration: list[int] = stages_duration
        self.Kcb_list: list[float] = Kcb
        self.f_c_list: list[float] = f_c_list
        self.Ky_list: list[float] = Ky_list

        # observations
        # state

        # observed disturbances
        # # weather
        self.precipitation: float = 0
        self.wind_speed: float = 0
        self.max_temperature: float = 0
        self.min_temperature: float = 0
        self.RH_max_temperature: float = 0
        self.RH_min_temperature: float = 0
        self.solar_radiation: float = 0

        self.crop_evapotranspiration: float = 0  # [mm]
        self.potential_crop_evapotranspiration: float = 0  # [mm]
        self.ref_evapotranspiration: float = 0  # [mm]}
        self.fw: float = 0.0
        self.depletion: float = 0.
        self.raw: float = 0.

        # # crop variables
        self.root_depth: float = self.root_depth_init  # [m]
        self.f_c: float = self.f_c_list[0]
        
        self.Kcb: float = self.Kcb_list[0]
        self.Ky: float = self.Ky_list[0]
        self.Ks: float = 1.0
        self.Ke: float = .5


        self.hist_data: list[np.ndarray] = []

        # parameters
        self.soil: Soil = soil_from_dicts(evp_layer_specs,
                                          [layer_specs, layer_specs2, layer_specs2, layer_specs])

        self.geological_parameters = jose_painecura
        self._is_active: bool = False

        self.taw: float = sum([layer.taw for layer in self.soil.layers])
        self.Kr: float = self.soil.get_Kr()
        self.Ke_bound: float = 1.0
        
        self.relative_yield: float = 1.0

    def __str__(self):
        return f"Crop: {self.crop_name}"


    def reset(self, doy: int = 0):
        

        self.soil.reset()
        del self.hist_data[:]
        self.days_since_plantation = 1
        self.doy = doy
        self.days_of_water_stress = 0
        

    def start(self) -> tuple[Any, dict[str, Any]]:
        """
        Starts the simulation of crop
        """

        self.days_since_plantation = 1
        self.days_of_water_stress = 0
        self._is_active = True
        self.doy = self.plantation_day  # let's see
        obs = self._get_observation()
        self.hist_data.append(obs)
        return obs, {}
    
    def set_state(self, state: np.ndarray[np.floating], doy: int):
        """Set the state of the crop"""
        n_layers = len(self.soil.layers) + 1
        self.soil.set_thetas(state[:n_layers]) # set the water content of the soil
        self.f_c = state[n_layers]
        self.root_depth = state[n_layers + 1]
        self.Ks = state[n_layers + 2]
        self.days_of_water_stress = state[n_layers + 3]
        self.days_since_plantation = state[n_layers + 4]

    def step(self, ET0: float, infiltrated_water: float = 0.0) -> np.ndarray:
        """
        Steps the crop model one day forward
        :param ET0: reference evapotranspiration [mm/day]
        :param infiltrated_water: incoming water [m/day]
        """

        self.days_since_plantation += 1
        self.doy = max(1, (self.doy + 1) % 365)
        self.ref_evapotranspiration = ET0
        # first i update the variables according to state
        self.update_water_uptake()
        self.update_coefficients(infiltrated_water)

        # then i compute the soil dynamics
        evaporation = self.Ke * self.ref_evapotranspiration / 1000  # [mm] -> [m]
        transpiration = self.Kcb * self.Ks * self.ref_evapotranspiration / 1000  # [mm] ->[m]
        self.soil.step(evaporation, infiltrated_water, transpiration)
        # then i grow both rooth and foliar coberture
        self.update_root_depth() # 

        obs = self._get_observation()
        self.hist_data.append(obs)
        # compute in which stage i am

        if self.days_since_plantation == sum(self.stages_duration):
            self._is_active = False

        return self._get_observation()
    
    def update_water_uptake(self):
        """Updates the water uptake percentage of every soil layer according the current state"""
        reversed_layers = self.soil.get_reversed_layers() # top to botton oredered
        cum_depths = np.cumsum(np.array([layer.depth for layer in reversed_layers]))
        
        percentages = [.4, .3, .2, .1]
        i = 0  # quarters idx 
        j = 0  # layers idx
        is_root_fractioned = False
        quarter = self.root_depth/4
        cum_quarters = quarter
        layer_percentage = 0.0
        for layer in reversed_layers:
            layer.set_uptake_percentage(0.0)
        while i < 4:
            if j >= len(reversed_layers):
                break
            if cum_quarters <= cum_depths[j]: # the quarter is inside the layer
                if is_root_fractioned:
                    layer_percentage = layer_percentage + percentages[i]*(1-fraction_of_cuarter)
                    is_root_fractioned = False
                else:
                    layer_percentage = layer_percentage + percentages[i]
                i += 1  # lets iterate over quarters
                cum_quarters += quarter
            else: # the quarter is larger than the current layer
                fraction_of_cuarter = (cum_depths[j] - (cum_quarters - quarter))/quarter
                new_percentage = layer_percentage + percentages[i]*fraction_of_cuarter
                reversed_layers[j].set_uptake_percentage(new_percentage)
                is_root_fractioned = True
                layer_percentage = 0.0
                j += 1 # lets iterate in layers
        try:
            reversed_layers[j].set_uptake_percentage(layer_percentage)
        except:
            pass
        total_percentages = np.sum(np.array([layer.uptake_percentage for layer in reversed_layers]))
        assert np.isclose(total_percentages, 1.0)

    def update_coefficients(self, irrigation: float = 0.0):
        """
        computes the crop coefficients and actual evapotranspiration
        :param irrigation: incoming water [m/day]
        """

        assert isinstance(irrigation, (float, np.floating))

        self.fw = 1.0 if irrigation > 0.0 else 0.8

        # evapotranspiration compute
        self.Kcb, self.Ke, self.Ky = self.get_dual_coefficients(self.days_since_plantation)
        self.potential_crop_evapotranspiration = self.ref_evapotranspiration * (self.Kcb + self.Ke)

        self.Ks, partial_Ks = self.update_Ks()
        self.crop_evapotranspiration = self.ref_evapotranspiration * (self.Ks * self.Kcb + self.Ke)

        self.soil.set_partial_Ks(partial_Ks)
        

    def is_active(self):
        """
        Returns true if the crop is already cultivated :)
        """
        return self._is_active

    def get_hist_data(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """returns a tuple of a dictionary containing the crop historic () data and the soil
        historic data.
        returns: crop_hist_data, soil_hist_data"""
        hist_data = np.array(self.hist_data)
        crop_hist_data = {"f_c": hist_data[:,0],
                          "root_depth": hist_data[:, 1],
                          "ET_p": hist_data[:, 2],
                          "ET_0": hist_data[:, 3],
                          "Kcb": hist_data[:, 4],
                          "ET_a": hist_data[:, 5],
                          "K_s": hist_data[:, 6],
                          "K_e": hist_data[:, 7],
                          "avg_h_c": hist_data[:, -1]}
        return crop_hist_data, self.soil.get_hist_data()

    def _get_observation(self) -> np.ndarray:
        """
        Returns the internal state of crop and store it in buffer.
        returns : root depth, ETP, ET0, Kcb, ETR, Ks
        """
        obs = np.array([self.f_c,
                        self.root_depth,
                        self.potential_crop_evapotranspiration,
                        self.ref_evapotranspiration,
                        self.Kcb,
                        self.crop_evapotranspiration,
                        self.Ks,
                        self.Ke,
                        self.Kr,
                        self.Ke_bound, 
                        self.soil.get_avg_hc()])
        return obs

    def get_obs2(self) -> np.ndarray:
        """
        Get crop observation to upper level
        returns: depletion, readily water available and mad
        """
        percent_depletion = self.depletion / self.taw
        return np.array([percent_depletion, self.raw, self.MAD])

    def get_obs(self) -> np.ndarray:
        """Get the current state of the crop: soil moistures, foliar coberture, root depth, Ks, days of water stress and days since plantation"""
     
        soil_moistures = self.soil.get_thetas()
        root_depth_and_ks = np.array([self.f_c, self.root_depth, self.Ks, self.days_of_water_stress, self.days_since_plantation])
        return np.concatenate((soil_moistures, root_depth_and_ks))

    def update_root_depth(self):
        """
        Updates the root depth as a function of time. It also updates the percentage water uptake. 
        params
        """
        t = self.days_since_plantation
        t0 = self.stages_duration[0]
        t1 = self.stages_duration[0] + self.stages_duration[1]
        

        # update root depth
        avg_hc = self.soil.get_avg_hc()
        growth_bonus = 0.0

        if self.Ks < 1.0 - 0.01:
            self.days_of_water_stress += 1
        else:
            self.days_of_water_stress = 0

        if self.days_of_water_stress >= 3:
            avg_hc = self.soil.get_avg_hc()
            growth_bonus = np.log(np.abs(avg_hc))/375
        
        if t <= t0:
            self.root_depth = self.root_depth_init
        elif t <= t1:
            growth_rate = (self.root_depth_max - self.root_depth_init) / (t1 - t0) 
            self.root_depth = np.min([self.root_depth + growth_rate + growth_bonus, self.root_depth_max*1.2]) 
        return

    def get_dual_coefficients(self, t: int) -> tuple[float, float, float]:
        """
        Dual Crop coefficient as a function of time
        params
        t: time since plantation day [days]
        """
        stages = self.stages_duration
        t0 = stages[0]
        t1 = stages[0] + stages[1]
        t2 = stages[0] + stages[1] + stages[2]
        t3 = stages[0] + stages[1] + stages[2] + stages[3]

        if t <= t0:  # initial stage
            Kcb = self.Kcb_list[0]
            Ky = self.Ky_list[0]
        elif t <= t1:  # crop development
            Kcb = (self.Kcb_list[1] - self.Kcb_list[0]) / (t1 - t0) * (t - t0) + self.Kcb_list[0]
            self.f_c = self.f_c + (self.f_c_list[1] - self.f_c_list[0]) / (t1 - t0)*self.Ks
            Ky = self.Ky_list[1]
        elif t <= t2:  # mid-season
            Kcb = self.Kcb_list[1]
            Ky = self.Ky_list[2]
        elif t <= t3:  # late season
            Kcb = (self.Kcb_list[2] - self.Kcb_list[1]) / (t3 - t2) * (t - t2) + self.Kcb_list[1]
            self.f_c = np.max([self.f_c + (self.f_c_list[2] - self.f_c_list[1]) / (t3 - t2), 0.1])
            Ky = self.Ky_list[3]
        else:  # goodbye
            Kcb, self.f_c = 0.0, 0.0
            Ky = 0.0

        Kc_max = max(1.2, Kcb + .05)
        self.Kr = self.soil.get_Kr()
        few = min(1 - self.f_c, (1 - 0.67 * self.f_c) * self.fw)
        Ke = min(self.Kr * (Kc_max - Kcb), few * Kc_max)
        self.Ke_bound = few * Kc_max

        return Kcb, Ke, Ky

    def update_Ks(self) -> tuple[float, np.ndarray]:
        """
        Update Ks as a function of the depletion
        """
        reversed_layers = self.soil.get_reversed_layers()  # top to bottom ordered
        thetas = np.array([soil.get_theta() for soil in reversed_layers])
        theta_wps = np.array([soil.theta_wp for soil in reversed_layers])
        theta_fcs = np.array([soil.theta_fc for soil in reversed_layers])
        nominal_et_frac = np.array([layer.uptake_percentage for layer in reversed_layers])
        lenghts = np.zeros_like(thetas)
        
        cum_depths = np.cumsum(np.array([layer.depth for layer in reversed_layers]))
        for idx, layer in enumerate(reversed_layers):
            if self.root_depth >= cum_depths[idx]:
                z = layer.depth  # length of root in the layer
                lenghts[idx] = z 
            else:
                if idx >= 1 :
                    z = self.root_depth - cum_depths[idx - 1]  # length of root in the layer
                else:
                    z = self.root_depth
                lenghts[idx] = z 
                break
        theta_ts = theta_fcs - (theta_fcs - theta_wps) * self.MAD
        partial_taws = (theta_fcs - theta_wps) * lenghts
        partial_depletions = np.clip(theta_fcs - thetas, 0, theta_fcs - theta_wps) * lenghts
        
        depletion, taw = partial_depletions.sum(), partial_taws.sum()
        
        Ks = (taw - depletion) / ((1 - self.MAD) * taw) if depletion > (taw * self.MAD) else 1.0

        adjustement = (thetas - theta_wps) / (theta_ts - theta_wps)
        first_estimate = np.clip(adjustement, 0, 1) * nominal_et_frac
        
        final_fraction = first_estimate/first_estimate.sum() if first_estimate.sum() != 0 else nominal_et_frac

        self.depletion = depletion
        self.raw = taw * self.MAD

        assert np.isclose(final_fraction.sum(), 1.0)

        # pad the array with zeros to reach the number of layers
        #final_fraction = np.pad(final_fraction, (0, len(self.soil.layers) + 1 - len(final_fraction)), 'constant')
        return Ks, final_fraction
    



def crop_from_dict(crop_dict: dict[str, Any]) -> Crop:
    return Crop(crop_dict["crop_name"],
                crop_dict["plantation_day"],
                crop_dict["stages_duration"],
                crop_dict["Kcb"],
                crop_dict["MAD"],
                crop_dict["root_depth_init"],
                crop_dict["root_depth_max"],
                crop_dict["price"],
                crop_dict["f_c"],
                crop_dict["Ky"])


def e_s(temp: float) -> float:
    """
    partial pressure
    """
    return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))



class Layer:
    """Layer of soil"""

    def __init__(self, depth: float,
                 theta_fc: float,
                 theta_wp: float,
                 theta_sat: float,
                 theta_res: float,
                 K0: float,
                 alpha: float,
                 n: float,
                 theta: float):
        self.depth = depth
        self.theta_fc: float = theta_fc
        self.theta_wp: float = theta_wp
        self.theta_sat: float = theta_sat
        self.theta_res: float = theta_res
        self.n = n
        self.m = 1 - 1 / n
        self.K0 = K0
        self.alpha = alpha
        self.theta_init:float = theta
        self.theta: float = theta
        self.awc: float = theta_fc - theta_wp
        self.taw: float = self.awc * depth
        self.hist_theta: list[np.ndarray] = []
        self.partial_Ks: float = 0.0
        self.uptake_percentage: float = 0.0
        self.h_c: float = 0.0


    def get_params(self):
        theta_e = (self.theta - self.theta_res) / (self.theta_sat - self.theta_res)
        if theta_e <= 0:
            theta_e = 1e-5
        h_c = - ((theta_e ** (-1 / self.m) - 1) ** (1 / self.n)) / self.alpha
        if np.isinf(h_c):
            pass # print("here")
        h_c = h_c / 100  # [cm] -> [m]
        K = self.K0 * np.clip(theta_e ** .5 * (1 - (1 - theta_e ** (1 / self.m)) ** self.m) ** 2, 0, 1)
        K = K / 100  # [cm] -> [m]
        self.h_c = h_c
        return theta_e, h_c, K

    def set_theta(self, new_theta: float, save_in_hist: bool = False):
        """set new theta value and stores the old one"""
        assert isinstance(new_theta, float)
        if np.isnan(new_theta):
            new_theta = self.theta
        if save_in_hist:
            self.hist_theta.append(self.get_obs())
        self.theta = new_theta

    def get_theta(self):
        return self.theta

    def get_hist_data(self):
        return np.array(self.hist_theta)

    def reset(self):
        # set random theta
        self.theta = np.random.uniform(self.theta_wp, self.theta_fc)
        del self.hist_theta[:]
        self.uptake_percentage = 0.0

    def set_partial_Ks(self, partial_Ks):
        self.partial_Ks = partial_Ks
        return
    
    def set_uptake_percentage(self, uptake_percentage):
        self.uptake_percentage = uptake_percentage
        return

    def get_obs(self):
        theta_e = (self.theta - self.theta_res) / (self.theta_sat - self.theta_res)
        return np.array([self.theta, theta_e, self.partial_Ks, self.theta * self.depth, self.h_c])


class EvpLayer(Layer):
    def __init__(self,
                 depth: float,
                 theta_fc: float,
                 theta_wp: float,
                 theta_sat: float,
                 theta_res: float,
                 K0: float,
                 alpha: float,
                 n: float,
                 theta: float,
                 rew: float):
        super().__init__(depth, theta_fc, theta_wp, theta_sat, theta_res, K0, alpha, n, theta)
        self.rew = rew
        self.tew = (self.theta_fc - 0.5 * self.theta_wp) * self.depth # [m]

    def get_depletion(self):
        depletion = np.clip(self.theta_fc - self.theta, 0, self.theta_fc - self.theta_wp) * self.depth
        return depletion

    def get_Kr(self):
        if self.theta < 0.5*self.theta_wp:
            #print("a")
            pass
        depletion = np.clip(self.theta_fc - self.theta, 0, self.tew/self.depth) * self.depth # [m]
        Kr = (self.tew - depletion) / (self.tew - self.rew) if depletion > self.rew else 1.0
        return Kr
    def reset(self):
        "Evaporation layer begins at field capacity"
        self.set_theta(self.theta_fc)
        return

def layer_from_dict(layer_dict: dict[str, float]) -> Layer:
    """Initialize a layer object from a dictionary"""
    return Layer(layer_dict["depth"],
                 layer_dict["theta_fc"],
                 layer_dict["theta_wp"],
                 layer_dict["theta_sat"],
                 layer_dict["theta_res"],
                 layer_dict["K0"],
                 layer_dict["alpha"],
                 layer_dict["n"],
                 layer_dict["theta"])


def evp_layer_from_dict(layer_dict: dict[str, float]) -> EvpLayer:
    """Initialize a layer object from a dictionary"""
    return EvpLayer(layer_dict["depth"],
                    layer_dict["theta_fc"],
                    layer_dict["theta_wp"],
                    layer_dict["theta_sat"],
                    layer_dict["theta_res"],
                    layer_dict["K0"],
                    layer_dict["alpha"],
                    layer_dict["n"],
                    layer_dict["theta"],
                    layer_dict["rew"])


class Soil:
    """Tipping bucket model of soil"""

    def __init__(self, evp_layer: EvpLayer):
        self.evp_layer: EvpLayer = evp_layer
        self.layers: list[Layer] = []
        self.partial_Ks:list[float] = []
        self.incoming_water:float = 0.0
        self.outcoming_water:float = 0.0
        self.hist_data = []
        self.n_layers:int = len(self.layers)
        self.depth:float = None
        self.theta_fc:float = None
        self.theta_wp:float = None

    def __get_item__(self, idx):
        if idx < 0 or idx > self.n_layers:
            raise IndexError("Index out of range")
        return self.all_layers[idx]

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        return
    
    def get_depth(self) -> float:
        if self.depth is None:
            self.depth = sum([layer.depth for layer in self.get_reversed_layers()])  
        return self.depth
    
    def get_theta_fc(self) -> float:
        if self.theta_fc is None:
            self.theta_fc = np.mean([layer.theta_fc for layer in self.get_reversed_layers()])
        return self.theta_fc
    
    def get_theta_wp(self) -> float:
        if self.theta_wp is None:
            self.theta_wp = np.mean([layer.theta_wp for layer in self.get_reversed_layers()])
        return self.theta_wp

    def set_evp_layer(self, evp_layer: EvpLayer):
        self.evp_layer = evp_layer
        return
    
    def get_all_layers(self) -> list[Layer]:
        """Returns all layers in the soil from bottom to top"""
        return self.layers + [self.evp_layer]
    
    def get_reversed_layers(self) -> list[Layer]:
        """Returns all layers in the soil from top to bottom"""
        return [self.evp_layer] + self.layers[::-1]

    def get_thetas(self):
        """Returns the theta values of the soil layers from top to bottom"""
        thetas = []
        for layer in self.get_reversed_layers():
            thetas.append(layer.get_theta())
        return np.array(thetas)
    
    def get_avg_hc(self) -> float:
        """Computes the average of the matric potential in the active layers"""
        if self.evp_layer.uptake_percentage <= 0.0:
            return 0.0
        else:
            layers = self.get_reversed_layers()
        h_c = 0.0
        n_layers = 0
        for layer in layers:
            if layer.uptake_percentage <= 0.0:
                break
            n_layers += 1
            h_c += layer.h_c 
        h_c = h_c / n_layers
        return h_c

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self.evp_layer.reset()
        self.hist_data = []

    def set_partial_Ks(self, partial_Ks):
        self.partial_Ks = partial_Ks
        reversed_layers = self.layers[::-1]
        self.evp_layer.set_partial_Ks(partial_Ks[0])
        for idx, Ks in enumerate(partial_Ks[1:]):
            reversed_layers[idx].set_partial_Ks(Ks)

    def step(self, evaporation: float, infiltrated_water: float = 0.0, transpiration: float = 0.0):
        """
        Steps the soil model one day forward
        :param evaporation: [m/day]
        :param irrigation: [m/day]
        :param precipitation: [m/day]
        :param transpiration: [m/day]
        """

        assert isinstance(infiltrated_water, (float, np.floating))

        layer = self.layers[0]  # layers starts from the bottom
        z = layer.depth
        theta_e, h_c, Ke = layer.get_params()  # effective theta, matric potential and hydraulic conductivity
        outgoing_water = (Ke / z**2) * z
        for idx, next_layer in enumerate(self.layers[1:] + [self.evp_layer]):

            theta_e_next, h_c_next, Ke_next = next_layer.get_params()
            z_next = next_layer.depth

            Ke_j_next = harmonic_mean(Ke, z, Ke_next, z_next)
            d_next = (z * .5 + z_next * .5)

            incoming_water = Ke_j_next * (h_c_next - h_c + d_next) / d_next**2
            assert not np.isnan(incoming_water)
            if next_layer.theta_res - 1e-5 < next_layer.get_theta() < next_layer.theta_res + 1e-5:
                incoming_water = np.minimum(0.0, incoming_water)

            delta_theta = incoming_water - outgoing_water
            layer.set_theta(np.clip(layer.get_theta() + delta_theta, layer.theta_res, layer.theta_sat))
            assert layer.get_theta() >= layer.theta_res
            outgoing_water = incoming_water
            layer = next_layer

        # evaporation layer
        delta_theta = - outgoing_water
        self.evp_layer.set_theta(np.clip(self.evp_layer.get_theta() + delta_theta,
                                         self.evp_layer.theta_res,
                                         self.evp_layer.theta_sat))

        ## Seepage
        d_initial = self.evp_layer.get_theta()*self.evp_layer.depth
        d_sum = d_initial + infiltrated_water - evaporation - self.evp_layer.partial_Ks * transpiration
        
        if d_sum > self.evp_layer.theta_fc*self.evp_layer.depth:
            d_seepage = d_sum - self.evp_layer.theta_fc*self.evp_layer.depth
        else:
            d_seepage = 0.0
        self.evp_layer.set_theta((d_sum-d_seepage) / self.evp_layer.depth, True)

        for layer in self.layers[::-1]:
            d_initial = layer.get_theta() * layer.depth
            d_sum = d_initial + d_seepage - layer.partial_Ks * transpiration
            if d_sum > layer.theta_fc * layer.depth:
                d_seepage = d_sum - layer.theta_fc * layer.depth
            else:
                d_seepage = 0.0
            layer.set_theta((d_sum - d_seepage) / layer.depth, True)

    def get_soil_data(self):
        return np.array(self.hist_data)

    def get_hist_data(self):
        hist_data = {}
        for idx, layer in enumerate(self.layers + [self.evp_layer]):
            for jdx, jtem in enumerate(layer.get_hist_data()[0, :]):
                hist_data[f"layer_{idx}_{jdx}"] = layer.get_hist_data()[:, jdx]  # item.get_hist_data()
        return hist_data

    def get_Kr(self):
        Kr = self.evp_layer.get_Kr()
        return Kr

    def set_theta(self, theta:float):
        self.evp_layer.set_theta(theta)
        for layer in self.layers:
            layer.set_theta(theta)
        return
    
    def set_thetas(self, thetas: np.ndarray):
        """Set the theta values of the soil layers"""
        assert len(thetas) == len(self.layers) + 1

        for idx, layer in enumerate(self.get_reversed_layers()):
            layer.set_theta(thetas[idx])


def soil_from_dicts(evp_layer: dict[str, float], layers: list[dict[str, float]]):
    soil = Soil(evp_layer_from_dict(evp_layer))
    for layer_info in layers:
        soil.add_layer(layer_from_dict(layer_info))
    return soil


def harmonic_mean(x1, z1, x2, z2):
    if x1 == 0:
        return 0.0
    if x2 == 0:
        return 0.0
    else:
        return (z1 * .5 + z1 * .5) / ((z1 * .5 / x1) + (z2 * .5 / x2))

def aritmethic_mean(x1, z1, x2, z2):
    return (x1 * z1 + x2 * z2) / (z1 + z2)
