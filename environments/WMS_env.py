from typing import Any, Tuple, Dict, List

import numpy as np
import pandas as pd


def irr_policy(obs: np.ndarray) -> float:
    """Irrigation policy"""
    depletion = obs[0]
    raw = obs[1]
    if depletion > 0.75*raw:
        irrigation = 5  # [mm]
    else:
        irrigation = 0.0
    return irrigation


class WMS:
    """Water Management System Class"""

    def __init__(self):
        self.crops: List[Crop] = [crop_from_dict(tomato)]
        self.weather_data: pd.DataFrame = pd.read_csv("environments/weather_data.csv")
        self.doy: int = 1
        self.wind_speed: float = 0.0
        self.max_temperature: float = 0.0
        self.min_temperature: float = 0.0
        self.RH_max_temperature: float = 0.0
        self.RH_min_temperature: float = 0.0
        self.solar_radiation: float = 0.0
        self.precipitation: float = 0.0
        self.geological_parameters = jose_painecura
        self.ET0: float = 0.0

    def start(self, doy: int = 295):
        self.doy = doy

    def update_climate_data(self, doy: int):

        data = self.weather_data.loc[self.weather_data["doy"] == doy]
        self.precipitation = data["precipitation"].values[0]
        try:
            self.ET0 = data["ET0"].iloc[0]
        except KeyError:
            self.ET0 = np.nan
            self.wind_speed = data["w_speed"].iloc[0]
            self.max_temperature = data["t_max"].iloc[0]
            self.min_temperature = data["t_min"].iloc[0]
            self.RH_max_temperature = data["RH_tmax"].iloc[0]
            self.RH_min_temperature = data["RH_tmin"].iloc[0]
            self.solar_radiation = data["rad"].iloc[0]
            self.ET0 = data["ET0"].iloc[0]

    def get_climate_data(self):
        climate_data = {"solar_radiation": self.solar_radiation,
                        "max_temperature": self.max_temperature,
                        "min_temperature": self.min_temperature,
                        "RH_max_temperature": self.RH_max_temperature,
                        "RH_min_temperature": self.RH_min_temperature,
                        "wind_speed": self.wind_speed,
                        "precipitation": self.precipitation}
        return climate_data

    def step(self):
        self.update_climate_data(self.doy)
        for crop in self.crops:
            if crop.is_active():
                obs = crop.get_obs()
                irrigation = irr_policy(obs)
                crop.step(self.ET0, self.precipitation + irrigation)
            else:
                if crop.plantation_day == self.doy:
                    crop.start()
        self.doy = max(1, (self.doy + 1) % 365)

    def get_hist_data(self):
        hist_data = {}
        for crop in self.crops:
            hist_data[crop.crop_parameters["crop_name"]] = crop.get_hist_data()
        return hist_data

    def get_ET0(self) -> float:
        """Returns daily reference evapotranspiration"""
        if self.ET0 is not np.nan:
            return self.ET0
        else:
            return self.estimate_ET0()

    def estimate_ET0(self) -> float:
        """Computes an estimation for daily evapotranspiration from weather data [mm/day]"""
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


class Crop:

    def __init__(self, crop_name, plantation_day, stages_duration, Kcb, MAD, root_depth_init, root_depth_max, price,
                 fc):
        """First implementation made for only one crop"""

        self.crop_name = crop_name
        self.days_since_plantation: int = 0
        self.doy: int = 0
        self.plantation_day: int = plantation_day
        self.harvest_day: int = 0
        self.stages_duration: List[int] = stages_duration
        self.Kcb_list: List[float] = Kcb
        self.MAD: float = MAD
        self.root_depth_init: float = root_depth_init
        self.root_depth_max: float = root_depth_max
        self.price: float = price
        self.fc_list: List[float] = fc
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
        self.root_depth: float = 0  # [m]
        self.Kcb: float = 1.0
        self.Ks: float = 1.0
        self.Ke: float = .5
        self.hist_data: List[np.ndarray] = []

        # parameters
        self.soil: Soil = soil_from_dicts(evp_layer_specs,
                                          [layer_specs, layer_specs2, layer_specs2, layer_specs])
        self.crop_parameters = tomato
        self.geological_parameters = jose_painecura

    def start(self) -> Tuple[Any, Dict[str, Any]]:
        self.days_since_plantation = 1
        self.doy = self.crop_parameters["plantation_day"]  # lets see
        self.update()
        obs = self._get_observation()
        self.hist_data.append(obs)
        return obs, {}

    def step(self, ET0: float, infiltrated_water: float = 0.0) -> np.ndarray:
        """
        Steps the crop model one day forward
        :param ET0: reference evapotranspiration [mm/day]
        :param infiltrated_water: incoming water [mm/day]
        """

        self.days_since_plantation += 1
        self.doy = max(1, (self.doy + 1) % 365)
        self.ref_evapotranspiration = ET0
        self.update(infiltrated_water)
        obs = self._get_observation()
        self.hist_data.append(obs)

        return self._get_observation()

    def update(self, irrigation: float = 0.0):
        self.fw = 1.0 if irrigation > 0.0 else 0.8

        self.root_depth = self.update_root_depth()
        self.Kcb, self.Ke = self.get_dual_coeffs(self.days_since_plantation)
        self.potential_crop_evapotranspiration = self.ref_evapotranspiration * (self.Kcb + self.Ke)

        self.Ks, partial_Ks = self.update_Ks()
        self.crop_evapotranspiration = self.ref_evapotranspiration * (self.Ks * self.Kcb + self.Ke)
        evaporation = self.Ke * self.potential_crop_evapotranspiration / 1000  # [mm] -> [m]
        transpiration = self.Kcb * self.Ks * self.potential_crop_evapotranspiration / 1000  # [mm] ->[m]

        self.soil.set_partial_Ks(partial_Ks)
        self.soil.step(evaporation, irrigation, self.precipitation, transpiration)

    def is_active(self):
        return True if self.days_since_plantation > 0 else False

    def get_hist_data(self) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
        hist_data = np.array(self.hist_data)
        crop_hist_data = {"root_depth": hist_data[:, 0],
                          "pcrop_evapotranspiration": hist_data[:, 1],
                          "ref_evapotranspiration": hist_data[:, 2],
                          "Kc": hist_data[:, 3],
                          "crop_evapotranspiration": hist_data[:, 4],
                          "Ks": hist_data[:, 5]}
        return crop_hist_data, self.soil.get_hist_data()

    def _get_observation(self) -> np.ndarray:
        return np.array([self.root_depth,
                         self.potential_crop_evapotranspiration,
                         self.ref_evapotranspiration,
                         self.Kcb,
                         self.crop_evapotranspiration,
                         self.Ks])

    def get_obs(self) -> np.ndarray:
        return np.array([self.depletion,
                         self.raw,
                         self.MAD])

    def update_root_depth(self) -> float:
        """Root depth as a function of time
        params
        """
        t = self.days_since_plantation
        t0 = self.stages_duration[0]
        t1 = self.stages_duration[0] + self.stages_duration[1]
        growth_rate = (self.root_depth_max - self.root_depth_init) / (t1 - t0)
        if t < t0:
            return self.root_depth_init
        elif t < t1:
            return self.root_depth + growth_rate
        else:
            return self.root_depth

    def get_dual_coeffs(self, t: int) -> Tuple[float, float]:
        """Dual Crop coefficient as a function of time
        params
        t: time since plantation day [days]
        """
        stages = self.stages_duration
        t0 = stages[0]
        t1 = stages[0] + stages[1]
        t2 = stages[0] + stages[1] + stages[2]
        t3 = stages[0] + stages[1] + stages[2] + stages[3]

        if t < t0:  # initial stage
            Kcb, fc = self.Kcb_list[0], self.fc_list[0]
        elif t < t1:  # crop development
            Kcb = (self.Kcb_list[1] - self.Kcb_list[0]) / (t1 - t0) * (t - t0) + self.Kcb_list[0]
            fc = (self.fc_list[1] - self.fc_list[0]) / (t1 - t0) * (t - t0) + self.fc_list[0]
        elif t < t2:  # mid-season
            Kcb, fc = self.Kcb_list[1], self.fc_list[1]
        elif t < t3:  # late season
            Kcb = (self.Kcb_list[2] - self.Kcb_list[1]) / (t3 - t2) * (t - t2) + self.Kcb_list[1]
            fc = (self.fc_list[2] - self.fc_list[1]) / (t3 - t2) * (t - t2) + self.fc_list[1]
        else:  # goodbye
            Kcb, fc = 0.0, 0.0

        Kcmax = max(1.2, Kcb + .05)
        Kr = self.soil.get_Kr()
        few = min(1 - fc, (1 - 0.67 * fc) * self.fw)
        Ke = min(Kr * (Kcmax - Kcb), few * Kcmax)

        return Kcb, Ke

    def update_Ks(self) -> Tuple[float, np.ndarray]:
        """Update Ks as a function of the depletion"""

        depletion = 0.0
        taw = 0.0

        stop = False
        theta_wps = []
        partial_depletions = []
        partial_taws = []
        lenghts = []
        theta_ts = []
        reversed_layers = [self.soil.evp_layer] + self.soil.layers[::-1]
        cum_depths = np.cumsum(np.array([layer.depth for layer in reversed_layers]))
        for idx, layer in enumerate(reversed_layers):
            theta, theta_fc, theta_wp = layer.get_theta(), layer.theta_fc, layer.theta_wp
            if self.root_depth >= cum_depths[idx]:
                z = layer.depth  # length of root in the layer
            else:
                try:
                    z = self.root_depth - cum_depths[idx - 1]  # length of root in the layer
                except IndexError:
                    z = self.root_depth
                stop = True
            lenghts.append(z)
            theta_wps.append(layer.theta_wp)
            theta_ts.append(layer.theta_fc - (layer.theta_fc - layer.theta_wp) * self.MAD)
            partial_depletions.append(np.clip(theta_fc - theta, 0, theta_fc - theta_wp) * z)
            partial_taws.append((theta_fc - theta_wp) * z)
            if stop:
                break
        depletion = np.array(partial_depletions).sum()
        taw = np.array(partial_taws).sum()
        if depletion > taw * self.MAD:
            Ks = (taw - depletion) / ((1 - self.MAD) * taw)
        else:
            Ks = 1.0
        lenghts = np.array(lenghts)
        nominal_et_frac = lenghts / lenghts.sum()

        adjustement = (np.array(theta_ts) - np.array(theta_wps)) / (np.array(theta_ts) - np.array(theta_wps))
        first_estimate = np.clip(adjustement, 0, 1) * nominal_et_frac
        final_fraction = first_estimate / first_estimate.sum()
        self.depletion = depletion
        self.raw = taw*self.MAD
        return Ks, final_fraction


# parameters of cultives
tomato = {"crop_name": "tomato",
          "plantation_day": 295,  # [doy]
          "stages_duration": [30, 40, 40, 25],  # [days]
          "Kcb": [0.6, 1.15, 0.6],
          "MAD": 0.5,
          "root_depth_init": 0.2 * 0.8,  # [m]
          "root_depth_max": 1.1 * 0.8,  # [m]
          "height_max": 0.6,  # [m]
          "production_max": 86910,  # [kg/ha]
          "price": 220,  # [$/kg]
          "fc": [.1, .8, .2]
          }


def crop_from_dict(crop_dict: Dict[str, Any]) -> Crop:
    return Crop(crop_dict["crop_name"],
                crop_dict["plantation_day"],
                crop_dict["stages_duration"],
                crop_dict["Kcb"],
                crop_dict["MAD"],
                crop_dict["root_depth_init"],
                crop_dict["root_depth_max"],
                crop_dict["price"],
                crop_dict["fc"])


# geographical parameters

jose_painecura = {"location_name": "Jose Painecura",
                  "latitude": -38.67948359212895,
                  "longitude": -73.47621873681696,  # [deg]
                  "elevation": 0,  # [m],
                  "field_capacity": 0.2,
                  "wilting_point": 0.1,
                  }


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
        self.theta_fc = theta_fc
        self.theta_wp = theta_wp
        self.theta_sat = theta_sat
        self.theta_res = theta_res
        self.n = n
        self.m = 1 - 1 / n
        self.K0 = K0
        self.alpha = alpha
        self.theta = theta
        self.awc = theta_fc - theta_wp
        self.hist_theta = []
        self.partial_Ks: float = 0.0
        self.h_c: float = 0.0

    def get_params(self):
        theta_e = (self.theta - self.theta_res) / (self.theta_sat - self.theta_res)
        h_c = - ((theta_e ** (-1 / self.m) - 1) ** (1 / self.n)) / self.alpha
        if np.isinf(h_c):
            print("here")
        h_c = h_c / 100  # [cm] -> [m]
        K = self.K0 * theta_e ** .5 * (1 - (1 - theta_e ** (1 / self.m)) ** self.m) ** 2
        K = K / 100  # [cm] -> [m]
        self.h_c = h_c*K
        return theta_e, h_c, K

    def set_theta(self, new_theta):
        """set new theta value and stores the old one"""
        if np.isnan(new_theta):
            new_theta = self.theta
        self.hist_theta.append(self.get_obs())
        self.theta = new_theta

    def get_theta(self):
        return self.theta

    def get_hist_data(self):
        return np.array(self.hist_theta)

    def reset(self):
        self.theta = self.theta_fc
        self.hist_theta = []

    def set_partial_Ks(self, partial_Ks):
        self.partial_Ks = partial_Ks
        return

    def get_obs(self):
        theta_e = (self.theta_sat - self.theta) / (self.theta_sat - self.theta_res)
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

    def get_depletion(self):
        depletion = np.clip(self.theta_fc - self.theta, 0, self.theta_fc - self.theta_wp) * self.depth
        return depletion

    def get_Kr(self):
        tew = (self.theta_fc - 0.5 * self.theta_wp) * self.depth
        depletion = np.clip(self.theta_fc - self.theta, 0, self.theta_fc - self.theta_wp) * self.depth
        Kr = (tew - depletion) / (tew - self.rew) if depletion > self.rew else 1.0
        return Kr


def layer_from_dict(layer_dict: Dict[str, float]) -> Layer:
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


def evp_layer_from_dict(layer_dict: Dict[str, float]) -> EvpLayer:
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
        self.layers: List[Layer] = []
        self.partial_Ks = []
        self.incoming_water = 0.0
        self.outcoming_water = 0.0
        self.hist_data = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        return

    def set_evp_layer(self, evp_layer: EvpLayer):
        self.evp_layer = evp_layer
        return

    def set_partial_Ks(self, partial_Ks):
        self.partial_Ks = partial_Ks
        reversed_layers = self.layers[::-1]
        self.evp_layer.set_partial_Ks(partial_Ks[0])
        for idx, Ks in enumerate(partial_Ks[1:]):
            reversed_layers[idx].set_partial_Ks(Ks)

    def step(self, evaporation: float, irrigation: float = 0.0,
             precipitation: float = 0.0, transpiration: float = 0.0):

        layer = self.layers[0]
        z = layer.depth
        theta_e, h_c, Ke = layer.get_params()
        outgoing_water = Ke / z
        outgoing_w = evaporation + transpiration + Ke
        for idx, next_layer in enumerate(self.layers[1:] + [self.evp_layer]):

            theta_e_next, h_c_next, Ke_next = next_layer.get_params()
            z_next = next_layer.depth

            Ke_j_next = harmonic_mean(Ke, z, Ke_next, z_next)
            d_next = (z * .5 + z_next * .5)
            if np.isinf(h_c_next):
                incoming_water = 0
            else:
                incoming_water = Ke_j_next * (h_c_next - h_c + d_next) / d_next

            delta_theta = (incoming_water - outgoing_water - layer.partial_Ks * transpiration)

            if layer.get_theta() + delta_theta > layer.theta_sat:  # saturated content case
                incoming_water = layer.theta_sat - layer.get_theta()

            if layer.get_theta() + delta_theta < layer.theta_res:  # dry content case
                outgoing_water = 0.0

            if layer.get_theta() > layer.theta_wp:
                delta_theta = incoming_water - outgoing_water
            else:
                delta_theta = incoming_water - outgoing_water - layer.partial_Ks * transpiration

            layer.set_theta(np.clip(layer.get_theta() + delta_theta, layer.theta_res, layer.theta_sat))
            outgoing_water = incoming_water
            layer = next_layer

        # evaporation layer
        outgoing_water += (evaporation - irrigation - precipitation) / self.evp_layer.depth

        incoming_w = (irrigation + precipitation)
        incoming_water = 0.0
        delta_theta = (incoming_water - outgoing_water - self.evp_layer.partial_Ks * transpiration - evaporation)
        self.evp_layer.set_theta(np.clip(self.evp_layer.get_theta() + delta_theta,
                                         self.evp_layer.theta_res,
                                         self.evp_layer.theta_sat))
        self.register_water(incoming_w, outgoing_w)

    def get_trajectory(self, n_steps, irr_policy: callable = lambda x: 0.0):
        trajectory = {}
        for idx, item in enumerate(self.layers):
            trajectory[f"layer_{idx}"] = np.zeros(n_steps + 1)
            trajectory[f"layer_{idx}"][0] = item.theta

        for i in range(n_steps):
            self.step(irr_policy(i))
            for idx, item in enumerate(self.layers):
                trajectory[f"layer_{idx}"][i + 1] = item.get_theta()
        return trajectory

    def register_water(self, incoming_w, outcoming_w):
        self.hist_data.append(np.array([self.incoming_water, self.outcoming_water]))
        self.incoming_water = incoming_w
        self.outcoming_water = outcoming_w

    def get_obs(self):
        return np.array([self.incoming_water, self.outcoming_water])

    def get_soil_data(self):
        return np.array(self.hist_data)

    def get_hist_data(self):
        w_flux = self.get_soil_data()
        hist_data = {"w_in": w_flux[:, 0],
                     "W_out": w_flux[:, 1]}
        for idx, item in enumerate(self.layers + [self.evp_layer]):
            for jdx, jtem in enumerate(item.get_hist_data()[0, :]):
                hist_data[f"layer_{idx}_{jdx}"] = item.get_hist_data()[:, jdx]  # item.get_hist_data()
        return hist_data

    def get_Kr(self):
        Kr = self.evp_layer.get_Kr()
        return Kr


def soil_from_dicts(evp_layer: Dict[str, float], layers: List[Dict[str, float]]):
    soil = Soil(evp_layer_from_dict(evp_layer))
    for layer_info in layers:
        soil.add_layer(layer_from_dict(layer_info))
    return soil


layer_specs = {"depth": .5,
               "theta_fc": .30,
               "theta_wp": .1,
               "theta_sat": .45,
               "theta_res": .067,
               "alpha": 0.078,  # .02,
               "K0": 10.8,
               "n": 1.75,  # 1.41,
               "theta": 0.3}

layer_specs2 = {"depth": .5,
                "theta_fc": .30,
                "theta_wp": .1,
                "theta_sat": .45,
                "theta_res": .067,
                "alpha": 0.078,  # .02,
                "K0": 10.8,
                "n": 1.75,  # 1.41,
                "theta": 0.25}

evp_layer_specs = {"depth": 0.1,
                   "theta_fc": 0.3,
                   "theta_wp": 0.1,
                   "theta_sat": 0.45,
                   "theta_res": 0.067,
                   "alpha": 0.078,
                   "K0": 10.8,
                   "n": 1.75,
                   "theta": 0.3,
                   "rew": 0.2}


def harmonic_mean(x1, z1, x2, z2):
    if x1 == 0:
        return 0.0
    if x2 == 0:
        return 0.0
    else:
        return (z1 * .5 + z1 * .5) / ((z1 * .5 / x1) + (z2 * .5 / x2))
