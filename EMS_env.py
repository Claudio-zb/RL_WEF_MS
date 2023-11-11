from typing import Any
from numpy import ndarray

from Funciones.funcionesEMS import *
import gymnasium as gym
from gymnasium import spaces


class EMS_env(gym.Env):
    """
    Environment for the Energy Management System
    """
    def __init__(self):
        # Hyperparams
        self.n_steps = 288  # 288
        self.start_index = 0
        self.k = 0
        self.dt = 600
        self.n_c = 0.85
        self.n_d = 1.15

        # Load meteorological data and demand

        self.radiation_data = get_rad('ver')
        self.temperature_data = get_temperatura('ver')
        self.demand_data = get_demand()
        self.L = len(self.temperature_data)  # length(temperatura)
        self.N_dias = 70

        self.V_refs = get_ref()  # V_refs

        # State variables en inputs

        self.I_1 = np.array([0.0])
        self.I_2 = np.array([0.0])
        self.d_I_1 = np.array([0.0])
        self.d_I_2 = np.array([0.0])
        self.V_I_1 = np.array([0.0])
        self.V_I_2 = np.array([0.0])
        self.Q_p = np.array([0.0])
        self.d_Q_p = np.array([0.0])
        self.Pbat = np.array([0.0])
        self.p_fv = np.array([0.0])
        self.temperatura = np.array([0.0])
        self.demanda = np.array([0.0])
        self.radiacion = np.array([0.0])
        self.V_1_ref = np.array([0.0])
        self.V_2_ref = np.array([0.0])
        self.Vt = np.array([0.0])
        self.SoE = np.array([0.0])
        self.E_Q_p = 0.0
        self.State = np.array([0.0])

        # Constants and bounds
        self.B_p = 1000 * 10
        self.h_p_const = 1
        # Tank Constants
        self.Vt_max = 5
        self.Vt_min = 1

        # Batteries Constants

        self.Pbat_max = 100
        self.SoE_max = 80
        self.SoE_min = 20

        # Irrigation Constants

        self.I_max = 1 / 1000  # 1L / s -> 0.001m3 / s
        self.I_min = 0
        self.d_I_bound = 1e-3  # I_max 
        self.Q_p_max = (1 / 1000)  # 1L / s <= > 0.001m3 / s
        self.d_Q_p_bound = 1e-3  # Q_p_max

        # Bounds for observations
        obs_low = np.array([self.Vt_min, self.Vt_min,
                            self.Vt_min, self.SoE_min,
                            self.I_min, self.I_min,
                            self.Vt_min, self.Vt_min,
                            0, 0, 0])
        obs_high = np.array([self.Vt_max, self.Vt_max,
                             self.Vt_max, self.SoE_max,
                             self.I_max, self.I_max,
                             self.Vt_max, self.Vt_max,
                             self.Q_p_max, 1000, 1000])

        # Bounds for actions
        action_low = np.array([-self.d_I_bound, -self.d_I_bound, -self.d_Q_p_bound, -self.Pbat_max],
                              dtype=np.float32)

        action_high = np.array([self.d_I_bound, self.d_I_bound, self.d_Q_p_bound, self.Pbat_max],
                               dtype=np.float32)

        # Agent params
        self.action_space = spaces.Box(low=action_low,
                                       high=action_high,
                                       shape=(4,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(11,),
                                            dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[ndarray, ndarray, bool, bool, dict[str, Any]]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        truncated = False
        terminated = False

        Info = {}

        self.d_I_1 = action[0]
        self.d_I_2 = action[1]
        self.d_Q_p = action[2]
        self.Pbat = action[3]

        self.I_1 = self.d_I_1 + self.I_1
        self.I_2 = self.d_I_2 + self.I_2

        self.Q_p = self.Q_p + self.d_Q_p

        # Read data from the arrays

        self.demanda = self.demand_data[self.start_index + self.k: self.start_index + self.n_steps + self.k]
        self.radiacion = self.radiation_data[self.start_index + self.k: self.start_index + self.n_steps + self.k]
        self.temperatura = self.temperature_data[self.start_index + self.k: self.start_index + self.n_steps + self.k]

        self.p_fv = solar_power(self.radiacion, self.temperatura)
        P_Q_p = self.dt * self.B_p * self.Q_p * self.h_p_const / 1e3  # water pump power

        # Set the limits of the variables

        self.I_1 = np.clip(self.I_1, self.I_min, self.I_max)
        self.I_2 = np.clip(self.I_2, self.I_min, self.I_max)
        self.Q_p = np.clip(self.Q_p, 0, self.Q_p_max)
        self.SoE = np.clip(self.SoE, self.SoE_min, self.SoE_max)

        # Energetic balance

        E_bat = self.Pbat * self.n_c / (60 * 60 / self.dt)  # battery consumption
        E_Q_p = P_Q_p * self.dt / (60 * 60 / self.dt)  # water pump consumption
        E_fv = self.p_fv[self.k]  # foto-voltaic generation
        E_demanda = self.demanda[self.k]  # demand

        E_residual = E_fv - E_demanda - E_Q_p - E_bat  # If it is positive there is an energy surplus. Otherwise,
        # buy energy will be needed

        self.SoE = self.SoE + self.Pbat * self.n_c / (60 * 60 / self.dt)

        self.Vt = self.Vt + self.dt * self.Q_p - self.dt * self.I_1 - self.dt * self.I_2
        self.V_I_1 = self.V_I_1 + self.dt * self.I_1
        self.V_I_2 = self.V_I_2 + self.dt * self.I_2

        observation = np.array([self.V_1_ref, self.V_2_ref,
                                self.Vt, self.SoE,
                                self.I_1, self.I_2,
                                self.V_I_1, self.V_I_2,
                                self.Q_p, self.p_fv[self.k], self.demanda[self.k]])

        reward = get_reward(E_residual,
                            self.d_I_2, self.d_I_1,
                            self.I_2, self.I_1, self.V_I_1, self.V_I_2,
                            self.V_1_ref, self.V_2_ref, self.d_Q_p, self.Q_p)

        reward = np.array([reward])

        self.k = self.k + 1

        if self.k >= self.n_steps - 1:  # Number of steps are completed
            truncated = True

        return observation, reward, terminated, truncated, Info

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state
        :param seed: random seed
        :param options: options for the environment
        :return: tuple of (initial_observation, info)
        """
        info = {}
        day_picked = np.random.randint(0, 70)  # Pick a random day from the data
        self.start_index = day_picked * 144  # 288 is the number of steps per day

        self.V_1_ref = self.V_refs[day_picked]
        self.V_2_ref = self.V_refs[day_picked + 1]
        self.Vt = (self.Vt_max - self.Vt_min) * np.random.random_sample() + self.Vt_min
        self.SoE = (self.SoE_max - self.SoE_min) * np.random.random_sample() + self.SoE_min
        self.I_1 = 0
        self.I_2 = 0
        self.V_I_2 = 0
        self.V_I_1 = 0
        self.radiacion = self.radiation_data[self.start_index + self.k]
        self.temperatura = self.temperature_data[self.start_index + self.k]
        self.p_fv = solar_power(self.radiacion, self.temperatura)
        self.demanda = self.demand_data[self.start_index + self.k]
        self.Q_p = 0
        self.d_Q_p = 0

        InitialObservation = np.array(
            [self.V_1_ref, self.V_2_ref, self.Vt, self.SoE,
             self.I_1, self.I_2, self.V_I_1, self.V_I_2,
             self.Q_p, self.p_fv, self.demanda])

        self.k = 1
        return InitialObservation, info
