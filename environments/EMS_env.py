from typing import Any
from numpy import ndarray
import matplotlib.pyplot as plt

from utils_functions.funcionesEMS import *
from gymnasium import spaces
from environments.custom_env import Custom_env


class EMS_env(Custom_env):
    """
    Environment for the Energy Management System
    """
    def __init__(self, render: bool = True):
        """
        Initialize the environment
        :param render:
        """

        self.fig = None
        self.axs = None
        if render:
            plt.ion()
            self.fig, self.axs = plt.subplots(4, 2)
            self.fig.suptitle('Energy Management System')
            self.fig.tight_layout()
            self.fig.set_size_inches(10, 10)

        # Hyperparams
        self.n_steps = 288  # 288
        self.start_index = 0
        self.k = 0
        self.dt = 600
        self.n_c = 0.85
        self.n_d = 1.15
        self.V_1_ref = 0.0
        self.V_2_ref = 0.0

        # Load meteorological data and demand

        self.radiation_data = get_rad('ver')
        self.temperature_data = get_temperatura('ver')
        self.demand_data = get_demand()
        self.L = len(self.temperature_data)  # length(temperatura)
        self.N_dias = 70

        self.V_refs = get_ref()  # V_refs

        # State variables en inputs

        self.Irr = np.array([0.0])
        # self.I_2 = np.array([0.0])
        self.d_Irr = np.array([0.0])
        # self.d_I_2 = np.array([0.0])
        self.V_Irr = np.array([0.0])
        # self.V_I_2 = np.array([0.0])
        self.Q_p = np.array([0.0])
        self.d_Q_p = np.array([0.0])
        self.Pbat = np.array([0.0])
        self.p_fv = np.array([0.0])
        self.temperatura = np.array([0.0])
        self.demanda = np.array([0.0])
        self.radiacion = np.array([0.0])
        self.V_ref = np.array([0.0])
        # self.V_2_ref = np.array([0.0])
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
        Pbat_nom = 100
        self.Pbat_max = Pbat_nom
        self.SoE_max = 0.8 * Pbat_nom
        self.SoE_min = 0.2 * Pbat_nom

        # Irrigation Constants

        self.I_max = 100  # 1 / 1000  # 1L / s -> 0.001m3 / s
        self.I_min = 0
        self.d_I_bound = 1e-3  # I_max 
        self.Q_p_max = 100  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
        self.d_Q_p_bound = 1e-3  # Q_p_max

        # Observation space
        # V_ref, Vt, SoE, I_prev, V_Irr, Q_p_prev, p_fv, demand

        # Bounds for observations
        obs_low = np.array([0.0,
                            self.Vt_min, self.SoE_min,
                            self.I_min,
                            0.0,
                            0.0, 0.0, 0.0], dtype=np.float32)

        obs_high = np.array([self.Vt_max,
                             self.Vt_max, self.SoE_max,
                             self.I_max,
                             self.Vt_max,
                             self.Q_p_max, 1000, 1000], dtype=np.float32)

        # Action space
        # Irr, Q_p, Pbat

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0, -self.Pbat_max],
                                   dtype=np.float32)

        self.action_high = np.array([100, 100, self.Pbat_max],
                                    dtype=np.float32)

        super().__init__(self.action_low, self.action_high)

        # Agent params
        self.action_space = spaces.Box(low=self.action_low,
                                       high=self.action_high,
                                       shape=(3,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(8,),
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

        # Store the previous values of the variables to compute the reward
        Irr_prev = self.Irr

        Q_p_prev = self.Q_p
        Pbat_prev = 0.0  # self.Pbat
        V_Irr_prev = self.V_Irr

        self.Irr = action[0]
        self.Q_p = action[1]
        self.Pbat = action[2]

        # Read data from the arrays

        self.demanda = self.demand_data[self.start_index + self.k: self.start_index + self.n_steps + self.k]
        self.radiacion = self.radiation_data[self.start_index + self.k: self.start_index + self.n_steps + self.k]
        self.temperatura = self.temperature_data[self.start_index + self.k: self.start_index + self.n_steps + self.k]

        # Set the limits of the variables

        self.Irr = np.clip(self.Irr, self.I_min, self.I_max)
        # self.I_2 = np.clip(self.I_2, self.I_min, self.I_max)
        # self.Q_p = np.clip(self.Q_p, 0, self.Q_p_max)
        # self.SoE = np.clip(self.SoE, self.SoE_min, self.SoE_max)

        Soe_to_charge = self.SoE_max - self.SoE  # Amount of energy that can be charged
        Soe_available = self.SoE - self.SoE_min  # Amount of energy that can be discharged

        if self.Pbat > 0:  # Charging
            self.SoE = self.SoE + (Soe_to_charge * self.Pbat * 0.01 * self.n_c) / (60 * 60 / self.dt)

        else:  # Discharging
            self.SoE = self.SoE + (Soe_available * self.Pbat * 0.01 * self.n_c) / (60 * 60 / self.dt)

        irrigation_penalty = 0
        if self.Vt <= 0:  # If the tank is empty, there is no irrigation
            self.Irr = 0
            irrigation_penalty = 0  #-300

        Vt_to_fill = self.Vt_max - self.Vt - self.dt * (self.Irr * 1e-6)  # Amount of water that can be filled
        Vt_available = self.Vt - self.Vt_min - self.dt * (self.Irr * 1e-6)  # Amount of water that can be discharged

        extraction_penalty = 0
        if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
            self.Q_p = 0
            extraction_penalty = 0  # -300

        if Vt_available <= 0 and self.Q_p < 0:  # If the tank is empty and the pump is extracting, the pump is turned off
            self.Q_p = 0
            extraction_penalty = 0  #-300

        self.p_fv = solar_power(self.radiacion, self.temperatura)
        P_Q_p = self.dt * self.B_p * (self.Q_p * 1e-6) * self.h_p_const / 1e3  # water pump power

        self.Vt = self.Vt + self.dt * (self.Q_p * 1e-6) - self.dt * (self.Irr * 1e-6)
        self.V_Irr = self.V_Irr + self.dt * (self.Irr * 1e-6)

        # Energetic balance

        E_bat = (Soe_to_charge * self.Pbat * 0.01 * self.n_c) / (60 * 60 / self.dt)  # battery consumption
        E_Q_p = P_Q_p * self.dt / (60 * 60 / self.dt)  # water pump consumption
        E_fv = self.p_fv[self.k]  # foto-voltaic generation
        E_demanda = self.demanda[self.k]  # Energetic demand

        E_residual = E_fv - E_demanda - E_Q_p - E_bat  # If it is positive there is an energy surplus. Otherwise,
        # buying energy will be required

        observation = np.array([self.V_ref,
                                self.Vt, self.SoE,
                                self.Irr,
                                self.V_Irr,
                                self.Q_p, self.p_fv[self.k], self.demanda[self.k]])

        reward = get_reward(E_residual,
                            Irr_prev,
                            self.Irr,
                            V_Irr_prev,
                            self.V_Irr,
                            self.V_ref,
                            Q_p_prev, self.Q_p,
                            Pbat_prev, self.Pbat)

        reward = np.array([reward]) - irrigation_penalty - extraction_penalty

        self.k = self.k + 1

        if self.k == self.n_steps//2:  # A day has passed
            self.V_ref = self.V_2_ref
            self.V_Irr = 0

        if self.k >= self.n_steps:  # Number of steps are completed
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

        self.V_ref = self.V_refs[day_picked]
        self.V_2_ref = self.V_refs[day_picked + 1]
        self.Vt = (self.Vt_max - self.Vt_min) * np.random.random_sample() + self.Vt_min
        self.SoE = (self.SoE_max - self.SoE_min) * np.random.random_sample() + self.SoE_min
        self.Irr = 0
        self.V_Irr = 0
        self.radiacion = self.radiation_data[self.start_index + self.k]
        self.temperatura = self.temperature_data[self.start_index + self.k]
        self.p_fv = solar_power(self.radiacion, self.temperatura)
        self.demanda = self.demand_data[self.start_index + self.k]
        self.Q_p = 0
        self.d_Q_p = 0

        InitialObservation = np.array(
            [self.V_ref,
             self.Vt, self.SoE,
             self.Irr,
             self.V_Irr,
             self.Q_p, self.p_fv,
             self.demanda])

        self.k = 0
        return InitialObservation, info

    def show_sample(self, policy):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """
        
        states, actions = self.sample_trajectory(policy)
        for ax in self.axs.flat:
            ax.clear()

        t = np.linspace(0, 48, 288)

        E_bat = np.diff(states[:, 2])
        Qp = np.diff(states[:, 1]) + self.dt * (actions[:, 0] * 1e-6)
        P_Q_p = self.dt * self.B_p * Qp / self.dt * self.h_p_const / 1e3
        E_Q_p = P_Q_p * self.dt / (60 * 60 / self.dt)  # water pump consumption
        E_fv = states[:-1, 6]  # foto-voltaic generation
        E_demanda = states[:-1, 7]  # Energetic demand

        E_residual = E_fv - E_demanda - E_Q_p - E_bat  # If it is positive there is an energy surplus. Otherwise,

        self.axs[0, 0].step(t, states[:-1, 0], where='post', label='V_ref')
        self.axs[0, 0].step(t, states[:-1, 4], where='post', label='V_Irr')
        self.axs[0, 0].set_title('Water demand fulfilled')
        # self.axs[0].set_xlabel('Time (h)')
        self.axs[0, 0].set_ylabel('Water volume (m3)')

        self.axs[0, 1].step(t, actions[:, 0], where='post', label='Irr')
        self.axs[0, 1].set_title('Irrigation level')
        # self.axs[1].set_xlabel('Time (h)')
        self.axs[0, 1].set_ylabel('Irrigation level (%)')

        self.axs[1, 0].step(t, states[:-1, 2], where='post', label='Soe')
        self.axs[1, 0].set_title('SoE batteries')
        # self.axs[2].set_xlabel('Time (h)')
        self.axs[1, 0].set_ylabel('SoE (kWh)')

        self.axs[1, 1].step(t, (E_bat / self.n_c) * (60 * 60 / self.dt), where='post', label='Pbat')
        self.axs[1, 1].set_title('Battery power')
        # self.axs[3].set_xlabel('Time (h)')
        self.axs[1, 1].set_ylabel('Power (kW)')

        self.axs[2, 0].step(t, states[:-1, 1], where='post', label='V_tank')
        self.axs[2, 0].set_title('Tank volume')
        self.axs[2, 0].set_xlabel('Time (h)')
        self.axs[2, 0].set_ylabel('Volume (m3)')

        self.axs[2, 1].step(t, actions[:, 1], where='post', label='Irr')
        self.axs[2, 1].set_title('Pump power')
        self.axs[2, 1].set_xlabel('Time (h)')
        self.axs[2, 1].set_ylabel('(%)')

        self.axs[3, 0].step(t, E_residual, where='post', label='E_residual')
        self.axs[3, 0].set_title('Residual energy')

        balance = np.cumsum(E_residual)
        self.axs[3, 1].step(t, balance, where='post', label='E_bat')
        self.axs[3, 1].set_title('Energy balance')

        self.axs[0, 0].legend()
        self.axs[0, 1].legend()
        self.axs[1, 0].legend()
        self.axs[1, 1].legend()
        self.axs[2, 0].legend()
        self.axs[2, 1].legend()

        plt.pause(0.1)
        plt.show()

    def close(self):
        """
        Close the environment
        :return:
        """
        plt.close()
        plt.ioff()
