from typing import Any

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import torch
import matplotlib

from utils_functions.funcionesEMS import *
from gymnasium import spaces
from environments.custom_env import Custom_env
from typing import Union

matplotlib.use('Qt5Agg')


class EMS_env(Custom_env):
    """
    Environment for the Energy Management System
    """

    def __init__(self, render: bool = True, continuous: bool = False):
        """
        Initialize the environment
        :param render:
        """
        self.isContinuous = continuous
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
        self.time = 0

        # Load meteorological data and demand

        self.radiation_data = get_rad('ver')
        self.temperature_data = get_temperatura('ver')
        self.demand_data = get_demand()
        self.L = len(self.temperature_data)  # length(temperatura)
        self.N_dias = 70

        self.V_refs = get_ref()  # V_refs

        # State variables en inputs

        self.Irr = np.array([0.0])
        self.d_Irr = np.array([0.0])
        self.V_Irr = np.array([0.0])
        self.Q_p = np.array([0.0])
        self.d_Q_p = np.array([0.0])
        self.Pbat = np.array([0.0])
        self.p_fv: np.ndarray[float] = np.array([0.0])
        self.temperatura: np.ndarray[float] = np.array([0.0])
        self.demanda: np.ndarray[float] = np.array([0.0])
        self.radiacion: np.ndarray[float] = np.array([0.0])
        self.V_ref = np.array([0.0])
        self.Vt = np.array([0.0])
        self.SoE = np.array([0.0])
        self.E_Q_p = 0.0
        self.State = np.array([0.0])
        self.E_surplus = 0.0
        self.E_deficit = 0.0
        self.Irr_levels = np.array([0.0, 33.33, 66.66, 100])
        self.Q_p_levels = np.array([0.0, 33.33, 66.66, 100])

        # Constants and bounds
        self.B_p = 1000 * 10
        self.h_p_const = 1
        # Tank Constants
        self.Vt_max = 5
        self.Vt_min = 1

        # Batteries Constants
        Pbat_nom = 100
        self.Pbat_max = 30  # [Kw]
        self.SoE_max = 0.8 * Pbat_nom
        self.SoE_min = 0.2 * Pbat_nom

        # Irrigation Constants

        self.I_max = 100  # 1 / 1000  # 1L / s -> 0.001m3 / s
        self.I_min = 0
        self.d_I_bound = 1e-3  # I_max 
        self.Q_p_max = 100  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
        self.d_Q_p_bound = 1e-3  # Q_p_max

        self.reward_fun = lambda s, a, s_next: np.array([(np.exp(-1*(s[0] - s_next[4]) ** 2 / (s[0]+1e-5) ** 2)) +
                                                         (np.exp(-2*(s[0] - s_next[4]) ** 2 / (s[0]+1e-5) ** 2)) -
                                                          0.5*np.sign(np.min([np.abs(s[0] - s_next[4]) - np.abs((s[0] - s[4])), 0])) -
                                                          3.0*np.sign(np.max([np.abs(s[0] - s_next[4]) - np.abs((s[0] - s[4])), 0]))])

        # Observation space
        # V_ref, Vt, SoE, I_prev, V_Irr, Q_p_prev, p_fv, demand

        # Bounds for observations
        obs_low = np.array([0.0,
                            self.Vt_min, self.SoE_min,
                            self.I_min,
                            0.0,
                            0.0, 0.0, 0.0,
                            0, 0, 0, 0], dtype=np.float32)

        obs_high = np.array([self.Vt_max,
                             self.Vt_max, self.SoE_max,
                             self.I_max,
                             self.Vt_max,
                             self.Q_p_max, 1000, 1000,
                             0, 0, 0, 143], dtype=np.float32)

        # Action space
        # Irr, Q_p, Pbat

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0, -self.Pbat_max],
                                   dtype=np.float32)

        self.action_high = np.array([100, 100, self.Pbat_max],
                                    dtype=np.float32)

        super().__init__(self.action_low, self.action_high)

        # Agent params
        if continuous:
            self.action_space = spaces.Box(low=self.action_low,
                                           high=self.action_high,
                                           shape=(3,),
                                           dtype=np.float32)

        else:
            self.action_space = spaces.Discrete(len(self.Irr_levels) * len(self.Q_p_levels))

        self.action_values = np.array(np.meshgrid(self.Irr_levels, self.Q_p_levels), dtype=np.float32).T.reshape(-1, 2)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(12,),
                                            dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[ndarray, ndarray, bool, bool, dict[str, Any]]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        if not self.isContinuous:
            action = self.map_action(action)
            action = action.flatten()

        truncated = False
        terminated = False

        Info = {}

        observation = np.array([self.V_ref,
                                self.Vt, self.SoE,
                                self.Irr,
                                self.V_Irr,
                                self.Q_p, self.p_fv[self.k],
                                self.demanda[self.k], self.Pbat,
                                self.E_surplus, self.E_deficit, self.k % 144])

        # Store the previous values of the variables to compute the reward
        Irr_prev = self.Irr
        Q_p_prev = self.Q_p
        V_Irr_prev = self.V_Irr

        self.Irr = action[0]
        self.Q_p = action[1]

        P_Q_p = self.B_p * (self.Q_p * 1e-6) * self.h_p_const / 1e3  # water pump power [kW]

        self.Pbat, self.SoE, self.E_surplus, self.E_deficit = self.manage_batteries(self.SoE,
                                                                                    self.p_fv[self.k],
                                                                                    self.demanda[self.k],
                                                                                    P_Q_p)

        #irrigation_penalty = 0

        if self.Vt <= self.Vt_min:  # If the tank is empty, there is no irrigation
            if self.Irr > 0:
                self.Irr = 0
                #irrigation_penalty = 10  # -300

        amount_to_irrigate = self.dt * (self.Irr * 1e-6)

        extraction_penalty = 0
        Vt_to_fill = self.Vt_max - self.Vt - amount_to_irrigate  # Amount of water that can be filled
        if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
            self.Q_p = 0
            extraction_penalty = 10  # -300

        amount_to_pump = self.dt * (self.Q_p * 1e-6)

        self.Vt = np.clip(self.Vt + amount_to_pump - amount_to_irrigate, self.Vt_min, self.Vt_max)
        self.V_Irr = self.V_Irr + amount_to_irrigate

        reward = get_reward(self.E_surplus,
                            self.E_deficit,
                            Irr_prev,
                            self.Irr,
                            self.V_Irr,
                            self.V_ref,
                            Q_p_prev, self.Q_p)

        if self.k == self.n_steps // 2:  # A day has passed
            self.V_ref = self.V_2_ref
            self.V_Irr = 0

        self.k = self.k + 1

        observation_next = np.array([self.V_ref,
                                     self.Vt, self.SoE,
                                     self.Irr,
                                     self.V_Irr,
                                     self.Q_p, self.p_fv[self.k],
                                     self.demanda[self.k], self.Pbat,
                                     self.E_surplus, self.E_deficit, self.k % 144])

        reward = self.reward_fun(observation, action, observation_next)

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
        self.k = 0
        day_picked = np.random.randint(0, 70)  # Pick a random day from the data
        self.start_index = day_picked * 144  # 144 is the number of steps per day

        self.V_ref = self.V_refs[day_picked]
        self.V_2_ref = self.V_refs[day_picked + 1]
        self.Vt = (self.Vt_max - self.Vt_min) * np.random.random_sample() + self.Vt_min
        self.SoE = (self.SoE_max - self.SoE_min) * np.random.random_sample() + self.SoE_min
        self.Irr = 0
        self.V_Irr = 0
        self.radiacion = self.radiation_data[self.start_index:self.start_index + self.n_steps + 1]
        self.temperatura = self.temperature_data[self.start_index:self.start_index + self.n_steps + 1]
        self.p_fv = solar_power(self.radiacion, self.temperatura)
        self.demanda = self.demand_data[self.start_index:self.start_index + self.n_steps + 1]
        self.Q_p = 0
        self.d_Q_p = 0
        self.Pbat = 0

        InitialObservation = np.array(
            [self.V_ref,
             self.Vt, self.SoE,
             self.Irr,
             self.V_Irr,
             self.Q_p, self.p_fv[self.k],
             self.demanda[self.k], self.Pbat,
             0, 0, self.k])
        return InitialObservation, info

    def show_sample(self, policy):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """

        states, actions, rewards = self.sample_trajectory(policy, max_steps=288, rew_fun=self.reward_fun)
        for ax in self.axs.flat:
            ax.clear()

        t = np.linspace(0, 48, 288)

        SoE = states[:, 2]
        Qp = np.diff(states[:, 1]) + self.dt * actions[:, 0] * 1e-6
        P_Q_p = self.dt * self.B_p * Qp / self.dt * self.h_p_const / 1e3
        P_fv = states[:-1, 6]  # foto-voltaic generation
        P_demanda = states[:-1, 7]  # Energetic demand
        P_bat = states[:-1, 8]  # Battery power

        P_residual = P_fv - P_demanda - P_Q_p - P_bat  # If it is positive there is an energy surplus. Otherwise,

        self.axs[0, 0].step(t, states[:-1, 0], where='post', label='V_ref')
        self.axs[0, 0].step(t, states[:-1, 4], where='post', label='V_Irr')
        self.axs[0, 0].set_title('Water demand fulfilled')
        # self.axs[0].set_xlabel('Time (h)')
        self.axs[0, 0].set_ylabel('Water volume (m3)')

        self.axs[0, 1].step(t, actions[:, 0], where='post', label='Irr')
        self.axs[0, 1].set_title('Irrigation level')
        # self.axs[1].set_xlabel('Time (h)')
        self.axs[0, 1].set_ylabel('Irrigation level (%)')
        self.axs[0, 1].legend()

        self.axs[1, 0].step(t, SoE[:-1], where='post', label='Soe')
        self.axs[1, 0].set_title('SoE batteries')
        # self.axs[2].set_xlabel('Time (h)')
        self.axs[1, 0].set_ylabel('SoE (kWh)')

        self.axs[1, 1].step(t, P_bat, where='post', label='Pbat')
        self.axs[1, 1].set_title('Battery power')
        # self.axs[3].set_xlabel('Time (h)')
        self.axs[1, 1].set_ylabel('Power (kW)')

        self.axs[2, 0].step(t, states[:-1, 1], where='post', label='V_tank')
        self.axs[2, 0].set_title('Tank volume')
        self.axs[2, 0].set_xlabel('Time (h)')
        self.axs[2, 0].set_ylabel('Volume (m3)')

        self.axs[2, 1].step(t, actions[:, 1], where='post', label='Q_pump')
        self.axs[2, 1].set_title('Pump power')
        self.axs[2, 1].set_xlabel('Time (h)')
        self.axs[2, 1].set_ylabel('(%)')

        self.axs[3, 0].step(t, P_residual, where='post', label='P_balance')
        self.axs[3, 0].set_title('Power balance')

        # balance = np.cumsum(P_residual * self.dt / 3600)
        # self.axs[3, 1].step(t, states[:-1, 9] + states[:-1, 10], where='post', label='E_residual')
        # self.axs[3, 1].set_title('Energy balance')

        self.axs[3, 1].step(t, rewards, where='post', label='E_residual')
        self.axs[3, 1].set_title('Transition Rewards')

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

    def map_action(self, action: torch.Tensor) -> np.ndarray:
        """
            Map the action from the policy to the action of the environment
            :param action:
            :return:
            """
        index = action.item()
        return np.array([self.action_values[index]])

    def manage_batteries(self,
                         SoE: float,
                         P_fv: float,
                         P_demanded: float,
                         P_pump: float) -> tuple[float, float, float, float]:
        """
        Choose the power to re/discharge the batteries and computes the next SoE
        :param SoE:
        :param P_fv:
        :param P_demanded:
        :param P_pump:
        :return:
        """
        E_surplus = 0
        E_deficit = 0
        Pbat = P_fv - P_demanded - P_pump
        if not -self.Pbat_max <= Pbat <= self.Pbat_max:  # The surplus is under the power of power bounds of the battery
            P_not_used = Pbat - np.clip(Pbat, -self.Pbat_max,
                                        self.Pbat_max)  # positive for surplus, negative for deficit
            Pbat = np.clip(Pbat, -self.Pbat_max, self.Pbat_max)
        else:
            P_not_used = 0
        delta_SoE = np.max([Pbat, 0]) * self.n_c * (self.dt / 3600) + np.min([Pbat, 0]) / self.n_d * (self.dt / 3600)
        next_SoE = SoE + delta_SoE
        if self.SoE_min <= next_SoE <= self.SoE_max:  # The recharge is done immediately
            Pbat = Pbat
        else:  # The re/discharge is done but there is a surplus/deficit of energy
            E_surplus = next_SoE - self.SoE_max if next_SoE > self.SoE_max else 0
            E_deficit = next_SoE - self.SoE_min if next_SoE < self.SoE_min else 0

            next_SoE = np.clip(next_SoE, self.SoE_min, self.SoE_max)
            Pbat = np.max([next_SoE - SoE, 0]) / (self.dt / 3600) / self.n_c + np.min(
                [next_SoE - SoE, 0]) * self.n_d / (self.dt / 3600)

        E_surplus = E_surplus + P_not_used * self.n_c * (self.dt / 3600) if P_not_used > 0 else E_surplus
        E_deficit = E_deficit - P_not_used / self.n_d * (self.dt / 3600) if P_not_used < 0 else E_deficit

        return Pbat, next_SoE, E_surplus, E_deficit
