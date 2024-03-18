from typing import Any, Union, Tuple, Callable, List, Iterable

import copy
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import torch
import matplotlib
from sklearn.preprocessing import StandardScaler

from utils_functions.funcionesEMS import *
from gymnasium import spaces
from environments.custom_env import Custom_env
matplotlib.use('Qt5Agg')


class EMS_env(Custom_env):
    """
    Environment for the Energy Management System
    """

    def __init__(self, energy_penalty=0, render: bool = True, continuous: bool = False):
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
        self.V_3_ref = 0.0
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
        self.E_residual = 0.0
        self.Irr_levels = np.array([0.0, 10.0, 20.0, 30.0])
        self.Q_p_levels = np.array([0.0, 33.33, 66.66, 100])
        self.day_picked = 0

        # Constants and bounds
        self.B_p = 1000 * 10 #1e5
        self.h_p_const = 20
        # Tank Constants
        self.Vt_max = 5
        self.Vt_min = 1

        # Batteries Constants
        Pbat_nom = 10 # 100
        self.Pbat_max = 10 # 100  # [Kw]
        self.SoE_max = Pbat_nom
        self.SoE_min = 0.2 * Pbat_nom

        # Irrigation Constants

        self.I_max = 100  # 1 / 1000  # 1L / s -> 0.001m3 / s
        self.I_min = 0
        self.d_I_bound = 1e-3  # I_max 
        self.Q_p_max = 100  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
        self.d_Q_p_bound = 1e-3  # Q_p_max

        def rwd_fun2(s, a, s_next, e_penal=energy_penalty):

            penalty = 0
            reward = 0

            # penalty for activating the pump when the tank is full 
            if a[1] > 0 and s[1] >= self.Vt_max:
                penalty += 50

            # penalty for irrigate when the tank is empty
            if a[0] > 0 and s[1] <= self.Vt_min:
                penalty += 50

            #reward for having water in the tank
            if self.Vt_min < s_next[1] <= self.Vt_max:
                reward = 50 * np.exp(-0.1*(self.Vt_max-s_next[1])**2/(self.Vt_max**2))

            # reward for ending the day close to the reference
            if s_next[-1] == 143:
                reward = 100 * np.exp(-.1 * (s_next[4] - s_next[0]) ** 2 / (s_next[0] + 1e-5) ** 2) + 100 * np.exp(
                    -.3 * (s_next[4] - s_next[0]) ** 2 / (s_next[0] + 1e-5) ** 2)
                if s_next[10] != 0:
                    reward = reward - e_penal
            
            else:
                if s_next[-2] < 0:
                    penalty += 50

            return np.array([reward - penalty], dtype=np.float32) 

        def rwd_fun(s, a, s_next, e_penal=energy_penalty):
            next_error = s[0] - s_next[4]
            current_error = s[0] - s[4]
            delta_error = np.abs(next_error) - np.abs(current_error)
            delta_Irr = (a[0] - s[3]) / 100
            delta_Q_p = (a[1] - s[5]) / 100

            if s[-1] != 143:
                reward = (np.exp(-1 * next_error ** 2 / (s[0] + 1e-5) ** 2 -
                                 delta_Irr ** 2 - delta_Q_p ** 2) +
                          2 * np.exp(-3 * next_error ** 2 / (s[0] + 1e-5) ** 2 -
                                     2 * delta_Irr ** 2 - 2 * delta_Q_p ** 2) -
                          0.5 * np.sign(np.min([delta_error, 0])) -
                          3.0 * np.sign(np.max([delta_error, 0])))
                if s[10] != 0:
                    reward = reward - e_penal
            else:
                reward = (np.exp(-1 * current_error ** 2 / (s[0] + 1e-5) ** 2 -
                                 delta_Irr ** 2 - delta_Q_p ** 2) +
                          2 * np.exp(-3 * current_error ** 2 / (s[0] + 1e-5) ** 2) -
                          2 * delta_Irr ** 2 - 2 * delta_Q_p ** 2)
                if s_next[10] != 0:
                    reward = reward - e_penal

            return np.array([reward])

        self.reward_fun = lambda s, a, s_next: rwd_fun2(s, a, s_next, energy_penalty)

        # Observation space
        # V_ref, Vt, SoE, I_prev, V_Irr, Q_p_prev, p_fv, demand

        # Bounds for observations
        obs_low = np.array([0.0,
                            self.Vt_min, self.SoE_min,
                            self.I_min,
                            0.0,
                            0.0, 0.0, 0.0,
                            0, 0, 0], dtype=np.float32)

        obs_high = np.array([self.Vt_max,
                             self.Vt_max, self.SoE_max,
                             self.I_max,
                             self.Vt_max,
                             self.Q_p_max, 1000, 1000,
                             0, 0, 143], dtype=np.float32)

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
                                            shape=(11,),
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
                                self.E_residual, self.k % 144])

        # Store the previous values of the variables to compute the reward
        self.Irr = action[0]
        self.Q_p = action[1]

        if self.Vt <= self.Vt_min:  # If the tank is empty, there is no irrigation
            if self.Irr > 0:
                self.Irr = 0

        amount_to_irrigate = self.dt * (self.Irr * 1e-5)

        Vt_to_fill = self.Vt_max - self.Vt - amount_to_irrigate  # Amount of water that can be filled
        if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
            self.Q_p = 0

        P_Q_p = self.B_p * (self.Q_p * 1e-5) * self.h_p_const / 1e3  # water pump power [kW]

        self.Pbat, self.SoE, self.E_residual = self.manage_batteries(self.SoE,
                                                                                    self.p_fv[self.k],
                                                                                    self.demanda[self.k],
                                                                                    P_Q_p)

        amount_to_pump = self.dt * (self.Q_p * 1e-5) # Volume [m3]

        self.Vt = np.clip(self.Vt + amount_to_pump - amount_to_irrigate, self.Vt_min, self.Vt_max)
        self.V_Irr = self.V_Irr + amount_to_irrigate

        self.k = self.k + 1

        observation_next = np.array([self.V_ref,
                                     self.Vt, self.SoE,
                                     self.Irr,
                                     self.V_Irr,
                                     self.Q_p, self.p_fv[self.k],
                                     self.demanda[self.k], self.Pbat,
                                     self.E_residual, self.k % 144])

        reward = self.reward_fun(observation, action, observation_next)

        if (self.k % 144) == 0:  # A day has passed
            self.V_ref = self.V_refs[self.day_picked + self.k//144]
            self.V_Irr = 0
            observation_next = np.array([self.V_ref,
                                         self.Vt, self.SoE,
                                         self.Irr,
                                         self.V_Irr,
                                         self.Q_p, self.p_fv[self.k],
                                         self.demanda[self.k], self.Pbat,
                                         self.E_residual, self.k % 144])

        if self.k >= self.n_steps:  # Number of steps are completed
            truncated = True

        return observation_next, reward, terminated, truncated, Info

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state
        :param seed: random seed
        :param options: options for the environment
        :return: tuple of (initial_observation, info)
        """
        self.day_picked = np.random.randint(0, 70)

        InitialObservation = self.set_initial_conditions(self.day_picked,
                                                         V_tank=(self.Vt_max - self.Vt_min) * np.random.random_sample() + self.Vt_min,
                                                         Soe=(self.SoE_max - self.SoE_min) * np.random.random_sample() + self.SoE_min,
                                                         Irr_prev=self.Irr_levels[np.random.randint(0, 4)],
                                                         Q_p_prev=self.Q_p_levels[np.random.randint(0, 4)],
                                                         instant_k=np.random.randint(0, 144),
                                                         V_irr=self.V_refs[self.day_picked] * np.random.random_sample())

        info = {"day_picked": self.day_picked,
                "V_tank": self.Vt,
                "SoE": self.SoE,
                "Irr_prev": self.Irr,
                "Q_p_prev": self.Q_p,
                "instant_k": self.k,
                "V_irr": self.V_Irr}

        return InitialObservation, info

    def load_initial_conditions(self, info):
        """
        Load the initial conditions of the environment
        :param info: dictionary with the initial conditions
        :return:
        """
        self.k = 0
        day_picked = info["day_picked"]
        InitialObservation = self.set_initial_conditions(day_picked,
                                                         V_tank=info["V_tank"],
                                                         Soe=info["SoE"],
                                                         Irr_prev=info["Irr_prev"],
                                                         Q_p_prev=info["Q_p_prev"],
                                                         instant_k=info["instant_k"],
                                                         V_irr=info["V_irr"]
                                                         )

        return InitialObservation

    def set_initial_conditions(self, day_picked, V_tank, Soe, Irr_prev, Q_p_prev, instant_k, V_irr):
        """
        Set the initial conditions of the environment
        :param V_irr:
        :param instant_k:
        :param day_picked:
        :param V_tank:
        :param Soe:
        :param Irr_prev:
        :param Q_p_prev:
        :return: Initial observation
        """
        self.k = instant_k
        self.n_steps = self.k + 288
        self.start_index = day_picked * 144
        self.Vt = V_tank
        self.SoE = Soe
        self.Irr = Irr_prev
        self.V_Irr = V_irr
        self.radiacion = self.radiation_data[self.start_index:self.start_index + self.n_steps + 1]
        self.temperatura = self.temperature_data[self.start_index:self.start_index + self.n_steps + 1]
        self.p_fv = solar_power(self.radiacion, self.temperatura)
        self.demanda = self.demand_data[self.start_index:self.start_index + self.n_steps + 1]
        self.Q_p = Q_p_prev
        self.V_ref = self.V_refs[day_picked]
        self.Pbat, _, self.E_residual = self.manage_batteries(self.SoE, self.p_fv[self.k], self.demanda[self.k], 0)

        InitialObservation = np.array(
            [self.V_ref,
             self.Vt, self.SoE,
             self.Irr,
             self.V_Irr,
             self.Q_p, self.p_fv[self.k],
             self.demanda[self.k], self.Pbat,
             self.E_residual, self.k])

        return InitialObservation

    def show_sample(self, policy, scaler: StandardScaler = None):
        """
        Render the environment
        :param policy: policy to be used
        :param scaler: scaler to be used
        :return:
        """

        states, actions, rewards = self.sample_trajectory(policy, scaler = scaler, max_steps=288, rew_fun=self.reward_fun)
        for ax in self.axs.flat:
            ax.clear()

        t = np.linspace(0, 48, len(actions))

        SoE = states[:, 2]
        Qp = self.dt * (actions[:, 0]/100) * 1e-3
        P_Q_p = self.B_p * Qp * self.h_p_const / 1e3
        P_fv = states[:-1, 6]  # foto-voltaic generation
        P_demanda = states[:-1, 7]  # Energetic demand
        P_bat = states[:-1, 8]  # Battery power

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

        self.axs[3, 0].step(t, states[:-1, 9], where='post', label='E_residual')
        self.axs[3, 0].set_title('Power balance')

        # balance = np.cumsum(P_residual * self.dt / 3600)
        # self.axs[3, 1].step(t, states[:-1, 9] + states[:-1, 10], where='post', label='E_residual')
        # self.axs[3, 1].set_title('Energy balance')

        self.axs[3, 1].step(t, rewards, where='post', label='rewards')
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
        :return: Pbat, next_SoE, E_surplus, E_deficit
        """
        E_surplus = 0
        E_deficit = 0

        P_residual = P_fv - P_demanded - P_pump
        if not -self.Pbat_max <= P_residual <= self.Pbat_max:  # The surplus is out of the power bounds of the battery
            Pbat = np.clip(P_residual, -self.Pbat_max, self.Pbat_max)  # positive for surplus, negative for deficit
            P_not_used = P_residual - Pbat  # positive for surplus, negative for deficit. In case of negative value is P not available

        else:
            P_not_used = 0
            Pbat = P_residual

        delta_SoE = np.max([Pbat, 0]) * self.n_c * (self.dt / 3600) + np.min([Pbat, 0]) / self.n_d * (self.dt / 3600)
        next_SoE = SoE + delta_SoE

        if self.SoE_min <= next_SoE <= self.SoE_max:  # The recharge is done immediately
            Pbat = Pbat
        else:  # The re/discharge is done but there is a surplus/deficit of energy
            E_surplus = next_SoE - self.SoE_max if next_SoE > self.SoE_max else 0
            E_deficit = next_SoE - self.SoE_min if next_SoE < self.SoE_min else 0

            next_SoE = np.clip(next_SoE, self.SoE_min, self.SoE_max)
            if delta_SoE > 0:
                Pbat = np.max([self.SoE_max - SoE, 0]) / (self.dt / 3600) / self.n_c
            else:
                Pbat = np.min([self.SoE_min - SoE, 0]) * self.n_d / (self.dt / 3600)
            #Pbat = np.max([self.SoE_max - SoE, 0]) / (self.dt / 3600) / self.n_c + np.min([self.SoE_min - SoE, 0]) * self.n_d / (self.dt / 3600)

        E_surplus = E_surplus + P_not_used * (self.dt / 3600) if P_not_used > 0 else E_surplus
        E_deficit = E_deficit + P_not_used / self.n_d * (self.dt / 3600) if P_not_used < 0 else E_deficit

        E_residual = E_surplus + E_deficit 

        return Pbat, next_SoE, E_residual

    def compare_policies(self,
                         policies: Iterable[Tuple[Callable[[Union[np.ndarray, torch.Tensor]], Union[ndarray, torch.Tensor]]]],
                         max_steps: int = 288,
                         rew_funs: Iterable[Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]] = None) -> list:
        """
        Compare the policies in the environment
        :param policies: list of policies to be compared
        :param max_steps: maximum number of steps to be taken
        :param rew_funs: set of reward functions to be used
        :return:
        """
        # initialize environment
        x0, info = self.reset()
        envs = [self]
        trajectories = []

        # make copies of the environment so they have same initial conditions
        for i in range(1, len(policies)):
            envs.append(copy.copy(self))
            envs[i].load_initial_conditions(info)

        # Run the policies in the environment

        if rew_funs is None:
            for idx, env in enumerate(envs):
                policy = policies[idx][0]
                scaler = policies[idx][1]
                s, a, r = env.sample_trajectory(policy, scaler,
                                                max_steps,
                                                self.reward_fun,
                                                initial_conditions=info)
                trajectories.append([s, a, r])

        else:
            for idx, env in enumerate(envs):
                policy = policies[idx][0]
                scaler = policies[idx][1]
                s, a, r = env.sample_trajectory(policy, scaler,
                                                max_steps,
                                                rew_funs[idx],
                                                initial_conditions=info)
                trajectories.append([s, a, r])

        return trajectories
