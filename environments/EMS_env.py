from typing import Any, Union, Tuple, Callable, List, Iterable
from environments.EMS_constants import *

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


class EMS_env(Custom_env):
    """
    Environment for the Energy Management System
    """

    def __init__(self, rwd_function = None, render: bool = True, continuous: bool = False):
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
        self.n_steps = 144  # 288
        self.start_index = 0
        self.k = 0
        self.dt = dt
        self.n_c = n_c
        self.n_d = n_d
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
        self.B_p = B_p
        self.h_p_const = h_p_const
        # Tank Constants
        self.Vt_max = 5
        self.Vt_min = 1

        # Batteries Constants
        self.Pbat_max = Pbat_max  # 100  # [Kw]
        self.SoE_max = SoE_max
        self.SoE_min = SoE_min

        # Irrigation Constants

        self.I_max = I_max
        self.I_min = I_min
        self.d_I_bound = d_I_bound  
        self.Q_p_max = Q_p_max  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
        self.d_Q_p_bound = d_Q_p_bound  # Q_p_max        

        def default_rwd_fun(s, a, s_next):
            """ Default reward function 
            :param s: current state
            :param a: action
            :param s_next: next state
            :param e_penal: penalty for energy deficit"""
            reward = 0
            next_error = s[0] - s_next[1]
            current_error = s[0] - s[1]
            delta_error = np.abs(next_error) - np.abs(current_error)
            reward = 0.0

            if s_next[-1] != 143:
                reward = (np.exp(-1 * next_error ** 2 / (s[0] + 1e-5) ** 2) +
                          2 * np.exp(-3 * next_error ** 2 / (s[0] + 1e-5) ** 2) -
                          0.5 * np.sign(np.min([delta_error, 0])) -
                          3.0 * np.sign(np.max([delta_error, 0])))
            return np.array([reward], dtype=np.float32)

        if rwd_function is not None:
            self.reward_fun = rwd_function
        else:
            self.reward_fun = lambda s, a, s_next: default_rwd_fun(s, a, s_next)


        # Bounds for observations
        obs_low = np.array([0.0,
                            self.Vt_min, self.SoE_min,
                            self.I_min,
                            0.0,
                            0.0, 0.0, 0.0,
                            0, 0], dtype=np.float32)

        obs_high = np.array([self.Vt_max,
                             self.Vt_max, self.SoE_max,
                             self.I_max,
                             self.Vt_max,
                             self.Q_p_max, 1000, 1000,
                             0, 143], dtype=np.float32)

        # Action space
        # Irr, Q_p, Pbat

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0],
                                   dtype=np.float32)

        self.action_high = np.array([100, 100],
                                    dtype=np.float32)

        super().__init__(self.action_low, self.action_high)

        # Agent params
        if continuous:
            self.action_space = spaces.Box(low=self.action_low,
                                           high=self.action_high,
                                           shape=(2,),
                                           dtype=np.float32)

        else:
            self.action_space = spaces.Discrete(len(self.Irr_levels) * len(self.Q_p_levels))

        self.action_values = np.array(np.meshgrid(self.Irr_levels, self.Q_p_levels), dtype=np.float32).T.reshape(-1, 2)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(10,),
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
                                self.V_Irr,
                                self.Irr,
                                self.Vt, 
                                self.Q_p,
                                self.SoE, 
                                self.Pbat,
                                self.p_fv[self.k],
                                self.demanda[self.k], 
                                self.k % 144])

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

        self.Pbat, self.SoE, self.E_residual = manage_batteries(self.SoE,
                                                                     self.p_fv[self.k],
                                                                     self.demanda[self.k],
                                                                     P_Q_p)

        amount_to_pump = self.dt * (self.Q_p * 1e-5) # Volume [m3]

        self.Vt = np.clip(self.Vt + amount_to_pump - amount_to_irrigate, self.Vt_min, self.Vt_max)
        self.V_Irr = self.V_Irr + amount_to_irrigate

        self.k = self.k + 1

        if (self.k % 144) == 143:  # A day is over
            terminated = True

        observation_next = np.array([self.V_ref,
                                    self.V_Irr,
                                    self.Irr,
                                    self.Vt, 
                                    self.Q_p,
                                    self.SoE, 
                                    self.Pbat,
                                    self.p_fv[np.min((self.k, 143))],
                                    self.demanda[np.min((self.k, 143))], 
                                    self.k % 144])

        reward = self.reward_fun(observation, action, observation_next)

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
                                                         V_ref=5*np.random.rand(),
                                                        V_tank=(self.Vt_max - self.Vt_min) * np.random.random_sample() + self.Vt_min,
                                                        Soe=(self.SoE_max - self.SoE_min) * np.random.random_sample() + self.SoE_min,
                                                        Irr_prev=self.Irr_levels[np.random.randint(0, 4)],
                                                        Q_p_prev=self.Q_p_levels[np.random.randint(0, 4)],
                                                        instant_k=0,
                                                        V_irr=0.0)

        info = {"day_picked": self.day_picked,
                "V_tank": self.Vt,
                "SoE": self.SoE,
                "Irr_prev": self.Irr,
                "Q_p_prev": self.Q_p,
                "instant_k": self.k,
                "V_irr": self.V_Irr, 
                "V_ref": self.V_ref}

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
                                                         V_ref=info["V_ref"],
                                                         V_tank=info["V_tank"],
                                                         Soe=info["SoE"],
                                                         Irr_prev=info["Irr_prev"],
                                                         Q_p_prev=info["Q_p_prev"],
                                                         instant_k=info["instant_k"],
                                                         V_irr=info["V_irr"]
                                                         )

        return InitialObservation

    def set_initial_conditions(self, day_picked, V_ref, V_tank, Soe, Irr_prev, Q_p_prev, instant_k, V_irr):
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
        self.n_steps = self.k + 143
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
        self.V_ref = V_ref #self.V_refs[day_picked]
        self.Pbat, _, self.E_residual = manage_batteries(self.SoE, self.p_fv[self.k], self.demanda[self.k], 0)

        InitialObservation = np.array([self.V_ref,
                                self.V_Irr,
                                self.Irr,
                                self.Vt, 
                                self.Q_p,
                                self.SoE, 
                                self.Pbat,
                                self.p_fv[self.k],
                                self.demanda[self.k], 
                                self.k])

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

        SoE = states[:, 5]
        Qp = np.diff(states[:,3])/(600)*1e5
        P_Q_p = self.B_p * Qp * self.h_p_const / 1e3
        P_bat = states[1:, 6]  # Battery power

        self.axs[0, 0].step(t, states[:-1, 0], where='post', label='V_ref')
        self.axs[0, 0].step(t, states[:-1, 1], where='post', label='V_Irr')
        self.axs[0, 0].set_title('Water demand fulfilled')
        # self.axs[0].set_xlabel('Time (h)')
        self.axs[0, 0].set_ylabel('Water volume (m3)')
        actual_irrigation = np.diff(states[:, 1])/(600)*1e5
        self.axs[0, 1].step(t, actions[:, 0], where='post', label='Irr')
        self.axs[0, 1].step(t, actual_irrigation, where='post', label='Actual_Irr')
        self.axs[0, 1].set_title('Irrigation level')
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

        self.axs[2, 0].step(t, states[:-1, 3], where='post', label='V_tank')
        self.axs[2, 0].set_title('Tank volume')
        self.axs[2, 0].set_xlabel('Time (h)')
        self.axs[2, 0].set_ylabel('Volume (m3)')

        self.axs[2, 1].step(t, actions[:, 1], where='post', label='Q_pump')
        self.axs[2, 1].step(t, Qp + actual_irrigation, where='post', label='Actual_Q_pump')
        self.axs[2, 1].set_title('Pump')
        self.axs[2, 1].set_xlabel('Time (h)')
        self.axs[2, 1].set_ylabel('(%)')

        self.axs[3, 0].step(t, states[:-1, 9], where='post', label='E_residual')
        self.axs[3, 0].set_title('Power balance')


        self.axs[3, 1].step(t, rewards, where='post', label='rewards')
        self.axs[3, 1].set_title('Transition Rewards')

        self.axs[0, 0].legend()
        self.axs[0, 1].legend()
        self.axs[1, 0].legend()
        self.axs[1, 1].legend()
        self.axs[2, 0].legend()
        self.axs[2, 1].legend()

        
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


    def compare_policies(self,
                         policies: Iterable[Tuple[Callable[[Union[np.ndarray, torch.Tensor]], Union[ndarray, torch.Tensor]]]],
                         max_steps: int = 144,
                         rew_funs: Iterable[Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]] = None, 
                         options = None) -> list:
        """
        Compare the policies in the environment
        :param policies: list of policies to be compared
        :param max_steps: maximum number of steps to be taken
        :param rew_funs: set of reward functions to be used
        :return:
        """
        # initialize environment
        if options is not None:
            x0, info = self.reset(options=options)
        else:
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
    
