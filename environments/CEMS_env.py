from typing import Any, Union, Tuple, Callable, List, Iterable
from environments.EMS_constants import *

import copy
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from utils_functions.funcionesEMS import *
from gymnasium import spaces
from environments.custom_env import ContinousCustomEnv
from matplotlib import figure

class ContinousEMSEnv(ContinousCustomEnv):
    """
    Environment for the Energy Management System
    """

    def __init__(self, rwd_function = None, render: bool = True):
        """
        Initialize the environment
        :param render:
        """

        self.fig: figure.Figure = None
        self.axs = []
        self.lines = None
        if render:
            self.fig = figure.Figure()
            self.axs = [self.fig.add_subplot(4, 2, i+1) for i in range(4*2)]
            self.fig.suptitle('Energy Management System')
            self.fig.tight_layout()
            self.fig.set_size_inches(10, 10)
            self.lines = []

            # lines for the reference traking
            self.lines.append(self.axs[0].plot([], [], label="V_ref")[0])
            self.lines.append(self.axs[0].plot([], [], label="V_Irr")[0])

            # lines for the irrigation
            self.lines.append(self.axs[1].plot([], [], label="Irrigation")[0])

            # lines for soe
            self.lines.append(self.axs[2].plot([], [], label="SoE")[0])

            # lines for the power consumed by the community
            self.lines.append(self.axs[3].plot([], [], label="P_d")[0])
            self.lines.append(self.axs[3].plot([], [], label="P_pump")[0])
            self.lines.append(self.axs[3].plot([], [], label="P_sun")[0])

            # lines for the tank volume
            self.lines.append(self.axs[4].plot([], [], label="V_tank")[0])

            # lines for the pump
            self.lines.append(self.axs[5].plot([], [], label="Q_pump")[0])

            # lines for the power balance
            self.lines.append(self.axs[6].plot([], [], label="E_residual")[0])

            # lines for the rewards
            self.lines.append(self.axs[7].plot([], [], label="Rewards")[0])

        self.Irr_levels = np.array([0.0, 10.0, 20.0, 30.0])
        self.Q_p_levels = np.array([0.0, 33.33, 66.66, 100])


        # Hyperparams
        self.max_steps: int = 143
        self.start_index:int = 0
        self.k:int = 0
        self.time:int = 0

        # Load meteorological data and demand

        self.radiation_data:np.ndarray = get_rad('ver')
        self.temperature_data:np.ndarray = get_temperatura('ver')
        self.demand_data:np.ndarray = get_demand()
        self.L = len(self.temperature_data)  # length(temperatura)
        self.N_dias:int = 70

        # Data variables 

        self.p_fv: np.ndarray[float] = np.array([0.0])
        self.temperatura: np.ndarray[float] = np.array([0.0])
        self.demanda: np.ndarray[float] = np.array([0.0])
        self.radiacion: np.ndarray[float] = np.array([0.0])
        self.V_refs = get_ref()  # V_refs 

        # State variables en inputs

        self.Irr = np.array([0.0])
        self.d_Irr = np.array([0.0])
        self.V_Irr = np.array([0.0])
        self.Q_p = np.array([0.0])
        self.d_Q_p = np.array([0.0])
        self.Pbat = np.array([0.0])
        self.V_ref = np.array([0.0])
        self.Vt = np.array([0.0])
        self.SoE = np.array([0.0])
        self.E_Q_p = 0.0
        self.State = np.array([0.0])
        self.E_residual = 0.0
        self.Irr_levels = np.array([0.0, 10.0, 20.0, 30.0])
        self.Q_p_levels = np.array([0.0, 33.33, 66.66, 100])
        self.day_picked = 0
        self.deficit_flag = 0

        self.accumulated_P_pump = 0.0
        self.accumulated_P_consumed = 0.0

        if rwd_function is not None:
            self.reward_fun = rwd_function
        else:
            self.reward_fun = lambda s, a, s_next: default_rwd_fun(s, a, s_next)


        # Bounds for observations
        obs_low = np.array([0.0,
                            Vt_min, SoE_min,
                            I_min,
                            0.0,
                            0.0, 0.0, 0.0,
                            -1, -1, 0], dtype=np.float32)

        obs_high = np.array([Vt_max,
                             Vt_max, SoE_max,
                             I_max,
                             Vt_max,
                             Q_p_max, 1000, 1000,
                             1, 1, 1], dtype=np.float32)

        # Action space
        # Irr, Q_p, Pbat

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0],
                                   dtype=np.float32)

        self.action_high = np.array([1, 1],
                                    dtype=np.float32)

        super().__init__(self.action_low, self.action_high)

        # Agent params
        self.action_space = spaces.Box(low=self.action_low,
                                           high=self.action_high,
                                           shape=(2,),
                                           dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(11,),
                                            dtype=np.float32)
        
        self.action_space = spaces.Discrete(16)
        #super().__init__(np.array([0.0, 25., 50., 75., 100.], dtype=np.float32))
        self.action_values = np.array(np.meshgrid(self.Q_p_levels, self.Irr_levels), dtype=np.float32).T.reshape(-1, 2)

    def map_action(self, action: torch.Tensor|np.ndarray) -> np.ndarray:
        """
            Map the action from the policy to the action of the environment
            :param action:
            :return:
            """
        index = action.item()
        return np.array([self.action_values[index]])

    def step(self, action: np.ndarray) -> tuple[ndarray, ndarray, bool, bool, dict[str, Any]]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """

        truncated = False
        terminated = False

        Info = {}
        moment_of_the_day = self.k % 144
        self.accumulated_P_consumed += self.demanda[self.k]

        # V_ref, V_Irr, Irr, Vt, Q_p, SoE, P_fv, demand, sin, cos, deficit_flag

        observation = np.array([self.V_ref,
                                self.V_Irr,
                                self.Irr,
                                self.Vt, 
                                self.Q_p,
                                self.SoE,
                                self.p_fv[self.k],
                                self.demanda[self.k], 
                                np.sin(2*np.pi*moment_of_the_day/143),
                                np.cos(2*np.pi*moment_of_the_day/143), 
                                self.E_residual])

        # Store the previous values of the variables to compute the reward
        self.Q_p = np.clip(action[0], 0, 1) # [l/s]

        self.Irr = np.clip(action[1], 0, 1) # [l/s] #10. if self.V_ref > self.V_Irr and self.Vt > Vt_min else 0.0 # action[0]

        if self.Vt <= Vt_min:  # If the tank is empty, there is no irrigation
            if self.Irr > 0:
                self.Irr = 0

        amount_to_irrigate = dt * (self.Irr * 1e-3)

        Vt_to_fill = Vt_max - self.Vt - amount_to_irrigate  # Amount of water that can be filled
        if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
            self.Q_p = 0

        P_Q_p = B_p * (self.Q_p * 1e-3) * h_p_const / 1e3  # water pump power [kW]

        self.Pbat, self.SoE, self.E_residual = manage_batteries(self.SoE,
                                                                self.p_fv[self.k],
                                                                self.demanda[self.k],
                                                                P_Q_p)
        
        deficit_flag = 0 if self.E_residual >= 0 else 1

        amount_to_pump = dt * (self.Q_p * 1e-3) # Volume [m3]

        self.Vt = np.clip(self.Vt + amount_to_pump - amount_to_irrigate, Vt_min, Vt_max)
        self.V_Irr = self.V_Irr + amount_to_irrigate

        self.k = self.k + 1
        self.accumulated_P_pump += P_Q_p

        if (self.k % 143) == 0:  # The time at s' is 00:00 i.e. the final day is over
            terminated = True

        observation_next = np.array([self.V_ref,
                                    self.V_Irr,
                                    self.Irr,
                                    self.Vt, 
                                    self.Q_p,
                                    self.SoE,
                                    self.p_fv[self.k],
                                    self.demanda[self.k], 
                                    np.sin(2*np.pi*(self.k % 143)/143),
                                    np.cos(2*np.pi*(self.k % 143)/143),
                                    self.E_residual])

        reward = self.reward_fun(observation, action, observation_next)

        # How much power is consumed by comunities and pumps 

        
       

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
                                                         V_ref=3.5*np.random.rand(),
                                                         V_tank=(Vt_max - Vt_min) * np.random.random_sample() + Vt_min,
                                                         Soe=(SoE_max - SoE_min) * np.random.random_sample() + SoE_min,
                                                         Irr_prev=0.0,
                                                         Q_p_prev=0.0,
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
        
        self.accumulated_P_pump = 0.0
        self.accumulated_P_consumed = 0.0

        return InitialObservation, info

    def load_initial_conditions(self, info):
        """
        Load the initial conditions of the environment
        :param info: dictionary with the initial conditions
        :return:
        """
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
        n_steps = self.k + self.max_steps
        self.start_index = day_picked * 144 # self.max_steps
        self.Vt = V_tank
        self.SoE = Soe
        self.Irr = Irr_prev
        self.V_Irr = V_irr
        self.radiacion = self.radiation_data[self.start_index:self.start_index + n_steps + 1]
        self.temperatura = self.temperature_data[self.start_index:self.start_index + n_steps + 1]
        self.p_fv = solar_power(self.radiacion, self.temperatura)
        self.demanda = self.demand_data[self.start_index:self.start_index + n_steps + 1]
        self.Q_p = Q_p_prev
        self.V_ref = V_ref #self.V_refs[day_picked]
        self.Pbat, _, self.E_residual = manage_batteries(self.SoE, self.p_fv[self.k], self.demanda[self.k], 0)

        InitialObservation = np.array([self.V_ref,
                                self.V_Irr,
                                self.Irr,
                                self.Vt, 
                                self.Q_p,
                                self.SoE,
                                self.p_fv[self.k],
                                self.demanda[self.k], 
                                0.0,
                                1.0,
                                self.E_residual])

        return InitialObservation

    def show_sample(self, policy, scaler: StandardScaler = None):
        """
        Render the environment
        :param policy: policy to be used
        :param scaler: scaler to be used
        :return:
        """

        states, actions, rewards = self.sample_trajectory(policy, scaler = scaler, max_steps=243, rew_fun=self.reward_fun)
        qp = actions[:, 0]
        P_q = B_p * (qp * 1e-3) * h_p_const / 1e3

        t = np.linspace(0, 24, states.shape[0] - 1)

        # update lines

        self.lines[0].set_data(t, states[:-1, 0])
        self.lines[1].set_data(t, states[:-1, 1])

        self.lines[2].set_data(t, actions[:, 1]*100)

        self.lines[3].set_data(t, states[:-1, 5])

        self.lines[4].set_data(t, states[:-1, 7])
        self.lines[5].set_data(t, P_q)
        self.lines[6].set_data(t, states[1:, 6])

        self.lines[7].set_data(t, states[:-1, 3])

        self.lines[8].set_data(t, actions[:, 0]*100)

        self.lines[9].set_data(t, states[:-1, -1])    
        #self.lines[10].set_data(t, rewards)

        self.lines[10].set_data(t, rewards[0:len(t)])


        # SoE = states[:, 5]
        # P_Q_p = B_p * (actions[:,0] * 1e-3) * h_p_const / 1e3
        # self.axs[0, 0].step(t, states[:-1, 0], where='post', label='V_ref')
        # self.axs[0, 0].step(t, states[:-1, 1], where='post', label='V_Irr')
        # self.axs[0, 0].set_title('Water demand fulfilled', weight = 'bold')
        # # self.axs[0].set_xlabel('Time (h)')
        # self.axs[0, 0].set_ylabel('Water volume (m3)')
        # actual_irrigation = np.diff(states[:, 1])/600*1e5
        # self.axs[0, 1].step(t, actions[:, 1]*100, where='post', label='Irr')
        # self.axs[0, 1].step(t, actual_irrigation, where='post', label='Actual_Irr')
        # self.axs[0, 1].set_title('Irrigation level', weight = 'bold')
        # self.axs[0, 1].set_ylabel('Irrigation level (%)')
        # self.axs[0, 1].legend()

        # self.axs[1, 0].step(t, SoE[:-1], where='post', label='Soe')
        # self.axs[1, 0].set_title('SoE batteries')
        # # self.axs[2].set_xlabel('Time (h)')
        # self.axs[1, 0].set_ylabel('SoE (kWh)')

        # self.axs[1, 1].step(t, states[:-1, 7], where='post', label='P_d')
        # self.axs[1, 1].step(t, states[:-1, 6], where='post', label='P_sun')
        # self.axs[1, 1].step(t, P_Q_p, where='post', label='P_pump')
        # self.axs[1, 1].set_title('Community demand', weight = 'bold')
        # # self.axs[3].set_xlabel('Time (h)')
        # self.axs[1, 1].set_ylabel('Power (kW)')

        # self.axs[2, 0].step(t, states[:-1, 3], where='post', label='V_tank')
        # self.axs[2, 0].set_title('Tank volume', weight = 'bold')
        # self.axs[2, 0].set_xlabel('Time (h)')
        # self.axs[2, 0].set_ylabel('Volume (m3)')

        # self.axs[2, 1].step(t, actions[:, 0]*100, where='post', label='Q_pump')
        # #self.axs[2, 1].step(t, Qp + actual_irrigation, where='post', label='Actual_Q_pump')
        # self.axs[2, 1].set_title('Pump', weight = 'bold')
        # self.axs[2, 1].set_xlabel('Time (h)')
        # self.axs[2, 1].set_ylabel('(%)')

        # self.axs[3, 0].step(t, states[1:,-1], where='post', label='E_residual')
        # self.axs[3, 0].set_title('Power balance', weight = 'bold')

        # self.axs[3, 1].step(t, rewards, where='post', label='rewards')
        # self.axs[3, 1].set_title('Transition Rewards', weight = 'bold')


    def get_figure(self):
        return self.fig

        
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
    
def default_rwd_fun(s, a, s_next):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state
    :param e_penal: penalty for energy deficit"""
    
    next_error = s[0] - s_next[1]
    current_error = s[0] - s[1]
    reward = -2 + 2*np.exp(-0.05*next_error**2)
    #if s_next[-3] != 0:
        #if s[3] >= Vt_max:
            #reward = -1.0 if delta_error > 0 else 0.0

    
    
    e_balance = s_next[-1]

    reward += e_balance*2 if e_balance < 0 else 0.0 
    """
    if a[0] < 0 or a[1] < 0:
        reward = -5.0

    # penalizations 
    reward += -4.0 if s_next[3] >= Vt_max and a[0] > 0 else 0.0  # penalize unfeasible action (pump is on and tank is full)
    reward += -4.0 if s_next[3] <= Vt_min and a[1] > 0 else 0.0  # penalize unfeasible action (irrigation is on and tank is empty)"""

    return np.array([reward], dtype=np.float32)
    
