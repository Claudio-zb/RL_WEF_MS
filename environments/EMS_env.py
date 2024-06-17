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
from environments.custom_env import CustomEnv
from matplotlib import figure

class ContinousEMSEnv(CustomEnv):
    """
    Environment for the Energy Management System
    """

    def __init__(self, rwd_function = None, render: bool = True):
        """
        Initialize the environment
        :param render:
        """
        self.render = render
        self._init_figure()

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
        self.V_Irr = np.array([0.0])
        self.Q_p = np.array([0.0])
        self.Pbat = np.array([0.0])
        self.V_ref = np.array([0.0])
        self.Vt = np.array([0.0])
        self.SoE = np.array([0.0])
        self.E_residual = 0.0
        self.day_picked = 0

        if rwd_function is not None:
            self.reward_fun = rwd_function
        else:
            self.reward_fun = default_rwd_fun

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


        # Bounds for actions
        self.action_low = np.array([0.0, 0.0],
                                   dtype=np.float32)

        self.action_high = np.array([1.0, 1.0],
                                    dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(11,),
                                            dtype=np.float32)
        
        self.action_space = spaces.Box(low=self.action_low,
                                        high=self.action_high,
                                        shape=(2,),
                                        dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[ndarray, ndarray, bool, bool, dict[str, Any]]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Store the previous values of the variables to compute the reward
        observation = self._get_obs()

        action = self.map_action(action)
        action = action.flatten()

        next_state, P_pump, self.E_residual = EMS_ode(self._get_state(), 
                                                 [self.V_ref, self.p_fv[self.k], self.demanda[self.k]], 
                                                 action)
        
        truncated = False
        terminated = False

        Info = {}
        moment_of_the_day = self.k % 144
        self.accumulated_P_consumed += self.demanda[self.k]
        
        self.V_Irr = next_state[0]
        self.Vt = next_state[1]
        self.SoE = next_state[2]

        self.k = self.k + 1

        if (self.k % 143) == 0:  # The time at s' is 00:00 i.e. the final day is over
            terminated = True

        observation_next = self._get_obs()

        reward = self.reward_fun(observation, action, observation_next)

        return observation_next, reward, terminated, truncated, Info
    
    def map_action(self, policy_output: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if torch.is_tensor(policy_output):
            return policy_output.detach().cpu().numpy()
        else:
            return policy_output

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state
        :param seed: random seed
        :param options: options for the environment
        :return: tuple of (initial_observation, info)
        """
        self.day_picked = np.random.randint(0, 70)

        V_tank = np.minimum((Vt_max - Vt_min) * np.random.random_sample() + Vt_min, (Vt_max - Vt_min) * np.random.random_sample() + Vt_min)
        InitialObservation = self.set_initial_conditions(self.day_picked,
                                                         V_ref=3.5*np.random.rand(),
                                                         V_tank=V_tank,
                                                         Soe=(SoE_max - SoE_min) * np.random.random_sample() + SoE_min,
                                                         Irr_prev=np.random.rand(),
                                                         Q_p_prev=np.random.rand(),
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
        self.radiacion = self.radiation_data[self.start_index:self.start_index + n_steps + 1] + 1e-4*np.random.randn(n_steps + 1)
        self.temperatura = self.temperature_data[self.start_index:self.start_index + n_steps + 1] + 1e-2*np.random.randn(n_steps + 1)
        self.p_fv = solar_power(self.radiacion, self.temperatura) + 1e-4*np.random.randn(n_steps + 1)
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

    def sample_trajectory(self,
                          policy: Callable,
                          max_steps: int = 288,
                          rew_fun=None,
                          initial_conditions: dict = None,
                          options: dict = None):

        policy.eval()
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        
        if initial_conditions is not None:
            x0 = self.load_initial_conditions(initial_conditions)
        elif options is not None:
            x0, _ = self.reset(options={"t_init": 0})
        else:
            x0, _ = self.reset()
        states[0] = x0

        actions = np.zeros((max_steps, self.action_low.shape[0]))

        for i in range(max_steps):
            action = policy(states[i])
            action = action.squeeze().detach().cpu().numpy()
            exploration_noise = np.random.normal(0, 0.3, size=action.shape)
            action = np.clip(action + exploration_noise, self.action_low, self.action_high) 
            x_next, _, terminated, truncated, _ = self.step(action)
            actions[i] = action
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
        policy.train()
        rewards = np.zeros(len(actions))
        if rew_fun is not None:
            for i in range(len(actions)):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])

        return states, actions, rewards
    
    def show_sample(self, policy):
        """
        Render the environment
        :param policy: policy to be used
        :return:
        """

        states, actions, rewards = self.sample_trajectory(policy, max_steps=243, rew_fun=self.reward_fun)
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
    
    def get_figure(self):
        return self.fig
    
    def _init_figure(self):
        self.fig: figure.Figure = None
        self.axs = []
        self.lines = None
        if self.render:
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

    def _get_state(self) -> np.ndarray:
        """Returns the state of the environment"""
        state = np.array([self.V_Irr, self.Vt, self.SoE])
        return state

    def _get_obs(self) -> np.ndarray:
        """Return the observation of the environment"""
        observation = np.array([self.V_ref,
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
        return observation
class DiscreteEMSEnv(ContinousEMSEnv):
    """Discrete action space implementation of the EMS environment"""
    def __init__(self, rwd_function = None, render: bool = True):
        super().__init__(rwd_function, render)
        self.action_space = spaces.Discrete(16)
        self.Irr_levels = np.array([0.0, .05, .1, .2])
        self.Q_p_levels = np.array([0.0, .3333, .6666, 1.0])
        self.action_values = np.array(np.meshgrid(self.Q_p_levels, self.Irr_levels), dtype=np.float32).T.reshape(-1, 2)
    
    def map_action(self, action: torch.Tensor) -> np.ndarray:
        """
            Map the action from the policy to the action of the environment
            :param action:
            :return:
            """
        index = action.item()
        return np.array([self.action_values[index]])
    
    def sample_trajectory(self,
                          policy: Callable,
                          max_steps: int = 288,
                          rew_fun=None,
                          initial_conditions: dict = None,
                          options: dict = None):
        policy.eval()
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        
        if initial_conditions is not None:
            x0 = self.load_initial_conditions(initial_conditions)
        elif options is not None:
            x0, _ = self.reset(options={"t_init": 0})
        else:
            x0, _ = self.reset()
        states[0] = x0

        a_shape = self.action_values.shape
        if len(a_shape) > 1:
            actions = np.zeros((max_steps, self.action_values.shape[1]))
        else:
            actions = np.zeros((max_steps, 1))

        for i in range(max_steps):

            action = policy(states[i])
            action = action.max(1).indices.view(1, 1) # the index
            actions[i] = self.action_values[action]

            x_next, _, terminated, truncated, _ = self.step(action)
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
        policy.train()
        rewards = np.zeros(max_steps)
        if rew_fun is not None:
            for i in range(len(actions)):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])
        else:
            rew_fun = self.reward_fun
            for i in range(len(actions)):
                rewards[i] = rew_fun(states[i], actions[i], states[i + 1])
        return states, actions, rewards

def default_rwd_fun(s, a, s_next):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state
    :param e_penal: penalty for energy deficit"""
    reward = 0.0
    norm_next_error = (s[0] - s_next[1])/s[0] # Normalize the error to be a fraction of the daily demand
    
    #reward = 1 - norm_next_error if norm_next_error > 0 else 1 + 2*norm_next_error

    reward = 1 - norm_next_error**2 if norm_next_error > 0 else 1 - 2*norm_next_error**2

    reward = np.maximum(reward, -1.0)

    reward += -1.0 if s[3] <= Vt_min and a[1] > 0 else 0.0 #penalize unfeasible action (irrigation is on and tank is empty)

    reward += -1.0 if s[3] >= Vt_max and a[0] > 0 else 0.0 #penalize unfeasible action (pump is on and tank is full)
    
    e_balance = s_next[-1]

    reward += e_balance if e_balance < 0 else e_balance

    if a[0] < 0.0 or 1.0 < a[0]:
        reward -= abs(a[0])
    
    if a[1] < 0.0 or 1.0 < a[1]:
        reward -= abs(a[1])

    return np.array([reward], dtype=np.float32)


def EMS_ode(x,d,u) -> Tuple[np.ndarray, float, float]:
    """The ODE of the micorgrid system
    :param x: state vector V_irr, V_tank, SoE
    :param d: disturbance vector V_ref, P_sun, P_demand
    :param u: control vector Q_pump, Irrigation, P_bat
    :return: next state, battery power, energy residual"""

    V_irr = x[0] # Irrigatated volume [m3]
    V_tank = x[1] # Tank volume [m3]
    SoE = x[2] # State of Energy [kWh]

    V_ref = d[0] # Reference volume [m3]
    P_sun = d[1] # Solar power [kW]
    P_demand = d[2] # Demand power [kW]

    u = np.clip(u, [0.0, 0.0], [1.0, 1.0]) # Clip the action to the feasible range

    Q_pump = u[0] # Pump flow rate [l/s]
    Irrigation = u[1] # Irrigation flow rate [l/s]
    #P_bat = u[2] # Battery power [kW]

    if V_tank <= Vt_min:  # If the tank is empty, there is no irrigation
            if Irrigation > 0:
                Irrigation = 0.0

    amount_to_irrigate = dt * (Irrigation * 1e-3)

    # Vt_to_fill = Vt_max - V_tank - amount_to_irrigate  # Amount of water that can be filled
        
        # if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
        #     self.Q_p = 0

    amount_to_pump = dt * (Q_pump * 1e-3) # Volume [m3]

    V_tank_next = np.clip(V_tank + amount_to_pump - amount_to_irrigate, Vt_min, Vt_max)
    V_Irr_next = V_irr + amount_to_irrigate

    P_Q_p = B_p * (Q_pump * 1e-5) * h_p_const / 1e3  # water pump power [kW]

    P_bat, SoE_next, E_residual = manage_batteries(SoE,
                                                   P_sun,
                                                   P_demand,
                                                   P_Q_p)
    x_next = np.array([V_Irr_next,
                       V_tank_next,
                       SoE_next])
    
    return x_next, P_bat, E_residual