from typing import Callable, List, Iterable
import gymnasium as gym

import copy
from numpy import ndarray
from scipy.special import exp1
import torch
from utils_functions.funcionesEMS import *
from gymnasium import spaces
from matplotlib import figure


class ContinousEMSEnv(gym.Env):
    """
    Environment for the Energy Management System
    """

    def __init__(self, rwd_function=None, render: bool = True):
        """
        Initialize the environment
        :param render:
        """
        self.render = render
        self._init_figure()

        # Hyper params
        self.max_steps: int = 288
        self.day_steps: int = 144
        self.start_index: int = 0
        self.k: int = 0
        self.time: int = 0

        # Load meteorological data and demand

        self.radiation_data: np.ndarray = get_rad('ver')
        self.temperature_data: np.ndarray = get_temperatura('ver')
        self.demand_data: np.ndarray = get_demand()
        self.L = len(self.temperature_data)  # length(temperatura)
        self.N_dias: int = 70
        self.transform = T_matrix
        self.transform_dim = self.transform.shape[1]

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
        self.dQ = []
        self.drawdown = 0.0

        self.reward_fun = continous_rwd_fun

        # Bounds for observations
        obs_low = np.array([0.0,
                            Vt_min, I_min,
                            SoE_min,
                            0.0,
                            0.0, 0,
                            0.0, 0.0,
                            -1, -1], dtype=np.float32)

        obs_high = np.array([Vt_max,
                             Vt_max, I_max,
                             SoE_max,
                             Vt_max,
                             Q_p_max, 4,
                             1000, 1000,
                             1, 1], dtype=np.float32)

        # Bounds for actions
        self.action_low = np.array([0.0, 0.0],
                                   dtype=np.float32)

        self.action_high = np.array([Q_p_max, I_max],
                                    dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high,
                                            shape=(11,),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=self.action_low,
                                       high=self.action_high,
                                       shape=(2,),
                                       dtype=np.float32)

    def step(self, action: np.ndarray, mode: str = "train") -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """
        Execute one step of the environment, given an action.
        :param action: Action to be executed
        :param mode: mode of the environment. Can be "train" or "eval"
        :return: tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Store the previous values of the variables to compute the reward
        observation = self._get_obs()

        action = self.map_action(action)
        action = action.flatten()

        action_ = self._low_level_control(action)

        # self.dQ[self.k + 144] = (action_[0] - self.Q_p) * 1e-3
        self.dQ.append((action_[0] - self.Q_p) * 1e-3)

        self.Q_p = action_[0]
        self.Irr = action_[1]

        next_state, P_pump, self.E_residual = EMS_ode(self._get_state(),
                                                      [self.V_ref, self.p_fv[self.k % 144], self.demanda[self.k % 144]],
                                                      action_)

        truncated = False
        terminated = False

        Info = {}

        self.k = self.k + 1

        self.V_Irr = next_state[0]
        self.Vt = next_state[1]
        self.SoE = next_state[2]
        self.drawdown = drawdown(self.k + 144, self.dQ[0:self.k + 144])
        # self.drawdown = drawdown(self.k + 144, self.dQ)

        if (self.k % 144) == 0:  # The time at s' is 00:00 i.e. the final day is over
            self.V_Irr = 0.0
            self.V_ref = 4.0 * np.random.rand()

            if mode == "eval":
                self.day_picked = (self.day_picked + 1) % 70
            else:
                self.day_picked = np.random.randint(0, 70)

            self.p_fv, self.demanda = self._pick_metereological_data(self.day_picked)

        if self.k % 288 == 0:
            self.V_Irr = 0.0
            self.V_ref = 4.0 * np.random.rand()
            terminated = True

        observation_next = self._get_obs()

        reward = self.reward_fun(observation, action, observation_next)

        return observation_next, reward, terminated, truncated, Info

    def _low_level_control(self, action: np.ndarray) -> np.ndarray:
        """Low level control for the pump and irrigation"""
        action_ = np.copy(action)
        if self.Vt <= Vt_min:
            action_[1] = 0.0
        if self.Vt >= Vt_max:
            action_[0] = 0.0
        if self.V_ref <= self.V_Irr:
            action_[1] = 0.0
        if self.drawdown > 1:
            action_[0] = 0.0
        return action_

    def map_action(self, policy_output: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if torch.is_tensor(policy_output):
            return policy_output.detach().cpu().numpy()
        else:
            return policy_output

    def reset(self, seed=None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state
        :param seed: random seed
        :param options: options for the environment
        :return: tuple of (initial_observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        if options is not None:
            if options["mode"] == "eval":
                np.random.seed(0)
                self.day_picked = 0
        else:
            self.day_picked = np.random.randint(0, 70)

        V_tank = np.minimum((Vt_max - Vt_min) * np.random.random_sample() + Vt_min,
                            (Vt_max - Vt_min) * np.random.random_sample() + Vt_min)
        V_ref = 3.0 * np.random.rand() + 1.0
        SoE = (SoE_max - SoE_min) * np.random.random_sample() + SoE_min

        InitialObservation = self.set_initial_conditions(self.day_picked,
                                                         V_ref=V_ref,
                                                         V_tank=V_tank,
                                                         Soe=SoE,
                                                         Irr_prev=0.0,
                                                         instant_k=0,
                                                         V_irr=0.0)
        info = {}

        return InitialObservation, info

    def _pick_metereological_data(self, day_picked: int):
        """returns p_fv, demanda for a given day"""

        n_steps = self.day_steps  # self.k + self.day_steps
        start_index = day_picked * self.day_steps  # self.max_steps
        radiacion = self.radiation_data[start_index:start_index + n_steps + 1] + 1e-4 * np.random.randn(n_steps + 1)
        temperatura = self.temperature_data[start_index:start_index + n_steps + 1] + 1e-2 * np.random.randn(n_steps + 1)
        p_fv = solar_power(radiacion, temperatura) + 1e-4 * np.random.randn(n_steps + 1)
        demanda = self.demand_data[start_index:start_index + n_steps + 1]

        return p_fv, demanda

    def set_initial_conditions(self, day_picked, V_ref, V_tank, Soe, Irr_prev, instant_k, V_irr):
        """
        Set the initial conditions of the environment
        :param V_irr:
        :param instant_k:
        :param day_picked:
        :param V_tank:
        :param Soe:
        :param Irr_prev:
        :return: Initial observation
        """
        self.k = instant_k

        self.Vt = V_tank
        self.SoE = Soe
        self.Irr = Irr_prev
        self.V_Irr = V_irr
        self.V_ref = V_ref
        self.p_fv, self.demanda = self._pick_metereological_data(day_picked)
        self.Pbat, _, self.E_residual = manage_batteries(self.SoE, self.p_fv[self.k], self.demanda[self.k], 0)
        self.dQ, self.Q_p = self._create_dQ(V_req=V_ref)
        self.drawdown = drawdown(self.k + 144, self.dQ[0:144])
        InitialObservation = np.array([self.V_ref,
                                       self.V_Irr,
                                       self.Irr,
                                       self.Vt,
                                       self.Q_p,
                                       self.drawdown,
                                       self.SoE,
                                       self.p_fv[self.k],
                                       self.demanda[self.k],
                                       0.0,
                                       self.E_residual])
        return InitialObservation

    def _create_dQ(self, V_req) -> Tuple[list, float]:
        "Returns the dQ sequence from a previous day and the last value for the pump action Q_p"
        L = self.day_steps + self.max_steps
        prev_Q = np.zeros(self.day_steps)
        # d_Q = np.zeros(L)  # []
        sum = 0
        K = Q_p_max * dt / 1000
        for i in range(L):
            if sum < V_req:
                x = np.random.rand() * K
                if sum + x < V_req:
                    prev_Q[i] = x
                    sum += x
                else:
                    prev_Q[i] = V_req - sum
                    break
            else:
                break
        prev_Q = prev_Q / K
        np.random.shuffle(prev_Q)
        last_Q = prev_Q[-1]
        prev_d_Q = np.diff(prev_Q, prepend=0) / 1000
        # d_Q[0:self.day_steps] = prev_d_Q
        d_Q = [dq for dq in prev_d_Q]
        return d_Q, last_Q

    def sample_trajectory(self,
                          policy: Callable,
                          max_steps: int = 288,
                          rew_fun=None):

        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        x0, _ = self.reset()
        states[0] = x0

        actions = np.zeros((max_steps, self.action_low.shape[0]))

        for i in range(max_steps):
            action = policy(states[i])
            if type(action) == torch.Tensor:
                action = action.squeeze().detach().cpu().numpy()
            x_next, _, terminated, truncated, _ = self.step(action)
            actions[i] = action
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
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
        states, actions, rewards = self.sample_trajectory(policy, max_steps=288, rew_fun=self.reward_fun)
        qp = actions[:, 0]
        P_q = np.array([P_Q_p(q, h_p_const) for q in qp])

        t = np.linspace(0, 48, states.shape[0] - 1)

        # update lines

        self.lines[0].set_data(t, states[:-1, 0])
        self.lines[1].set_data(t, states[:-1, 1])

        self.lines[2].set_data(t, actions[:, 1] * 100)
        self.lines[3].set_data(t, actions[:, 0] * 100)

        self.lines[4].set_data(t, states[:-1, 6])

        self.lines[5].set_data(t, states[:-1, 8])
        self.lines[6].set_data(t, P_q)
        self.lines[7].set_data(t, states[1:, 7])

        self.lines[8].set_data(t, states[:-1, 3])

        self.lines[9].set_data(t, states[:-1, 5])

        self.lines[10].set_data(t, states[:-1, -1])
        #self.lines[10].set_data(t, rewards)

        self.lines[11].set_data(t, rewards[0:len(t)])

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def compare_policies(self,
                         policies: Iterable[
                             Tuple[Callable[[Union[np.ndarray, torch.Tensor]], Union[ndarray, torch.Tensor]]]],
                         max_steps: int = 144,
                         rew_funs: Iterable[
                             Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]] = None,
                         options=None) -> list:
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
            self.axs = [self.fig.add_subplot(4, 2, i + 1) for i in range(4 * 2)]
            self.fig.suptitle('Energy Management System')
            self.fig.tight_layout()
            self.fig.set_size_inches(10, 10)
            self.lines = []

            # lines for the reference traking
            self.lines.append(self.axs[0].plot([], [], label="V_req")[0])
            self.lines.append(self.axs[0].plot([], [], label="V_Irr")[0])
            self.axs[0].set_title("Requerimiento hídrico y volumen irrigado")
            self.axs[0].set_xlabel("Tiempo [h]")
            self.axs[0].set_ylabel("Volumen [m3]")
            self.axs[0].legend()

            # lines for the irrigation and pump
            self.lines.append(self.axs[1].plot([], [], label="Irr")[0])
            self.lines.append(self.axs[1].plot([], [], label="Q_pump")[0])
            self.axs[1].set_title("Irrigación y bombeo")
            self.axs[1].set_xlabel("Tiempo [h]")
            self.axs[1].set_ylabel("Flujo [l/s]")
            self.axs[1].legend()

            # lines for soe
            self.lines.append(self.axs[2].plot([], [], label="SoE")[0])
            self.axs[2].set_title("State of Energy")
            self.axs[2].set_xlabel("Tiempo [h]")
            self.axs[2].set_ylabel("SoE [kWh]")
            self.axs[2].legend()

            # lines for the power consumed by the community
            self.lines.append(self.axs[3].plot([], [], label="P_d")[0])
            self.lines.append(self.axs[3].plot([], [], label="P_pump")[0])
            self.lines.append(self.axs[3].plot([], [], label="P_sun")[0])
            self.axs[3].set_title("Consumo doméstico, bombeo y solar")
            self.axs[3].set_xlabel("Tiempo [h]")
            self.axs[3].set_ylabel("Potencia [kW]")
            self.axs[3].legend()

            # lines for the tank volume
            self.lines.append(self.axs[4].plot([], [], label="V_tank")[0])
            self.axs[4].set_title("Volumen del tanque")
            self.axs[4].set_xlabel("Tiempo [h]")
            self.axs[4].set_ylabel("Volumen [m3]")

            #lines for the drawdown
            self.lines.append(self.axs[5].plot([], [], label="drawdown")[0])
            self.axs[5].set_title("Descenso del pozo")
            self.axs[5].set_xlabel("Tiempo [h]")
            self.axs[5].set_ylabel("Descenso [m]")

            # lines for the power balance
            self.lines.append(self.axs[6].plot([], [], label="E_residual")[0])
            self.axs[6].set_title("Balance de energía")
            self.axs[6].set_xlabel("Tiempo [h]")

            # lines for the rewards
            self.lines.append(self.axs[7].plot([], [], label="Rewards")[0])
            self.axs[7].set_title("Recompensas")
            self.axs[7].set_xlabel("Tiempo [h]")
            self.axs[7].set_ylabel("Rewards")

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
                                self.drawdown,
                                self.SoE,
                                self.p_fv[self.k % 144],
                                self.demanda[self.k % 144],
                                (self.k % 144),
                                self.E_residual])
        return observation


class normalizationWrapper(gym.Wrapper):
    def __init__(self, env: ContinousEMSEnv):
        super().__init__(env)
        self.env = env
        self.transform = T_matrix
        self.prev_action: np.ndarray = np.zeros(self.env.action_space.shape[-1])
        low = np.matmul(self.transform, env.observation_space.low)
        high = np.matmul(self.transform, env.observation_space.high)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_high,
                                       high=self.action_high,
                                       shape=(2,),
                                       dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.prev_action = np.zeros_like(self.env.action_space.shape[-1])
        state, info = self.env.reset()
        t_state = np.matmul(self.transform, state)
        info = state
        return t_state, state

    def step(self, action):
        action_ = self.prev_action + action
        action_ = np.clip(action_, self.env.action_low, self.env.action_high)
        state, reward, terminated, truncated, _ = self.env.step(action_)
        t_state = np.matmul(self.transform, state)
        info = state
        return t_state, reward, terminated, truncated, info


class DiscreteEMSEnv(ContinousEMSEnv):
    """Discrete action space implementation of the EMS environment"""

    def __init__(self, rwd_function=None, render: bool = True):
        super().__init__(rwd_function, render)
        self.action_space = spaces.Discrete(16)
        self.Irr_levels = np.array([0.0, .05, .1, I_max])
        self.Q_p_levels = np.array([0.0, .3333, .6666, Q_p_max])
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
                          rew_fun=None):
        states = np.zeros((max_steps + 1, self.observation_space.shape[0]))
        x0, _ = self.reset()
        states[0] = x0
        a_shape = self.action_values.shape
        if len(a_shape) > 1:
            actions = np.zeros((max_steps, self.action_values.shape[1]))
        else:
            actions = np.zeros((max_steps, 1))

        for i in range(max_steps):

            action = policy(states[i])
            action = action.max(0).indices.view(1, 1)  # the index
            actions[i] = self.action_values[action]

            x_next, _, terminated, truncated, _ = self.step(action)
            states[i + 1] = x_next
            if terminated or truncated:
                states = states[:i + 2]
                actions = actions[:i + 1]
                break
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
    norm_next_error = (s[0] - s_next[1]) / s[0]  # Normalize the error to be a fraction of the daily demand

    reward = - norm_next_error if norm_next_error > 0 else norm_next_error

    reward += -a[1] if s[3] <= Vt_min and a[
        1] > 0 else 0.0  #penalize unfeasible action (irrigation is on and tank is empty)

    reward += -a[0] if s[3] >= Vt_max and a[0] > 0 else 0.0  #penalize unfeasible action (pump is on and tank is full)

    e_balance = s_next[-1]

    reward += e_balance if e_balance < 0 else 0

    if a[0] < 0.0 or 1.0 < a[0]:
        reward -= abs(a[0]) * 2

    if a[1] < 0.0 or 1.0 < a[1]:
        reward -= abs(a[1]) * 2

    return np.array([reward], dtype=np.float32)


def continous_rwd_fun(s, a, s_next):
    """ Default reward function 
    :param s: current state
    :param a: action
    :param s_next: next state"""
    reward = 0.0
    p_reward = 0.0
    K = s[-2] / 143
    if s_next[-2] != 0:
        norm_next_error = (s[0] - s_next[1]) / s[0]
        if s_next[0] <= s_next[1] and a[1] > 0:
            p_reward = -a[1]
    else:
        norm_next_error = (s[0] - s[1]) / s[0]

    reward = (1 - norm_next_error) ** 2 if norm_next_error > 0 else (1 + norm_next_error) ** 2  # ta gucci
    if norm_next_error < -1:
        reward = -1.0
    reward = K * np.maximum(reward, -1.0)  # ta gucci
    reward = reward  #+ p_reward

    reward += -10 * a[1] if s[3] <= Vt_min and a[
        1] > 0 else 0.0  # penalize unfeasible action (irrigation is on and tank is empty)

    reward += -10 * a[0] if s[3] >= Vt_max and a[
        0] > 0 else 0.0  # penalize unfeasible action (pump is on and tank is full)

    reward += -10 * a[0] if s[5] > 1 else 0.0  # penalize drawdown

    e_balance = s_next[-1]

    reward += e_balance if e_balance < 0 else 0.0

    if a[0] < 0.0 or 1.0 < a[0]:
        reward -= 10 * abs(a[0])

    if a[1] < 0.0 or 1.0 < a[1]:
        reward -= 10 * abs(a[1])

    return reward


def EMS_ode(x, d, u) -> Tuple[np.ndarray, float, float]:
    """The ODE of the micorgrid system
    :param x: state vector V_irr, V_tank, SoE
    :param d: disturbance vector V_ref, P_sun, P_demand
    :param u: control vector Q_pump, Irrigation, P_bat
    :return: next state, battery power, energy residual"""

    V_irr = x[0]  # Irrigatated volume [m3]
    V_tank = x[1]  # Tank volume [m3]
    SoE = x[2]  # State of Energy [kWh]
    # s = x[3] # Descenso del pozo

    V_ref = d[0]  # Reference volume [m3]
    P_sun = d[1]  # Solar power [kW]
    P_demand = d[2]  # Demand power [kW]

    u = np.clip(u, [0.0, 0.0], [1.0, 1.0])  # Clip the action to the feasible range

    Q_pump = u[0]  # Pump flow rate [l/s]
    Irrigation = u[1]  # Irrigation flow rate [l/s]
    # P_bat = u[2] # Battery power [kW]

    if V_tank <= Vt_min:  # If the tank is empty, there is no irrigation
        if Irrigation > 0:
            Irrigation = 0.0

    amount_to_irrigate = dt * (Irrigation * 1e-3)

    # Vt_to_fill = Vt_max - V_tank - amount_to_irrigate  # Amount of water that can be filled

    # if Vt_to_fill <= 0 < self.Q_p:  # If the tank is full and the pump is feeding, the pump is turned off
    #     self.Q_p = 0

    amount_to_pump = dt * (Q_pump * 1e-3)  # Volume [m3]

    V_tank_next = np.clip(V_tank + amount_to_pump - amount_to_irrigate, Vt_min, Vt_max)
    V_Irr_next = V_irr + amount_to_irrigate

    P_Q_p_ = P_Q_p(Q_pump, h_p_const)  # water pump power [kW]

    P_bat, SoE_next, E_residual = manage_batteries(SoE,
                                                   P_sun,
                                                   P_demand,
                                                   P_Q_p_)
    x_next = np.array([V_Irr_next,
                       V_tank_next,
                       SoE_next])

    return x_next, P_bat, E_residual


def drawdown(k: int, dQ: Union[np.ndarray, List]):  # drawdown of the well
    assert k == len(dQ), "The length of dQ should match the number of temporal k points"
    if type(dQ) == list:
        dQ = np.array(dQ)
    l = np.arange(1, k + 1)
    arg = (r_pozos ** 2 * S) / (4 * T * (k - l + 1) * 600)
    sum_ = np.dot(dQ, exp1(arg))
    s_val = 1 / (4 * np.pi * T) * sum_
    return s_val


def P_Q_p(q_p: float, h_p: float):
    """Water pump power [kW]
    :param q_p: flow rate [l/s]
    :param h_p: height [m]
    :return: power [kW]"""
    P_Q_p_ = B_p * (q_p * 1e-3) * h_p / 1e3
    return P_Q_p_


class EMS_Wrapper(gym.Wrapper):
    """This class normalize the observations of the EMS environment"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.transform = T_matrix

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info
