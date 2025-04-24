import numpy as np
from environments.Data.EMS.EMS_constants import *

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
    # self.lines[10].set_data(t, rewards)

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

        # lines for the drawdown
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