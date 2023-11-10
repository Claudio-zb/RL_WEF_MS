import numpy as np


# Controller classes

class RL_du_controller:
    def __init__(self, policy):
        self.policy = policy
        self.y = None
        self.y_prev = None
        self.int_error = 0
        self.prev_error = 0
        self.delta_error = 0
        self.prev_action = 0

    def __call__(self, y, ref):
        if self.y is None:
            self.y = y
            self.y_prev = y
            error = ref - y
            self.int_error = error
            self.delta_error = error

        error = ref - y
        self.delta_error = error - self.prev_error
        self.int_error += error
        state = np.array(
            [y, ref, self.y_prev ** 2,
             self.prev_action])  # np.array([y, ref, self.int_error, self.delta_error])
        self.prev_error = error
        du = self.policy(state).cpu().detach().numpy()
        u = du[0] + self.prev_action
        self.prev_action = u = du[0] + self.prev_action

        return u


class RL_controller:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, y, ref):
        state = np.array([y, ref, y ** 2])
        u = self.policy(state).cpu().detach().numpy()
        return u


class RL_PID_controller:
    def __init__(self, policy):
        self.policy = policy
        self.y = None
        self.y_prev = None
        self.int_error = 0
        self.prev_error = 0
        self.delta_error = 0
        self.prev_action = 0

    def __call__(self, y, ref):
        if self.y is None:
            self.y = y
            self.y_prev = y
            error = ref - y
            self.int_error = error
            self.delta_error = error

        error = ref - y
        self.delta_error = error - self.prev_error
        self.int_error += error
        state = np.array(
            [y, ref, self.int_error, self.delta_error])  # np.array([y, ref, self.int_error, self.delta_error])
        self.prev_error = error
        u = self.policy(state).cpu().detach().numpy()
        return u


class LQR_id:
    def __init__(self, K):
        self.K = K
        self.x_prev = None
        self.int_error = 0
        self.delta_error = 0

    def __call__(self, x, x_ref):
        if self.x_prev is None:
            self.x_prev = x
        error = x_ref - x
        self.int_error += error
        u = -self.K @ np.array([x, self.x_prev, self.int_error, self.delta_error])
        self.x_prev = x
        return u


class LQR_i:
    def __init__(self, K):
        self.K = K
        self.int_error = 0

    def __call__(self, x, x_ref):
        self.int_error += x_ref - x
        u = -self.K @ np.array([x, self.int_error])
        return u


class LQR:
    def __init__(self, K):
        self.K = K

    def __call__(self, x, x_ref):
        u = -self.K @ (x - x_ref)
        return u


def next_step(x, u):
    x_next = np.zeros(2)
    x_next[0] = 1.1 * x[0] - 0.1 * x[1] ** 2 + 0.8 * u

    if x_next[0] > 1e4:
        x_next[0] = 1e4
    elif x_next[0] < -1e4:
        x_next[0] = -1e4

    x_next[1] = x[0]

    return x_next


def sample_trajectory(controller, type: str = 'baseline'):
    n_steps = 50
    operation_point = np.random.uniform(-6, 6)
    x_0 = np.array([operation_point, operation_point])
    x_ref = np.random.uniform(-6, 6)
    x = x_0
    if type == 'baseline':
        rl_controller = RL_controller(controller)
    elif type == 'pid':
        rl_controller = RL_PID_controller(controller)
    elif type == 'baseline_du':
        rl_controller = RL_du_controller(controller)
    responses = np.zeros(n_steps + 1)
    responses[0] = x[0]
    actions = np.zeros(n_steps)
    for i in range(n_steps):
        u = rl_controller(x[0], x_ref)
        x = next_step(x, u)
        responses[i + 1] = x[0]
        actions[i] = u

    return responses, actions, x_ref


def sin_reference(controller, type: str = 'baseline', operation_point=None):
    n_steps = 200
    if operation_point is None:
        operation_point = np.random.uniform(-6, 6)
    responses = np.zeros(n_steps + 1)

    actions = np.zeros(n_steps)
    x_0 = np.array([operation_point, operation_point])
    x = x_0
    responses[0] = x[0]
    t = np.linspace(0, 3 * np.pi, n_steps)
    ref = 2 * np.sin(t) + 3

    if type == 'baseline':
        rl_controller = RL_controller(controller)
    elif type == 'pid':
        rl_controller = RL_PID_controller(controller)
    elif type == 'baseline_du':
        rl_controller = RL_du_controller(controller)
    else:
        rl_controller = RL_du_controller(controller)

    for i in range(n_steps):
        x_ref = ref[i]
        u = rl_controller(x[0], x_ref)
        x = next_step(x, u)
        responses[i + 1] = x[0]
        actions[i] = u

    return responses, actions, ref
