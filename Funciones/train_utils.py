import numpy as np
from torch import distributions
from EMS_env import EMS_env


def get_action(policy, state, cov_mat):
    """
    Sample an action from the policy considering exploration noise
    :param policy: policy to be used
    :param state: current state
    :param cov_mat: covariance matrix of the exploration noise
    :return: action and log probability of the action
    """
    mean = policy(state)
    dist = distributions.MultivariateNormal(mean, cov_mat)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.detach().cpu().numpy(), log_prob.detach()


def sample_trajectory(env: EMS_env, policy, max_steps: int = 288):
    """
    Sample a trajectory from the environment using the policy
    :param env: Energy Management System Environment
    :param policy: Policy to be used
    :param max_steps: Maximum number of steps to be taken
    :return: states and actions of the trajectory
    """
    states = np.zeros((max_steps + 1, env.observation_space.shape[0]))
    actions = np.zeros((max_steps, env.action_space.shape[0]))
    x0, _ = env.reset()
    states[0] = x0
    for i in range(max_steps-1):
        action = policy(states[i])
        actions[i] = action.detach().cpu().numpy()
        x_next, _, _, _, _ = env.step(actions[i])
        states[i + 1] = x_next
    return states, actions
