import numpy as np
from torch import distributions
from typing import Tuple


def get_action(policy, state, cov_mat) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample an action from the policy considering exploration noise
    :param policy: policy to be used
    :param state: current state
    :param cov_mat: covariance matrix of the exploration noise
    :return: action and log probability of the action
    """
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    mean = policy(state)
    mean = mean.reshape(-1)
    dist = distributions.MultivariateNormal(mean, cov_mat)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.detach().cpu().numpy(), log_prob.detach()

