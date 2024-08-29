from torch import nn, distributions
import torch
import numpy as np


def get_action(policy, state, cov_mat):
    mean = policy(state)
    dist = distributions.Normal(mean, cov_mat)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.detach().cpu().numpy(), log_prob.detach()
