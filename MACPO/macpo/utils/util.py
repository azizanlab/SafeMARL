import math

import numpy as np
import torch


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


def get_shape_from_obs_space(obs_space):
    assert obs_space.__class__.__name__ == "Box"
    return obs_space.shape


def get_shape_from_act_space(act_space):
    assert act_space.__class__.__name__ == "Box"
    return act_space.shape[0]
