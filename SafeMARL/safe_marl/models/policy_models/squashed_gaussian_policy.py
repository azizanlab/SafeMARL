import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from safe_marl.models.base.plain_mlp import PlainMLP
from safe_marl.utils.envs_tools import get_shape_from_obs_space

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = list(args["hidden_sizes"])
        obs_dim = get_shape_from_obs_space(obs_space)[0]
        act_dim = action_space.shape[0]
        self.net = PlainMLP([obs_dim] + hidden_sizes, args["activation_func"], args["final_activation_func"])
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = float(action_space.high[0])
        self.to(device)

    def forward(self, obs, stochastic=True, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std, validate_args=False)
        pi_action = pi_distribution.rsample() if stochastic else mu

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1, keepdim=True)
        else:
            logp_pi = None

        return self.act_limit * torch.tanh(pi_action), logp_pi
