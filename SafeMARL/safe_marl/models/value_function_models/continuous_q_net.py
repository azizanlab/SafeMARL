import torch
import torch.nn as nn
from safe_marl.models.base.plain_mlp import PlainMLP
from safe_marl.utils.envs_tools import get_shape_from_obs_space


class ContinuousQNet(nn.Module):
    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(ContinuousQNet, self).__init__()
        input_dim = get_shape_from_obs_space(cent_obs_space)[0] + sum(space.shape[0] for space in act_spaces)
        self.mlp = PlainMLP([input_dim] + list(args["hidden_sizes"]) + [1], args["activation_func"])
        self.to(device)

    def forward(self, cent_obs, actions):
        return self.mlp(torch.cat([cent_obs, actions], dim=-1))
