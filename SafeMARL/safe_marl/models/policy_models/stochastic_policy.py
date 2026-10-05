import torch
import torch.nn as nn
from safe_marl.models.base.act import ACTLayer
from safe_marl.models.base.mlp import MLPBase
from safe_marl.utils.envs_tools import check, get_shape_from_obs_space


class StochasticPolicy(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(StochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPBase(args, get_shape_from_obs_space(obs_space))
        self.act = ACTLayer(action_space, self.hidden_sizes[-1], self.initialization_method, args["gain"], args)
        self.to(device)

    def forward(self, obs, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        return self.act(self.base(obs), deterministic)

    def evaluate_actions(self, obs, action, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        return self.act.evaluate_actions(self.base(obs), action, active_masks=active_masks)
