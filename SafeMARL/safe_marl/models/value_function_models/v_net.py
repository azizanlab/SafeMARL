import torch
import torch.nn as nn
from safe_marl.models.base.mlp import MLPBase
from safe_marl.utils.envs_tools import check, get_shape_from_obs_space
from safe_marl.utils.models_tools import init, get_init_method


class VNet(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(args["initialization_method"])
        self.base = MLPBase(args, get_shape_from_obs_space(cent_obs_space))
        self.v_out = init(nn.Linear(self.hidden_sizes[-1], 1), init_method, lambda x: nn.init.constant_(x, 0))
        self.to(device)

    def forward(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        return self.v_out(self.base(cent_obs))
