import torch
import torch.nn as nn
from safe_marl.models.base.plain_mlp import PlainMLP
from safe_marl.utils.envs_tools import get_shape_from_obs_space


class DeterministicPolicy(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_dim = get_shape_from_obs_space(obs_space)[0]
        act_dim = action_space.shape[0]
        self.pi = PlainMLP(
            [obs_dim] + list(args["hidden_sizes"]) + [act_dim], args["activation_func"], args["final_activation_func"]
        )
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(**self.tpdv)
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
        self.to(device)

    def forward(self, obs):
        return self.scale * self.pi(obs) + self.mean
