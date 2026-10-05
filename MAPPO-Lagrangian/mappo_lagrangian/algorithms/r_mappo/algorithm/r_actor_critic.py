import torch
import torch.nn as nn
from mappo_lagrangian.algorithms.utils.util import init, check
from mappo_lagrangian.algorithms.utils.mlp import MLPBase
from mappo_lagrangian.algorithms.utils.act import ACTLayer
from mappo_lagrangian.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        actor_features = self.base(obs)
        return self.act(actor_features, deterministic)

    def evaluate_actions(self, obs, action, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        actor_features = self.base(obs)
        return self.act.evaluate_actions(actor_features, action, active_masks=active_masks)


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = MLPBase(args, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        return self.v_out(self.base(cent_obs))
