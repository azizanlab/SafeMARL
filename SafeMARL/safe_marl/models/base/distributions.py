import torch
import torch.nn as nn
from safe_marl.utils.models_tools import init, get_init_method


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01, args=None):
        super(DiagGaussian, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.std_x_coef = args["std_x_coef"] if args is not None else 1.0
        self.std_y_coef = args["std_y_coef"] if args is not None else 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = torch.nn.Parameter(torch.ones(num_outputs) * self.std_x_coef)

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
