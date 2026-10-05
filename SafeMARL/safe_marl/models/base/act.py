import torch.nn as nn
from safe_marl.models.base.distributions import DiagGaussian


class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, initialization_method, gain, args=None):
        super(ACTLayer, self).__init__()
        assert action_space.__class__.__name__ == "Box", "only continuous action spaces are supported"
        self.action_out = DiagGaussian(inputs_dim, action_space.shape[0], initialization_method, gain, args)

    def forward(self, x, deterministic=False):
        action_distribution = self.action_out(x)
        actions = action_distribution.mode() if deterministic else action_distribution.sample()
        return actions, action_distribution.log_probs(actions)

    def evaluate_actions(self, x, action, active_masks=None):
        action_distribution = self.action_out(x)
        action_log_probs = action_distribution.log_probs(action)
        entropy = action_distribution.entropy().unsqueeze(-1)
        if active_masks is not None:
            dist_entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy, action_distribution
