import torch.nn as nn
from .distributions import DiagGaussian


class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        assert action_space.__class__.__name__ == "Box"
        self.action_out = DiagGaussian(inputs_dim, action_space.shape[0], use_orthogonal, gain, args)

    def forward(self, x, deterministic=False):
        action_logits = self.action_out(x)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, active_masks=None):
        action_logits = self.action_out(x)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy() * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy
