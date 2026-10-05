import torch
from safe_marl.models.policy_models.stochastic_policy import StochasticPolicy


class OnPolicyBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = args["lr"]
        self.actor = StochasticPolicy(args, obs_space, act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=args["opti_eps"], weight_decay=args["weight_decay"]
        )

    def get_actions(self, obs, deterministic=False):
        return self.actor(obs, deterministic)

    def evaluate_actions(self, obs, action, active_masks=None):
        return self.actor.evaluate_actions(obs, action, active_masks)

    def act(self, obs, deterministic=False):
        actions, _ = self.actor(obs, deterministic)
        return actions

    def update(self, sample):
        raise NotImplementedError

    def train(self, actor_buffer, advantages):
        raise NotImplementedError

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()
