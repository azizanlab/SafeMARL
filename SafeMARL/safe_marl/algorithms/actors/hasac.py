import torch
from safe_marl.models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from safe_marl.utils.envs_tools import check


class HASAC:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert act_space.__class__.__name__ == "Box", "HASAC in this repo supports continuous actions only"
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.actor = SquashedGaussianPolicy(args, obs_space, act_space, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args["lr"])
        self.turn_off_grad()

    def get_actions(self, obs, stochastic=True):
        actions, _ = self.actor(check(obs).to(**self.tpdv), stochastic=stochastic, with_logprob=False)
        return actions

    def get_actions_with_logprobs(self, obs, stochastic=True):
        return self.actor(check(obs).to(**self.tpdv), stochastic=stochastic, with_logprob=True)

    def turn_on_grad(self):
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        for p in self.actor.parameters():
            p.requires_grad = False

    def save(self, save_dir, id):
        torch.save(self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt")
