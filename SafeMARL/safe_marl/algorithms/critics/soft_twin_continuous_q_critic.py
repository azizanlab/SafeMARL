import itertools
from copy import deepcopy

import torch
import torch.nn.functional as F
from safe_marl.models.value_function_models.continuous_q_net import ContinuousQNet
from safe_marl.utils.envs_tools import check


class SoftTwinContinuousQCritic:
    def __init__(self, args, share_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.num_agents = num_agents
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.critic2 = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in itertools.chain(self.target_critic.parameters(), self.target_critic2.parameters()):
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.polyak = args["polyak"]
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.critic.parameters(), self.critic2.parameters()), lr=args["critic_lr"]
        )
        self.turn_off_grad()

    def soft_update(self):
        for target, source in ((self.target_critic, self.critic), (self.target_critic2, self.critic2)):
            for p_t, p in zip(target.parameters(), source.parameters()):
                p_t.data.copy_(p_t.data * (1.0 - self.polyak) + p.data * self.polyak)

    def get_values(self, share_obs, actions):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return torch.min(self.critic(share_obs, actions), self.critic2(share_obs, actions))

    def train(self, share_obs, actions, reward, term, next_share_obs, next_actions, next_logp_actions, gamma_n, alpha):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = torch.cat([check(a).to(**self.tpdv) for a in actions], dim=-1)
        reward = check(reward).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma_n = check(gamma_n).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        next_logp = torch.sum(torch.cat(next_logp_actions, dim=-1), dim=-1, keepdim=True).to(**self.tpdv)
        with torch.no_grad():
            next_q = torch.min(
                self.target_critic(next_share_obs, next_actions), self.target_critic2(next_share_obs, next_actions)
            )
            q_targets = reward + gamma_n * (next_q - alpha * next_logp) * (1 - term)
        critic_loss = F.mse_loss(self.critic(share_obs, actions), q_targets) + F.mse_loss(
            self.critic2(share_obs, actions), q_targets
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def save(self, save_dir):
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent.pt")
        torch.save(self.critic2.state_dict(), str(save_dir) + "/critic_agent2.pt")
        torch.save(self.target_critic.state_dict(), str(save_dir) + "/target_critic_agent.pt")
        torch.save(self.target_critic2.state_dict(), str(save_dir) + "/target_critic_agent2.pt")

    def turn_on_grad(self):
        for param in itertools.chain(self.critic.parameters(), self.critic2.parameters()):
            param.requires_grad = True

    def turn_off_grad(self):
        for param in itertools.chain(self.critic.parameters(), self.critic2.parameters()):
            param.requires_grad = False
