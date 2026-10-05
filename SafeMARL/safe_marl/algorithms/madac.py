import itertools
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from safe_marl.models.policy_models.deterministic_policy import DeterministicPolicy
from safe_marl.models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from safe_marl.models.value_function_models.continuous_q_net import ContinuousQNet
from safe_marl.utils.envs_tools import check


class TwinQ:
    def __init__(self, args, share_obs_space, act_spaces, lr, device):
        self.q1 = ContinuousQNet(args, share_obs_space, act_spaces, device)
        self.q2 = ContinuousQNet(args, share_obs_space, act_spaces, device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.target_parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.tau = args["polyak"]

    def parameters(self):
        return list(itertools.chain(self.q1.parameters(), self.q2.parameters()))

    def target_parameters(self):
        return list(itertools.chain(self.q1_target.parameters(), self.q2_target.parameters()))

    def value(self, share_obs, joint_action):
        return torch.min(self.q1(share_obs, joint_action), self.q2(share_obs, joint_action))

    def target_value(self, share_obs, joint_action):
        return torch.min(self.q1_target(share_obs, joint_action), self.q2_target(share_obs, joint_action))

    def update(self, share_obs, joint_action, target, max_grad_norm, expectile=0.5):
        def loss_fn(q):
            if expectile == 0.5:
                return F.mse_loss(q, target)
            weight = torch.where(target > q, 2.0 * expectile, 2.0 * (1.0 - expectile))
            return (weight.detach() * (q - target) ** 2).mean()

        loss = loss_fn(self.q1(share_obs, joint_action)) + loss_fn(self.q2(share_obs, joint_action))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def soft_update(self):
        with torch.no_grad():
            for p_t, p in zip(self.target_parameters(), self.parameters()):
                p_t.mul_(1.0 - self.tau).add_(p, alpha=self.tau)

    def requires_grad_(self, flag):
        for p in self.parameters():
            p.requires_grad = flag

    def state_dict(self):
        return {"q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict()}

    def load_state_dict(self, state):
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])


class MADAC:
    def __init__(self, args, obs_spaces, act_spaces, share_obs_space, num_agents, device):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.n = num_agents
        self.act_half_ranges = [
            torch.as_tensor((space.high - space.low) / 2.0, dtype=torch.float32, device=device) for space in act_spaces
        ]

        self.gamma = float(args["gamma"])
        self.gamma_h = float(args["safety_gamma"])
        self.alpha = float(args["alpha"])
        self.max_grad_norm = float(args["max_grad_norm"])
        self.safety_expectile = float(args["safety_expectile"])
        assert 0.0 < self.safety_expectile <= 0.5
        self.safety_proximity = float(args["safety_proximity"])
        assert self.safety_proximity >= 0.0
        self.region_margin = float(args["region_margin"])
        assert self.region_margin <= 0.0
        self.constraint_tightening = float(args["constraint_tightening"])
        assert self.constraint_tightening >= 0.0
        self.rho = float(args["rho"])
        self.beta_lambda = float(args["multiplier_lr"])
        assert 0.0 < self.beta_lambda <= self.rho
        self.lambda_max = float(args["multiplier_max"])
        self.lambda_ = [0.0] * self.n
        self.policy_average_fraction = float(args["policy_average_fraction"])
        assert self.policy_average_fraction > 0.0
        self.actor_interval = int(args["actor_interval"])
        assert self.actor_interval >= 1

        self.task_actors = [SquashedGaussianPolicy(args, obs_spaces[i], act_spaces[i], device) for i in range(self.n)]
        self.safety_actors = [DeterministicPolicy(args, obs_spaces[i], act_spaces[i], device) for i in range(self.n)]
        self.task_optimizers = [torch.optim.Adam(a.parameters(), lr=args["lr"]) for a in self.task_actors]
        self.safety_optimizers = [torch.optim.Adam(a.parameters(), lr=args["safety_lr"]) for a in self.safety_actors]
        self.reward_critic = TwinQ(args, share_obs_space, act_spaces, args["critic_lr"], device)
        self.safety_critic = TwinQ(args, share_obs_space, act_spaces, args["safety_critic_lr"], device)
        self.eval_actors = [deepcopy(a) for a in self.task_actors]
        for a in self.eval_actors:
            for p in a.parameters():
                p.requires_grad = False

        self.first_iteration = True
        self.n_updates = 0
        self.n_actor_updates = 0

    def _joint(self, actions):
        return torch.cat(actions, dim=-1)

    def _replace(self, actions, i, new_action):
        return self._joint([new_action if j == i else actions[j] for j in range(self.n)])

    def _order(self):
        return list(np.random.permutation(self.n))

    def _to(self, x):
        return check(x).to(**self.tpdv)

    def _sample(self, i, obs):
        return self.task_actors[i](obs, stochastic=True, with_logprob=True)

    def _step(self, optimizer, params, loss):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

    @staticmethod
    def _masked_mean(values, mask):
        count = mask.sum()
        return (values * mask).sum() / count if count > 0 else values.sum() * 0.0

    @torch.no_grad()
    def act(self, obs, share_obs=None, stochastic=True):
        actors = self.task_actors if stochastic else self.eval_actors
        actions = [actors[i](self._to(obs[:, i]), stochastic=stochastic, with_logprob=False)[0] for i in range(self.n)]
        return np.stack([a.cpu().numpy() for a in actions]).transpose(1, 0, 2)

    def _safety_target(self, h_chain, h_valid, bootstrap):
        value = bootstrap
        for k in range(h_chain.shape[0] - 1, -1, -1):
            value = torch.where(h_valid[k] > 0, self.gamma_h * torch.min(h_chain[k], value), value)
        return value

    @staticmethod
    def _successor_chain(h_seq, h_valid, next_h):
        nxt = next_h.unsqueeze(0)
        if h_seq.shape[0] == 1:
            return nxt
        later_valid = torch.cat([h_valid[1:], torch.zeros_like(h_valid[:1])], dim=0)
        return torch.where(later_valid > 0, torch.cat([h_seq[1:], nxt], dim=0), nxt)

    def _constraint(self, h):
        return h - self.constraint_tightening if self.constraint_tightening > 0 else h

    def _state_value(self, w, h):
        return self.gamma_h * torch.min(h, w)

    def update(self, batch):
        self.n_updates += 1
        b = self._tensors(batch)
        stats = {}
        if self.n_updates % self.actor_interval != 0:
            self._update_safety_critic(b, stats)
            self._update_reward_critic(b, stats)
            self._update_targets(actors_updated=False)
            return stats
        self.n_actor_updates += 1
        chi_old = self._old_membership(b)
        self._update_safety_critic(b, stats)
        region = self._update_safety_actors(b, stats)
        stats["chi_old_frac"] = chi_old.mean().item()
        stats["chi_new_frac"] = region.mask_in.mean().item()
        stats["safety_value_mean"] = region.value.mean().item()
        self._initialize_task_actors(b, chi_old, region, stats)
        self._update_reward_critic(b, stats)
        self._update_task_actors(b, region, stats)
        self._update_targets(actors_updated=True)
        self.first_iteration = False
        return stats

    def _tensors(self, batch):
        h_seq = self._constraint(self._to(batch["h_seq"]))
        return SimpleNamespace(
            s=self._to(batch["share_obs"]), s2=self._to(batch["next_share_obs"]),
            obs=[self._to(o) for o in batch["obs"]], obs2=[self._to(o) for o in batch["next_obs"]],
            u=self._joint([self._to(x) for x in batch["actions"]]),
            rewards=self._to(batch["rewards"]), gamma_n=self._to(batch["gamma_n"]), term=self._to(batch["term"]),
            h=h_seq[0], h2=self._constraint(self._to(batch["next_h"])), h_seq=h_seq,
            h_valid=self._to(batch["h_valid"]),
        )

    @torch.no_grad()
    def _old_membership(self, b):
        if self.first_iteration:
            return torch.zeros_like(b.term)
        u_h_old = self._joint([self.safety_actors[i](b.obs[i]) for i in range(self.n)])
        return (self._state_value(self.safety_critic.value(b.s, u_h_old), b.h) >= self.region_margin).float()

    def _bootstrap_value(self, b):
        u2_h = self._joint([self.safety_actors[i](b.obs2[i]) for i in range(self.n)])
        v2_h = self.safety_critic.target_value(b.s2, u2_h).clamp(max=0.0)
        return torch.where(b.term > 0, self.gamma_h * b.h2.clamp(max=0.0), v2_h)

    def _update_safety_critic(self, b, stats):
        with torch.no_grad():
            bootstrap = self._bootstrap_value(b)
            safety_target = self._safety_target(self._successor_chain(b.h_seq, b.h_valid, b.h2), b.h_valid, bootstrap)
        self.safety_critic.requires_grad_(True)
        stats["safety_critic_loss"] = self.safety_critic.update(b.s, b.u, safety_target, self.max_grad_norm,
                                                                self.safety_expectile)
        self.safety_critic.requires_grad_(False)

    def _update_safety_actors(self, b, stats):
        with torch.no_grad():
            u_h = [self.safety_actors[i](b.obs[i]) for i in range(self.n)]
            mu = [self.task_actors[i](b.obs[i], stochastic=False, with_logprob=False)[0] for i in range(self.n)]
        for i in self._order():
            u_hi = self.safety_actors[i](b.obs[i])
            q_h_i = self.safety_critic.value(b.s, self._replace(u_h, i, u_hi))
            half = self.act_half_ranges[i]
            g = torch.autograd.grad(q_h_i.sum(), u_hi, retain_graph=True)[0].detach() * half
            beta_h = self.safety_proximity * g.pow(2).sum(dim=-1).mean().sqrt()
            loss = -q_h_i.mean() + beta_h * (((u_hi - mu[i]) / half) ** 2).sum(dim=-1).mean()
            self._step(self.safety_optimizers[i], self.safety_actors[i].parameters(), loss)
            with torch.no_grad():
                u_h[i] = self.safety_actors[i](b.obs[i])
            stats[f"safety_actor_loss/agent{i}"] = loss.item()

        with torch.no_grad():
            value = self._state_value(self.safety_critic.value(b.s, self._joint(u_h)), b.h)
            mask_in = (value >= self.region_margin).float()
        return SimpleNamespace(mask_in=mask_in, mask_out=1.0 - mask_in, u_h_target=[a.detach() for a in u_h],
                               value=value)

    def _initialize_task_actors(self, b, chi_old, region, stats):
        if chi_old.min() >= 1:
            return
        not_old = 1.0 - chi_old
        for i in range(self.n):
            u_i, _ = self._sample(i, b.obs[i])
            fit = (((u_i - region.u_h_target[i]) / self.act_half_ranges[i]) ** 2).sum(dim=-1, keepdim=True)
            loss = (not_old * fit).mean()
            self._step(self.task_optimizers[i], self.task_actors[i].parameters(), loss)
            stats[f"init_loss/agent{i}"] = loss.item()

    def _update_reward_critic(self, b, stats):
        with torch.no_grad():
            u2, logp2 = zip(*[self._sample(i, b.obs2[i]) for i in range(self.n)])
            log_pi2 = sum(logp2)
            u2_h = self._joint([self.safety_actors[i](b.obs2[i]) for i in range(self.n)])
            v2_h = self._state_value(self.safety_critic.value(b.s2, u2_h), b.h2)
            log_pi2 = log_pi2 * (v2_h >= self.region_margin).float()
            reward_target = b.rewards + b.gamma_n * (1 - b.term) * (
                self.reward_critic.target_value(b.s2, self._joint(u2)) - self.alpha * log_pi2
            )
        self.reward_critic.requires_grad_(True)
        stats["critic_loss"] = self.reward_critic.update(b.s, b.u, reward_target, self.max_grad_norm)
        self.reward_critic.requires_grad_(False)

    def _in_region_loss(self, i, q, c, logp_i):
        lam = self.lambda_[i]
        return self.alpha * logp_i - q + (F.relu(lam - self.rho * c) ** 2 - lam ** 2) / (2.0 * self.rho)

    def _update_multiplier(self, i, c_new, mask_in):
        lam = self.lambda_[i]
        target = self._masked_mean(F.relu(lam - self.rho * c_new), mask_in).item()
        new_lam = lam + self.beta_lambda / self.rho * (target - lam)
        self.lambda_[i] = min(max(new_lam, 0.0), self.lambda_max)

    def _update_task_actors(self, b, region, stats):
        with torch.no_grad():
            u_pi = [self._sample(i, b.obs[i])[0] for i in range(self.n)]
        for i in self._order():
            u_i, logp_i = self._sample(i, b.obs[i])
            joint = self._replace(u_pi, i, u_i)
            q = self.reward_critic.value(b.s, joint)
            c = self.safety_critic.value(b.s, joint)
            loss_in = self._masked_mean(self._in_region_loss(i, q, c, logp_i), region.mask_in)
            fit_out = (((u_i - region.u_h_target[i]) / self.act_half_ranges[i]) ** 2).sum(dim=-1, keepdim=True)
            loss_out = self._masked_mean(fit_out, region.mask_out)
            self._step(self.task_optimizers[i], self.task_actors[i].parameters(), loss_in + loss_out)

            with torch.no_grad():
                u_pi[i] = self._sample(i, b.obs[i])[0]
                if region.mask_in.sum() > 0:
                    self._update_multiplier(i, self.safety_critic.value(b.s, self._joint(u_pi)), region.mask_in)
                active = ((c < 0).float() * region.mask_in).sum() / max(region.mask_in.sum().item(), 1.0)
            stats[f"task_actor_loss/agent{i}"] = (loss_in + loss_out).item()
            stats[f"lambda/agent{i}"] = float(self.lambda_[i])
            stats[f"violation_frac/agent{i}"] = active.item()

        with torch.no_grad():
            joint = self._joint(u_pi)
            stats["task_q_mean"] = self.reward_critic.value(b.s, joint).mean().item()
            c_pi = self.safety_critic.value(b.s, joint)
            stats["task_safety_value_mean"] = c_pi.mean().item()
            stats["task_infeasible_frac"] = self._masked_mean((c_pi < 0).float(), region.mask_in).item()

    def _update_targets(self, actors_updated):
        self.reward_critic.soft_update()
        self.safety_critic.soft_update()
        if actors_updated:
            rate = min(1.0, 1.0 / (self.policy_average_fraction * self.n_actor_updates))
            with torch.no_grad():
                for avg, actor in zip(self.eval_actors, self.task_actors):
                    for p_avg, p in zip(avg.parameters(), actor.parameters()):
                        p_avg.lerp_(p, rate)

    def save(self, save_dir):
        state = {
            "task_actors": [a.state_dict() for a in self.task_actors],
            "safety_actors": [a.state_dict() for a in self.safety_actors],
            "reward_critic": self.reward_critic.state_dict(),
            "safety_critic": self.safety_critic.state_dict(),
            "lambda": [float(v) for v in self.lambda_],
            "eval_actors": [a.state_dict() for a in self.eval_actors],
        }
        torch.save(state, str(save_dir) + "/madac.pt")

    def restore(self, model_dir):
        state = torch.load(str(model_dir) + "/madac.pt", map_location=self.device)
        for a, sd in zip(self.task_actors, state["task_actors"]):
            a.load_state_dict(sd)
        for a, sd in zip(self.safety_actors, state["safety_actors"]):
            a.load_state_dict(sd)
        for a, sd in zip(self.eval_actors, state["eval_actors"]):
            a.load_state_dict(sd)
        self.reward_critic.load_state_dict(state["reward_critic"])
        self.safety_critic.load_state_dict(state["safety_critic"])
        self.lambda_ = [float(v) for v in state["lambda"]]
        self.first_iteration = False
