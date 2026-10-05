import numpy as np
import torch
import torch.nn as nn
from safe_marl.utils.envs_tools import check
from safe_marl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(HAPPO, self).__init__(args, obs_space, act_space, device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.max_grad_norm = args["max_grad_norm"]

    def update(self, sample):
        obs_batch, actions_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, factor_batch = sample
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        action_log_probs, dist_entropy, _ = self.evaluate_actions(obs_batch, actions_batch, active_masks_batch)
        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch), dim=-1, keepdim=True)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        surrogate = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = (surrogate * active_masks_batch).sum() / active_masks_batch.sum()

        self.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages):
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0}
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        advantages_copy = advantages.copy()
        advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
        advantages = (advantages - np.nanmean(advantages_copy)) / (np.nanstd(advantages_copy) + 1e-5)

        for _ in range(self.ppo_epoch):
            for sample in actor_buffer.feed_forward_generator_actor(advantages, self.actor_num_mini_batch):
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates
        return train_info
