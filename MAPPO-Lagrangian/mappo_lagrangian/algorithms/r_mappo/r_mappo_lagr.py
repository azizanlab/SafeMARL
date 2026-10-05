import numpy as np
import torch
import torch.nn as nn
from mappo_lagrangian.utils.util import huber_loss
from mappo_lagrangian.utils.popart import PopArt
from mappo_lagrangian.algorithms.utils.util import check


class R_MAPPO_Lagr:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma
        self._use_value_active_masks = args.use_value_active_masks

        self.lagrangian_coef = args.lagrangian_coef_rate
        self.lamda_lagr = args.lamda_lagr
        self.safety_bound = args.safety_bound

        self.value_normalizer = PopArt(1, device=self.device)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
        error_original = self.value_normalizer(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        return value_loss

    def ppo_update(self, sample):
        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, active_masks_batch, \
            old_action_log_probs_batch, adv_targ, factor_batch, cost_preds_batch, cost_returns_barch, cost_adv_targ, \
            aver_episode_costs = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        cost_returns_barch = check(cost_returns_barch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, cost_values = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, actions_batch, active_masks_batch)

        adv_targ_hybrid = adv_targ - self.lamda_lagr * cost_adv_targ

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid
        policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) *
                              active_masks_batch).sum() / active_masks_batch.sum()
        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        delta_lamda_lagr = -((aver_episode_costs.mean() - self.safety_bound) * (1 - self.gamma)
                             + (imp_weights * cost_adv_targ)).mean().detach()
        R_Relu = torch.nn.ReLU()
        new_lamda_lagr = R_Relu(self.lamda_lagr - (delta_lamda_lagr * self.lagrangian_coef))
        self.lamda_lagr = new_lamda_lagr

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_barch, active_masks_batch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.value_loss_coef).backward()
        cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
        self.policy.cost_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, \
            cost_grad_norm

    def train(self, buffer):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        cost_adv = buffer.cost_returns[:-1] - self.value_normalizer.denormalize(buffer.cost_preds[:-1])
        cost_adv_copy = cost_adv.copy()
        cost_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_adv = np.nanmean(cost_adv_copy)
        std_cost_adv = np.nanstd(cost_adv_copy)
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        train_info = {}
        for k in ("value_loss", "policy_loss", "dist_entropy", "actor_grad_norm", "critic_grad_norm", "ratio",
                  "cost_grad_norm", "cost_loss"):
            train_info[k] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, cost_adv)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, \
                    cost_grad_norm = self.ppo_update(sample)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["cost_loss"] += cost_loss.item()
                train_info["cost_grad_norm"] += cost_grad_norm

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost_critic.eval()
