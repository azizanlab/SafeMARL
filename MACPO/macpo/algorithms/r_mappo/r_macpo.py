import numpy as np
import torch
import torch.nn as nn
from macpo.utils.util import huber_loss
from macpo.utils.popart import PopArt
from macpo.algorithms.utils.util import check
from macpo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor


class R_MACTRPO_CPO():
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.episode_length = args.episode_length

        self.kl_threshold = args.kl_threshold
        self.safety_bound = args.safety_bound
        self.ls_step = args.ls_step
        self.EPS = args.EPS
        self.gamma = args.gamma
        self.line_search_fraction = args.line_search_fraction
        self.fraction_coef = args.fraction_coef
        self._use_value_active_masks = args.use_value_active_masks

        self._max_quad_constraint_val = args.kl_threshold
        self._max_lin_constraint_val = args.safety_bound

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

    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    def kl_divergence(self, obs, action, active_masks, new_actor, old_actor):
        _, _, mu, std = new_actor.evaluate_actions(obs, action, active_masks)
        _, _, mu_old, std_old = old_actor.evaluate_actions(obs, action, active_masks)
        logstd = torch.log(std)
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        logstd_old = torch.log(std_old)
        kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / (self.EPS + 2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def conjugate_gradient(self, actor, obs, action, active_masks, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device=self.device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(actor, obs, action, active_masks, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def fisher_vector_product(self, actor, obs, action, active_masks, p):
        p.detach()
        kl = self.kl_divergence(obs, action, active_masks, new_actor=actor, old_actor=actor)
        kl = kl.mean()
        kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True, allow_unused=True)
        kl_grad = self.flat_grad(kl_grad)
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters(), allow_unused=True)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p

    def trpo_update(self, sample):
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

        values, action_log_probs, dist_entropy, cost_values, action_mu, action_std = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, actions_batch, active_masks_batch)

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

        rescale_constraint_val = (aver_episode_costs.mean() - self._max_lin_constraint_val) * (1 - self.gamma)
        if rescale_constraint_val == 0:
            rescale_constraint_val = self.EPS

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        reward_loss = (torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True) *
                       active_masks_batch).sum() / active_masks_batch.sum()
        reward_loss = - reward_loss
        reward_loss_grad = torch.autograd.grad(reward_loss, self.policy.actor.parameters(), retain_graph=True,
                                               allow_unused=True)
        reward_loss_grad = self.flat_grad(reward_loss_grad)

        cost_loss = (torch.sum(ratio * factor_batch * (cost_adv_targ), dim=-1, keepdim=True) *
                     active_masks_batch).sum() / active_masks_batch.sum()
        cost_loss_grad = torch.autograd.grad(cost_loss, self.policy.actor.parameters(), retain_graph=True,
                                             allow_unused=True)
        cost_loss_grad = self.flat_grad(cost_loss_grad)
        B_cost_loss_grad = cost_loss_grad.unsqueeze(0)
        B_cost_loss_grad = self.flat_grad(B_cost_loss_grad)

        g_step_dir = self.conjugate_gradient(self.policy.actor, obs_batch, actions_batch, active_masks_batch,
                                             reward_loss_grad.data, nsteps=10)
        b_step_dir = self.conjugate_gradient(self.policy.actor, obs_batch, actions_batch, active_masks_batch,
                                             B_cost_loss_grad.data, nsteps=10)

        q_coef = (reward_loss_grad * g_step_dir).sum(0, keepdim=True)
        r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)
        s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)

        fraction = self.line_search_fraction
        loss_improve = 0

        B_cost_loss_grad_dot = torch.dot(B_cost_loss_grad, B_cost_loss_grad)
        if (torch.dot(B_cost_loss_grad, B_cost_loss_grad)) <= self.EPS and rescale_constraint_val < 0:
            b_step_dir = torch.tensor(0)
            r_coef = torch.tensor(0)
            s_coef = torch.tensor(0)
            positive_Cauchy_value = torch.tensor(0)
            whether_recover_policy_value = torch.tensor(0)
            optim_case = 4
        else:
            r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)
            s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)
            if r_coef == 0:
                r_coef = self.EPS
            if s_coef == 0:
                s_coef = self.EPS
            positive_Cauchy_value = (q_coef - (r_coef ** 2) / (self.EPS + s_coef))
            whether_recover_policy_value = 2 * self._max_quad_constraint_val - (rescale_constraint_val ** 2) / (
                self.EPS + s_coef)
            if rescale_constraint_val < 0 and whether_recover_policy_value < 0:
                optim_case = 3
            elif rescale_constraint_val < 0 and whether_recover_policy_value >= 0:
                optim_case = 2
            elif rescale_constraint_val >= 0 and whether_recover_policy_value >= 0:
                optim_case = 1
            else:
                optim_case = 0

        if whether_recover_policy_value == 0:
            whether_recover_policy_value = self.EPS

        if optim_case in [3, 4]:
            lam = torch.sqrt((q_coef / (2 * self._max_quad_constraint_val)))
            nu = torch.tensor(0)
        elif optim_case in [1, 2]:
            LA, LB = [0, r_coef / rescale_constraint_val], [r_coef / rescale_constraint_val, np.inf]
            LA, LB = (LA, LB) if rescale_constraint_val < 0 else (LB, LA)
            proj = lambda x, L: max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(positive_Cauchy_value / whether_recover_policy_value), LA)
            lam_b = proj(torch.sqrt(q_coef / (torch.tensor(2 * self._max_quad_constraint_val))), LB)

            f_a = lambda lam: -0.5 * (positive_Cauchy_value / (self.EPS + lam) + whether_recover_policy_value * lam) \
                - r_coef * rescale_constraint_val / (self.EPS + s_coef)
            f_b = lambda lam: -0.5 * (q_coef / (self.EPS + lam) + 2 * self._max_quad_constraint_val * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * rescale_constraint_val - r_coef) / (self.EPS + s_coef)
        else:
            lam = torch.tensor(0)
            nu = torch.sqrt(torch.tensor(2 * self._max_quad_constraint_val) / (self.EPS + s_coef))

        x_a = (1. / (lam + self.EPS)) * (g_step_dir + nu * b_step_dir)
        x_b = (nu * b_step_dir)
        x = x_a if optim_case > 0 else x_b

        reward_loss = reward_loss.data.cpu().numpy()
        cost_loss = cost_loss.data.cpu().numpy()
        params = self.flat_params(self.policy.actor)

        old_actor = R_Actor(self.policy.args, self.policy.obs_space, self.policy.act_space, self.device)
        self.update_model(old_actor, params)

        expected_improve = -torch.dot(x, reward_loss_grad).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()

        flag = False
        fraction_coef = self.fraction_coef
        for i in range(self.ls_step):
            x_norm = torch.norm(x)
            if x_norm > 0.5:
                x = x * 0.5 / x_norm

            new_params = params - fraction_coef * (fraction ** i) * x
            self.update_model(self.policy.actor, new_params)
            values, action_log_probs, dist_entropy, new_cost_values, action_mu, action_std = \
                self.policy.evaluate_actions(share_obs_batch, obs_batch, actions_batch, active_masks_batch)

            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            new_reward_loss = (torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True) *
                               active_masks_batch).sum() / active_masks_batch.sum()
            new_cost_loss = (torch.sum(ratio * factor_batch * cost_adv_targ, dim=-1, keepdim=True) *
                             active_masks_batch).sum() / active_masks_batch.sum()

            new_reward_loss = new_reward_loss.data.cpu().numpy()
            new_reward_loss = -new_reward_loss
            new_cost_loss = new_cost_loss.data.cpu().numpy()
            loss_improve = new_reward_loss - reward_loss

            kl = self.kl_divergence(obs_batch, actions_batch, active_masks_batch, new_actor=self.policy.actor,
                                    old_actor=old_actor)
            kl = kl.mean()

            if ((kl < self.kl_threshold) and (loss_improve < 0 if optim_case > 1 else True)
                    and (new_cost_loss.mean() - cost_loss.mean() <= max(-rescale_constraint_val, 0))):
                flag = True
                break
            expected_improve *= fraction

        if not flag:
            print("line search failed")
            params = self.flat_params(old_actor)
            self.update_model(self.policy.actor, params)

        return value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, ratio, cost_loss, \
            cost_grad_norm, whether_recover_policy_value, cost_preds_batch, cost_returns_barch, B_cost_loss_grad, \
            lam, nu, g_step_dir, b_step_dir, x, action_mu, action_std, B_cost_loss_grad_dot

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
        for k in ("value_loss", "kl", "dist_entropy", "loss_improve", "expected_improve", "critic_grad_norm", "ratio",
                  "cost_loss", "cost_grad_norm", "whether_recover_policy_value", "cost_preds_batch",
                  "cost_returns_barch", "B_cost_loss_grad", "lam", "nu", "g_step_dir", "b_step_dir", "x", "action_mu",
                  "action_std", "B_cost_loss_grad_dot"):
            train_info[k] = 0

        data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, cost_adv)
        for sample in data_generator:
            value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, imp_weights, cost_loss, \
                cost_grad_norm, whether_recover_policy_value, cost_preds_batch, cost_returns_barch, B_cost_loss_grad, \
                lam, nu, g_step_dir, b_step_dir, x, action_mu, action_std, B_cost_loss_grad_dot = \
                self.trpo_update(sample)

            train_info["value_loss"] += value_loss.item()
            train_info["kl"] += kl
            train_info["loss_improve"] += loss_improve
            train_info["expected_improve"] += expected_improve
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["critic_grad_norm"] += critic_grad_norm
            train_info["ratio"] += imp_weights.mean()
            train_info["cost_loss"] += value_loss.item()
            train_info["cost_grad_norm"] += cost_grad_norm
            train_info["whether_recover_policy_value"] += whether_recover_policy_value
            train_info["cost_preds_batch"] += cost_preds_batch.mean()
            train_info["cost_returns_barch"] += cost_returns_barch.mean()
            train_info["B_cost_loss_grad"] += B_cost_loss_grad.mean()
            train_info["g_step_dir"] += g_step_dir.float().mean()
            train_info["b_step_dir"] += b_step_dir.float().mean()
            train_info["x"] = x.float().mean()
            train_info["action_mu"] += action_mu.float().mean()
            train_info["action_std"] += action_std.float().mean()
            train_info["B_cost_loss_grad_dot"] += B_cost_loss_grad_dot.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
