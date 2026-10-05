import numpy as np
import torch

from macpo.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        self.aver_episode_costs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape),
                                           dtype=np.float32)
        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.costs = np.zeros_like(self.rewards)
        self.cost_preds = np.zeros_like(self.value_preds)
        self.cost_returns = np.zeros_like(self.returns)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.factor = None
        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def return_aver_insert(self, aver_episode_costs):
        self.aver_episode_costs = aver_episode_costs.copy()

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, active_masks, costs,
               cost_preds):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.active_masks[self.step + 1] = active_masks.copy()
        self.costs[self.step] = costs.copy()
        self.cost_preds[self.step] = cost_preds.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) \
                * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])

    def compute_cost_returns(self, next_cost, value_normalizer):
        self.cost_preds[-1] = next_cost
        gae = 0
        for step in reversed(range(self.costs.shape[0])):
            delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1]) \
                * self.masks[step + 1] - value_normalizer.denormalize(self.cost_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])

    def feed_forward_generator(self, advantages, num_mini_batch, cost_adv):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        assert batch_size >= num_mini_batch
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        cost_preds = self.cost_preds[:-1].reshape(-1, 1)
        cost_returns = self.cost_returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        aver_episode_costs = self.aver_episode_costs
        factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)
        cost_adv = cost_adv.reshape(-1, 1)

        for indices in sampler:
            yield share_obs[indices], obs[indices], actions[indices], value_preds[indices], returns[indices], \
                active_masks[indices], action_log_probs[indices], advantages[indices], factor[indices], \
                cost_preds[indices], cost_returns[indices], cost_adv[indices], aver_episode_costs
