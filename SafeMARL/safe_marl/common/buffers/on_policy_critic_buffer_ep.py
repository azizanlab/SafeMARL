import numpy as np
import torch
from safe_marl.utils.envs_tools import get_shape_from_obs_space


class OnPolicyCriticBufferEP:
    def __init__(self, args, share_obs_space):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        T, N = self.episode_length, self.n_rollout_threads
        self.share_obs = np.zeros((T + 1, N, *share_obs_shape), dtype=np.float32)
        self.value_preds = np.zeros((T + 1, N, 1), dtype=np.float32)
        self.returns = np.zeros((T + 1, N, 1), dtype=np.float32)
        self.rewards = np.zeros((T, N, 1), dtype=np.float32)
        self.masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.step = 0

    def insert(self, share_obs, value_preds, rewards, masks, bad_masks):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        return np.mean(self.rewards)

    def compute_returns(self, next_value, value_normalizer):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = (
                self.rewards[step]
                + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]
                - value_normalizer.denormalize(self.value_preds[step])
            )
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            gae = self.bad_masks[step + 1] * gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])

    def feed_forward_generator_critic(self, critic_num_mini_batch):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        assert batch_size >= critic_num_mini_batch
        mini_batch_size = batch_size // critic_num_mini_batch
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(critic_num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)

        for indices in sampler:
            yield share_obs[indices], value_preds[indices], returns[indices]
