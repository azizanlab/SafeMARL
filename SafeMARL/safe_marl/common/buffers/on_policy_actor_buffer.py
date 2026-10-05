import numpy as np
import torch
from safe_marl.utils.envs_tools import get_shape_from_act_space, get_shape_from_obs_space


class OnPolicyActorBuffer:
    def __init__(self, args, obs_space, act_space):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        obs_shape = get_shape_from_obs_space(obs_space)
        act_shape = get_shape_from_act_space(act_space)
        T, N = self.episode_length, self.n_rollout_threads
        self.obs = np.zeros((T + 1, N, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((T, N, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((T, N, act_shape), dtype=np.float32)
        self.active_masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.factor = None
        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def insert(self, obs, actions, action_log_probs, active_masks):
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.active_masks[self.step + 1] = active_masks.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def feed_forward_generator_actor(self, advantages, actor_num_mini_batch):
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        assert batch_size >= actor_num_mini_batch
        mini_batch_size = batch_size // actor_num_mini_batch
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(actor_num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            yield (obs[indices], actions[indices], active_masks[indices], action_log_probs[indices],
                   advantages[indices], factor[indices])
