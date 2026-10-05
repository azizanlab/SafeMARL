import numpy as np
from safe_marl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OffPolicyBuffer:
    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        self.buffer_size = int(args["buffer_size"])
        self.batch_size = int(args["batch_size"])
        self.n_step = int(args["n_step"])
        self.n_rollout_threads = int(args["n_rollout_threads"])
        self.gamma = float(args["gamma"])
        self.num_agents = num_agents
        self.cur_size = 0
        self.idx = 0

        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        self.share_obs = np.zeros((self.buffer_size, *share_obs_shape), dtype=np.float32)
        self.next_share_obs = np.zeros((self.buffer_size, *share_obs_shape), dtype=np.float32)
        self.obs, self.next_obs, self.actions = [], [], []
        for agent_id in range(num_agents):
            obs_shape = get_shape_from_obs_space(obs_spaces[agent_id])
            act_dim = get_shape_from_act_space(act_spaces[agent_id])
            self.obs.append(np.zeros((self.buffer_size, *obs_shape), dtype=np.float32))
            self.next_obs.append(np.zeros((self.buffer_size, *obs_shape), dtype=np.float32))
            self.actions.append(np.zeros((self.buffer_size, act_dim), dtype=np.float32))
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.h = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_h = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=bool)
        self.terms = np.zeros((self.buffer_size, 1), dtype=bool)

    def insert(self, data):
        length = data["share_obs"].shape[0]
        slots = (self.idx + np.arange(length)) % self.buffer_size
        self.share_obs[slots] = data["share_obs"]
        self.next_share_obs[slots] = data["next_share_obs"]
        self.rewards[slots] = data["rewards"]
        self.h[slots] = data["h"]
        self.next_h[slots] = data["next_h"]
        self.dones[slots] = data["dones"]
        self.terms[slots] = data["terms"]
        for agent_id in range(self.num_agents):
            self.obs[agent_id][slots] = data["obs"][agent_id]
            self.next_obs[agent_id][slots] = data["next_obs"][agent_id]
            self.actions[agent_id][slots] = data["actions"][agent_id]
        self.idx = (self.idx + length) % self.buffer_size
        self.cur_size = min(self.cur_size + length, self.buffer_size)

    def _update_end_flag(self):
        self.end_flag = self.dones[:, 0].copy()
        newest = (self.idx - 1 - np.arange(self.n_rollout_threads)) % self.cur_size
        self.end_flag[newest] = True

    def _next(self, indices):
        return (indices + (1 - self.end_flag[indices]) * self.n_rollout_threads) % self.buffer_size

    def sample(self):
        self._update_end_flag()
        first = np.random.randint(0, self.cur_size, size=self.batch_size)
        batch_size = first.shape[0]

        indices = [first]
        for _ in range(self.n_step - 1):
            indices.append(self._next(indices[-1]))
        n_valid = np.full(batch_size, self.n_step, dtype=np.int64)
        for k in range(self.n_step - 1, -1, -1):
            n_valid[self.end_flag[indices[k]]] = k + 1

        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        for k in range(self.n_step - 1, -1, -1):
            active = (k < n_valid)[:, None]
            rewards = np.where(active, self.rewards[indices[k]] + self.gamma * rewards, rewards)
        last = np.stack(indices, axis=0)[n_valid - 1, np.arange(batch_size)]
        h_seq = np.stack([self.h[indices[k]] for k in range(self.n_step)], axis=0)
        h_valid = (np.arange(self.n_step)[:, None] < n_valid[None, :])[:, :, None]

        return {
            "share_obs": self.share_obs[first],
            "obs": [self.obs[i][first] for i in range(self.num_agents)],
            "actions": [self.actions[i][first] for i in range(self.num_agents)],
            "rewards": rewards,
            "gamma_n": (self.gamma ** n_valid)[:, None].astype(np.float32),
            "next_share_obs": self.next_share_obs[last],
            "next_obs": [self.next_obs[i][last] for i in range(self.num_agents)],
            "next_h": self.next_h[last],
            "term": self.terms[last].astype(np.float32),
            "h_seq": h_seq,
            "h_valid": h_valid.astype(np.float32),
        }
