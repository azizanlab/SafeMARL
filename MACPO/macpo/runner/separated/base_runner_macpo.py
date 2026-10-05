import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

from macpo.utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / "models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        from macpo.algorithms.r_mappo.r_macpo import R_MACTRPO_CPO as TrainAlgo
        from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            po = Policy(self.all_args, self.envs.observation_space[agent_id],
                        self.envs.share_observation_space[agent_id], self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            bu = SeparatedReplayBuffer(self.all_args, self.envs.observation_space[agent_id],
                                       self.envs.share_observation_space[agent_id], self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

            next_costs = self.trainer[agent_id].policy.get_cost_values(self.buffer[agent_id].share_obs[-1])
            next_costs = _t2n(next_costs)
            self.buffer[agent_id].compute_cost_returns(next_costs, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            old_actions_logprob, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            factor = factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(
                self.episode_length, self.n_rollout_threads, action_dim))
            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def save_checkpoint(self, total_num_steps):
        latest = self.save_dir
        self.save_dir = os.path.join(str(latest), "step_" + str(total_num_steps))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save()
        self.save_dir = latest
