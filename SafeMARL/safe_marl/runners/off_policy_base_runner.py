import os

import numpy as np
import setproctitle
import torch

from safe_marl.common.buffers.off_policy_buffer import OffPolicyBuffer
from safe_marl.common.local_logger import LocalLogger
from safe_marl.utils.configs_tools import get_task_name, init_dir, save_config
from safe_marl.utils.envs_tools import get_num_agents, make_eval_env, make_train_env, set_seed
from safe_marl.utils.models_tools import init_device


class OffPolicyBaseRunner:
    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.train_args = algo_args["train"]
        self.n_threads = self.train_args["n_rollout_threads"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"], env_args, args["algo"], args["exp_name"], algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        self.logger = LocalLogger(self.run_dir, self.writter)
        setproctitle.setproctitle(f"{args['algo']}-{self.task_name}-{args['exp_name']}")

        self.envs = make_train_env(args["env"], algo_args["seed"]["seed"], self.n_threads, env_args)
        self.eval_envs = (
            make_eval_env(args["env"], algo_args["seed"]["seed"], algo_args["eval"]["n_eval_rollout_threads"], env_args)
            if algo_args["eval"]["use_eval"]
            else None
        )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)
        print("observation_space:", self.envs.observation_space)
        print("share_observation_space:", self.envs.share_observation_space)
        print("action_space:", self.envs.action_space)

        self.buffer = OffPolicyBuffer(
            {**algo_args["train"], **algo_args["algo"]},
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
        )
        self.total_it = 0
        self._build()

    def _build(self):
        raise NotImplementedError

    def get_actions(self, obs, share_obs, stochastic=True):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def run(self):
        self.episode_returns = np.zeros(self.n_threads)
        self.episode_costs = np.zeros(self.n_threads)
        self.finished_returns, self.finished_costs = [], []

        print("start warmup", flush=True)
        obs, share_obs = self.warmup()
        self.finished_returns, self.finished_costs = [], []
        print("finish warmup, start training", flush=True)

        steps = self.train_args["num_env_steps"] // self.n_threads
        update_num = int(self.train_args["update_per_train"] * self.train_args["train_interval"])
        train_stats = []

        if self.eval_envs is not None:
            self.eval(0)
        if self.train_args.get("keep_checkpoints"):
            self._save_checkpoint(0)
        for step in range(1, steps + 1):
            total_steps = step * self.n_threads
            actions = self.get_actions(obs, share_obs, stochastic=True)
            new_obs, new_share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)
            self.insert(share_obs, obs, actions, rewards, costs, dones, infos, new_share_obs, new_obs)
            obs, share_obs = new_obs, new_share_obs

            if step % self.train_args["train_interval"] == 0:
                for _ in range(update_num):
                    train_stats.append(self.train())

            if step % self.train_args["log_interval"] == 0 and train_stats:
                keys = sorted(set().union(*train_stats))
                stats = {k: float(np.mean([s[k] for s in train_stats if k in s])) for k in keys}
                if self.finished_returns:
                    stats["rollout_episode_return"] = float(np.mean(self.finished_returns))
                    stats["rollout_episode_cost"] = float(np.mean(self.finished_costs))
                    self.finished_returns, self.finished_costs = [], []
                self.logger.log_train(total_steps, stats)
                train_stats = []

            if step % self.train_args["eval_interval"] == 0:
                if self.eval_envs is not None:
                    self.eval(total_steps)
                self._save_checkpoint(total_steps)

    def _save_checkpoint(self, total_steps):
        self.save()
        if self.train_args.get("keep_checkpoints"):
            latest = self.save_dir
            self.save_dir = os.path.join(str(latest), f"step_{total_steps}")
            os.makedirs(self.save_dir, exist_ok=True)
            self.save()
            self.save_dir = latest

    def warmup(self):
        warmup_steps = self.train_args["warmup_steps"] // self.n_threads
        obs, share_obs, _ = self.envs.reset()
        for _ in range(warmup_steps):
            actions = np.array(
                [[self.action_spaces[a].sample() for a in range(self.num_agents)] for _ in range(self.n_threads)]
            )
            new_obs, new_share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)
            self.insert(share_obs, obs, actions, rewards, costs, dones, infos, new_share_obs, new_obs)
            obs, share_obs = new_obs, new_share_obs
        return obs, share_obs

    def insert(self, share_obs, obs, actions, rewards, costs, dones, infos, next_share_obs, next_obs):
        next_obs = next_obs.copy()
        next_share_obs = next_share_obs.copy()
        dones_env = np.all(dones, axis=1)
        h = np.array([info[0]["constraint_value_before"][0][0] for info in infos], dtype=np.float32)[:, None]
        next_h = np.array([info[0]["constraint_value_after"][0][0] for info in infos], dtype=np.float32)[:, None]

        self.episode_returns += rewards[:, 0, 0]
        self.episode_costs += costs[:, 0, 0]
        terms = np.zeros((self.n_threads, 1), dtype=bool)
        for i in range(self.n_threads):
            if dones_env[i]:
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"]
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"]
                terms[i, 0] = not infos[i][0].get("bad_transition", False)
                self.finished_returns.append(self.episode_returns[i])
                self.finished_costs.append(self.episode_costs[i])
                self.episode_returns[i] = 0.0
                self.episode_costs[i] = 0.0

        self.buffer.insert(
            {
                "share_obs": share_obs[:, 0],
                "obs": [obs[:, a] for a in range(self.num_agents)],
                "actions": [actions[:, a] for a in range(self.num_agents)],
                "rewards": rewards[:, 0],
                "h": h,
                "next_h": next_h,
                "dones": dones_env[:, None],
                "terms": terms,
                "next_share_obs": next_share_obs[:, 0],
                "next_obs": [next_obs[:, a] for a in range(self.num_agents)],
            }
        )

    @torch.no_grad()
    def eval(self, total_steps):
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]
        returns, costs, lengths = [], [], []
        ep_return, ep_cost, ep_len = np.zeros(n_eval), np.zeros(n_eval), np.zeros(n_eval, dtype=int)
        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        while len(returns) < self.algo_args["eval"]["eval_episodes"]:
            actions = self.get_actions(eval_obs, eval_share_obs, stochastic=False)
            eval_obs, eval_share_obs, rewards, step_costs, dones, _, _ = self.eval_envs.step(actions)
            ep_return += rewards[:, 0, 0]
            ep_cost += step_costs[:, 0, 0]
            ep_len += 1
            for i in np.where(np.all(dones, axis=1))[0]:
                returns.append(ep_return[i])
                costs.append(ep_cost[i])
                lengths.append(ep_len[i])
                ep_return[i], ep_cost[i], ep_len[i] = 0.0, 0.0, 0
        self.logger.log_eval(total_steps, float(np.mean(returns)), float(np.mean(costs)), float(np.mean(lengths)))
        self.logger.log_eval_episodes(total_steps, returns, costs)

    def close(self):
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()
        self.writter.export_scalars_to_json(os.path.join(self.log_dir, "summary.json"))
        self.writter.close()
        self.logger.close()
