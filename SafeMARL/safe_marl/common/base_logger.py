import os
import time

import numpy as np


class BaseLogger:
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.n_threads = algo_args["train"]["n_rollout_threads"]
        self.log_file = open(os.path.join(run_dir, "progress.csv"), "w", encoding="utf-8")
        self.log_file.write("total_steps,eval_return,eval_cost,eval_length\n")

    def get_task_name(self):
        raise NotImplementedError

    def init(self, episodes):
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(self.n_threads)
        self.train_episode_costs = np.zeros(self.n_threads)
        self.train_episode_lens = np.zeros(self.n_threads, dtype=int)
        self.done_episodes_rewards, self.done_episodes_costs, self.done_episodes_lens = [], [], []

    def episode_init(self, episode):
        self.episode = episode

    def per_step(self, data):
        obs, share_obs, rewards, costs, dones, infos, values, actions, action_log_probs = data
        dones_env = np.all(dones, axis=1)
        self.train_episode_rewards += np.mean(rewards, axis=1).flatten()
        self.train_episode_costs += np.mean(costs, axis=1).flatten()
        self.train_episode_lens += 1
        for t in np.where(dones_env)[0]:
            self.done_episodes_rewards.append(self.train_episode_rewards[t])
            self.done_episodes_costs.append(self.train_episode_costs[t])
            self.done_episodes_lens.append(self.train_episode_lens[t])
            self.train_episode_rewards[t] = 0
            self.train_episode_costs[t] = 0
            self.train_episode_lens[t] = 0

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        self.total_num_steps = self.episode * self.algo_args["train"]["episode_length"] * self.n_threads
        fps = int(self.total_num_steps / max(time.time() - self.start, 1e-6))
        print(
            f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} "
            f"updates {self.episode}/{self.episodes} episodes, total num timesteps "
            f"{self.total_num_steps}/{self.algo_args['train']['num_env_steps']}, FPS {fps}."
        )
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)
        print("Average step reward is {}.".format(critic_train_info["average_step_rewards"]))

        if self.done_episodes_rewards:
            aver_rewards = np.mean(self.done_episodes_rewards)
            aver_costs = np.mean(self.done_episodes_costs)
            aver_lens = np.mean(self.done_episodes_lens)
            print(f"Some episodes done, average episode reward {aver_rewards:.2f}, cost {aver_costs:.2f}, length {aver_lens:.1f}.\n")
            self.writter.add_scalar("train/rollout_episode_return", aver_rewards, self.total_num_steps)
            self.writter.add_scalar("train/rollout_episode_cost", aver_costs, self.total_num_steps)
            self.writter.add_scalar("train/rollout_episode_length", aver_lens, self.total_num_steps)
            self.done_episodes_rewards, self.done_episodes_costs, self.done_episodes_lens = [], [], []

    def log_train(self, actor_train_infos, critic_train_info):
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                self.writter.add_scalar(f"agent{agent_id}/{k}", v, self.total_num_steps)
        for k, v in critic_train_info.items():
            self.writter.add_scalar(f"critic/{k}", v, self.total_num_steps)

    def eval_init(self):
        self.total_num_steps = self.episode * self.algo_args["train"]["episode_length"] * self.n_threads
        n_eval = self.algo_args["eval"]["n_eval_rollout_threads"]
        self.eval_episode_rewards = [[] for _ in range(n_eval)]
        self.eval_episode_costs = [[] for _ in range(n_eval)]
        self.eval_episode_lengths = [[] for _ in range(n_eval)]
        self.one_episode_rewards = np.zeros(n_eval)
        self.one_episode_costs = np.zeros(n_eval)
        self.one_episode_lengths = np.zeros(n_eval, dtype=int)

    def eval_per_step(self, eval_data):
        eval_obs, eval_share_obs, eval_rewards, eval_costs, eval_dones, eval_infos = eval_data
        self.one_episode_rewards += eval_rewards[:, 0, 0]
        self.one_episode_costs += eval_costs[:, 0, 0]
        self.one_episode_lengths += 1

    def eval_thread_done(self, tid):
        self.eval_episode_rewards[tid].append(self.one_episode_rewards[tid])
        self.eval_episode_costs[tid].append(self.one_episode_costs[tid])
        self.eval_episode_lengths[tid].append(self.one_episode_lengths[tid])
        self.one_episode_rewards[tid] = 0.0
        self.one_episode_costs[tid] = 0.0
        self.one_episode_lengths[tid] = 0

    def eval_log(self, eval_episode):
        eval_avg_rew = float(np.mean(np.concatenate([r for r in self.eval_episode_rewards if r])))
        eval_avg_cost = float(np.mean(np.concatenate([c for c in self.eval_episode_costs if c])))
        eval_avg_length = float(np.mean(np.concatenate([l for l in self.eval_episode_lengths if l])))
        print(f"[eval] steps={self.total_num_steps} return={eval_avg_rew:.2f} cost={eval_avg_cost:.2f} "
              f"length={eval_avg_length:.1f}", flush=True)
        self.log_file.write(f"{self.total_num_steps},{eval_avg_rew},{eval_avg_cost},{eval_avg_length}\n")
        self.log_file.flush()
        self.writter.add_scalar("eval/average_episode_return", eval_avg_rew, self.total_num_steps)
        self.writter.add_scalar("eval/average_episode_cost", eval_avg_cost, self.total_num_steps)
        self.writter.add_scalar("eval/average_episode_length", eval_avg_length, self.total_num_steps)
        return eval_avg_rew, eval_avg_cost, eval_avg_length

    def close(self):
        self.log_file.close()
