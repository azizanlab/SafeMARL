import os
import time

import numpy as np
import torch

from mappo_lagrangian.runner.separated.base_runner_mappo_lagr import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs = [0 for _ in range(self.n_rollout_threads)]
        train_episode_lens = [0 for _ in range(self.n_rollout_threads)]

        for episode in range(0, episodes + 1):
            done_episodes_rewards = []
            done_episodes_costs = []
            done_episodes_lens = []

            for step in range(self.episode_length):
                values, actions, action_log_probs, cost_preds = self.collect(step)

                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                cost_env = np.mean(costs, axis=1).flatten()
                train_episode_rewards += reward_env
                train_episode_costs += cost_env
                train_episode_lens += np.ones(np.shape(cost_env), dtype=int)

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_costs.append(train_episode_costs[t])
                        train_episode_costs[t] = 0
                        done_episodes_lens.append(train_episode_lens[t])
                        train_episode_lens[t] = 0
                data = obs, share_obs, rewards, costs, dones, infos, values, actions, action_log_probs, cost_preds

                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = episode * self.episode_length * self.n_rollout_threads
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario, self.algorithm_name, self.experiment_name, episode, episodes,
                              total_num_steps, self.num_env_steps, int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_costs = np.mean(done_episodes_costs)
                    self.return_aver_cost(aver_episode_costs)
                    print("some episodes done, average rewards: {}, average costs: {}".format(aver_episode_rewards,
                                                                                              aver_episode_costs))

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                if self.all_args.keep_checkpoints:
                    self.save_checkpoint(total_num_steps)

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        obs, share_obs, _ = self.envs.reset()
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        cost_preds_collector = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, cost_pred = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step], self.buffer[agent_id].obs[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            cost_preds_collector.append(_t2n(cost_pred))
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        cost_preds = np.array(cost_preds_collector).transpose(1, 0, 2)

        return values, actions, action_log_probs, cost_preds

    def insert(self, data):
        obs, share_obs, rewards, costs, dones, infos, values, actions, action_log_probs, cost_preds = data
        dones_env = np.all(dones, axis=1)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id], values[:, agent_id], rewards[:, agent_id],
                                         masks[:, agent_id], active_masks[:, agent_id], costs[:, agent_id],
                                         cost_preds[:, agent_id])

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_costs = []
        one_episode_costs = []
        eval_episode_lengths = []
        one_episode_lengths = []

        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
            one_episode_costs.append([])
            eval_episode_costs.append([])
            one_episode_lengths.append([])
            eval_episode_lengths.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions = self.trainer[agent_id].policy.act(eval_obs[:, agent_id], deterministic=True)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            eval_obs, eval_share_obs, eval_rewards, eval_costs, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                one_episode_costs[eval_i].append(eval_costs[eval_i])
                one_episode_lengths[eval_i].append(1)

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    eval_episode_costs[eval_i].append(np.sum(one_episode_costs[eval_i], axis=0))
                    one_episode_costs[eval_i] = []
                    eval_episode_lengths[eval_i].append(np.sum(one_episode_lengths[eval_i], axis=0))
                    one_episode_lengths[eval_i] = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_episode_costs = np.concatenate(eval_episode_costs)
                eval_episode_lengths = np.concatenate(eval_episode_lengths)

                eval_average_episode_rewards = np.mean(eval_episode_rewards)
                eval_average_episode_costs = np.mean(eval_episode_costs)
                eval_average_episode_lengths = np.mean(eval_episode_lengths)

                eval_env_infos = {"eval_average_episode_rewards": eval_average_episode_rewards,
                                  "eval_average_episode_costs": eval_average_episode_costs,
                                  "eval_average_episode_lengths": eval_average_episode_lengths}
                for k, v in eval_env_infos.items():
                    self.writter.add_scalar(k, v, total_num_steps)
                with open(os.path.join(str(self.run_dir), "progress.csv"), "a", encoding="utf-8") as f:
                    if f.tell() == 0:
                        f.write("total_steps,eval_average_episode_rewards,eval_average_episode_costs,"
                                "eval_average_episode_lengths\n")
                    f.write(f"{total_num_steps},{eval_average_episode_rewards},{eval_average_episode_costs},"
                            f"{eval_average_episode_lengths}\n")
                print(f"[eval] steps={total_num_steps} reward={eval_average_episode_rewards:.2f} "
                      f"cost={eval_average_episode_costs:.2f} length={eval_average_episode_lengths:.1f}")
                break
