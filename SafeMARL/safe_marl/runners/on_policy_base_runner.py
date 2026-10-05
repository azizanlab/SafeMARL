import os

import numpy as np
import setproctitle
import torch

from safe_marl.algorithms.actors import ALGO_REGISTRY
from safe_marl.algorithms.critics.v_critic import VCritic
from safe_marl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from safe_marl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from safe_marl.common.mujoco_logger import MAMuJoCoLogger
from safe_marl.common.valuenorm import ValueNorm
from safe_marl.utils.configs_tools import init_dir, save_config
from safe_marl.utils.envs_tools import get_num_agents, make_eval_env, make_train_env, set_seed
from safe_marl.utils.models_tools import init_device
from safe_marl.utils.trans_tools import _t2n


class OnPolicyBaseRunner:
    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.n_threads = algo_args["train"]["n_rollout_threads"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"], env_args, args["algo"], args["exp_name"], algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        setproctitle.setproctitle(f"{args['algo']}-{args['env']}-{args['exp_name']}")

        self.envs = make_train_env(args["env"], algo_args["seed"]["seed"], self.n_threads, env_args)
        self.eval_envs = (
            make_eval_env(args["env"], algo_args["seed"]["seed"], algo_args["eval"]["n_eval_rollout_threads"], env_args)
            if algo_args["eval"]["use_eval"]
            else None
        )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        print("observation_space:", self.envs.observation_space)
        print("share_observation_space:", self.envs.share_observation_space)
        print("action_space:", self.envs.action_space)

        actor_args = {**algo_args["model"], **algo_args["algo"]}
        self.actor = [
            ALGO_REGISTRY[args["algo"]](
                actor_args, self.envs.observation_space[a], self.envs.action_space[a], device=self.device
            )
            for a in range(self.num_agents)
        ]
        self.actor_buffer = [
            OnPolicyActorBuffer(
                {**algo_args["train"], **algo_args["model"]}, self.envs.observation_space[a], self.envs.action_space[a]
            )
            for a in range(self.num_agents)
        ]
        share_observation_space = self.envs.share_observation_space[0]
        self.critic = VCritic({**algo_args["model"], **algo_args["algo"]}, share_observation_space, device=self.device)
        self.critic_buffer = OnPolicyCriticBufferEP(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]}, share_observation_space
        )
        self.value_normalizer = ValueNorm(1, device=self.device)
        self.logger = MAMuJoCoLogger(args, algo_args, env_args, self.num_agents, self.writter, self.run_dir)

    def run(self):
        print("start running")
        self.warmup()
        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.n_threads
        )
        self.logger.init(episodes)

        for episode in range(0, episodes + 1):
            self.logger.episode_init(episode)

            self.prep_rollout()
            for step in range(self.algo_args["train"]["episode_length"]):
                values, actions, action_log_probs = self.collect(step)
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)
                self.logger.per_step((obs, share_obs, rewards, costs, dones, infos, values, actions, action_log_probs))
                self.insert(obs, share_obs, rewards, dones, infos, values, actions, action_log_probs)

            self.compute()
            self.prep_training()
            actor_train_infos, critic_train_info = self.train()

            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(actor_train_infos, critic_train_info, self.actor_buffer, self.critic_buffer)
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.eval_envs is not None:
                    self.prep_rollout()
                    self.eval()
                self._save_checkpoint(episode * self.algo_args["train"]["episode_length"] * self.n_threads)
            self.after_update()

    def warmup(self):
        obs, share_obs, _ = self.envs.reset()
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
        self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()

    @torch.no_grad()
    def collect(self, step):
        actions, action_log_probs = [], []
        for agent_id in range(self.num_agents):
            action, action_log_prob = self.actor[agent_id].get_actions(self.actor_buffer[agent_id].obs[step])
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))
        value = self.critic.get_values(self.critic_buffer.share_obs[step])
        return _t2n(value), np.array(actions).transpose(1, 0, 2), np.array(action_log_probs).transpose(1, 0, 2)

    def insert(self, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs):
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = 0.0
        active_masks = np.ones((self.n_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones] = 0.0
        active_masks[dones_env] = 1.0
        bad_masks = np.array([[0.0] if info[0].get("bad_transition", False) else [1.0] for info in infos])

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id], actions[:, agent_id], action_log_probs[:, agent_id], active_masks[:, agent_id]
            )
        self.critic_buffer.insert(share_obs[:, 0], values, rewards[:, 0], masks[:, 0], bad_masks)

    @torch.no_grad()
    def compute(self):
        next_value = self.critic.get_values(self.critic_buffer.share_obs[-1])
        self.critic_buffer.compute_returns(_t2n(next_value), self.value_normalizer)

    def train(self):
        raise NotImplementedError

    def after_update(self):
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        eval_obs, _, _ = self.eval_envs.reset()
        while True:
            actions = [
                _t2n(self.actor[agent_id].act(eval_obs[:, agent_id], deterministic=True))
                for agent_id in range(self.num_agents)
            ]
            eval_obs, eval_share_obs, eval_rewards, eval_costs, eval_dones, eval_infos, _ = self.eval_envs.step(
                np.array(actions).transpose(1, 0, 2)
            )
            self.logger.eval_per_step((eval_obs, eval_share_obs, eval_rewards, eval_costs, eval_dones, eval_infos))
            for eval_i in np.where(np.all(eval_dones, axis=1))[0]:
                eval_episode += 1
                self.logger.eval_thread_done(eval_i)
            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                break

    def prep_rollout(self):
        for actor in self.actor:
            actor.prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        for actor in self.actor:
            actor.prep_training()
        self.critic.prep_training()

    def save(self):
        for agent_id in range(self.num_agents):
            torch.save(self.actor[agent_id].actor.state_dict(), f"{self.save_dir}/actor_agent{agent_id}.pt")
        torch.save(self.critic.critic.state_dict(), f"{self.save_dir}/critic_agent.pt")
        torch.save(self.value_normalizer.state_dict(), f"{self.save_dir}/value_normalizer.pt")

    def _save_checkpoint(self, total_steps):
        self.save()
        if self.algo_args["train"].get("keep_checkpoints"):
            latest = self.save_dir
            self.save_dir = os.path.join(str(latest), f"step_{total_steps}")
            os.makedirs(self.save_dir, exist_ok=True)
            self.save()
            self.save_dir = latest

    def close(self):
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()
        self.writter.export_scalars_to_json(os.path.join(self.log_dir, "summary.json"))
        self.writter.close()
        self.logger.close()
