import numpy as np
import torch

from safe_marl.algorithms.actors.hasac import HASAC
from safe_marl.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic
from safe_marl.runners.off_policy_base_runner import OffPolicyBaseRunner
from safe_marl.utils.trans_tools import _t2n


class OffPolicyHARunner(OffPolicyBaseRunner):
    def _build(self):
        algo_args = self.algo_args
        self.actor = [
            HASAC(
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
                device=self.device,
            )
            for agent_id in range(self.num_agents)
        ]
        self.critic = SoftTwinContinuousQCritic(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            self.envs.share_observation_space[0],
            self.envs.action_space,
            self.num_agents,
            device=self.device,
        )
        self.alpha = [torch.tensor(algo_args["algo"]["alpha"], device=self.device)] * self.num_agents

    @torch.no_grad()
    def get_actions(self, obs, share_obs, stochastic=True):
        actions = [
            _t2n(self.actor[agent_id].get_actions(obs[:, agent_id], stochastic=stochastic))
            for agent_id in range(self.num_agents)
        ]
        return np.array(actions).transpose(1, 0, 2)

    def train(self):
        self.total_it += 1
        batch = self.buffer.sample()
        stats = {}

        self.critic.turn_on_grad()
        next_actions, next_logp_actions = [], []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                action, logp = self.actor[agent_id].get_actions_with_logprobs(batch["next_obs"][agent_id])
                next_actions.append(action)
                next_logp_actions.append(logp)
        stats["critic_loss"] = self.critic.train(
            batch["share_obs"], batch["actions"], batch["rewards"], batch["term"], batch["next_share_obs"],
            next_actions, next_logp_actions, batch["gamma_n"], sum(self.alpha) / self.num_agents,
        )
        self.critic.turn_off_grad()

        actions, logp_actions = [], []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                action, logp = self.actor[agent_id].get_actions_with_logprobs(batch["obs"][agent_id])
                actions.append(action)
                logp_actions.append(logp)
        for agent_id in list(np.random.permutation(self.num_agents)):
            self.actor[agent_id].turn_on_grad()
            actions[agent_id], logp_actions[agent_id] = self.actor[agent_id].get_actions_with_logprobs(
                batch["obs"][agent_id]
            )
            value_pred = self.critic.get_values(batch["share_obs"], torch.cat(actions, dim=-1))
            actor_loss = -torch.mean(value_pred - self.alpha[agent_id] * logp_actions[agent_id])
            self.actor[agent_id].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[agent_id].actor_optimizer.step()
            self.actor[agent_id].turn_off_grad()
            stats[f"actor_loss/agent{agent_id}"] = actor_loss.item()

            with torch.no_grad():
                actions[agent_id], _ = self.actor[agent_id].get_actions_with_logprobs(batch["obs"][agent_id])

        self.critic.soft_update()
        return stats

    def save(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)
