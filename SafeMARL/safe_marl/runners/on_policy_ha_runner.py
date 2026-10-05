import numpy as np
import torch
from safe_marl.runners.on_policy_base_runner import OnPolicyBaseRunner
from safe_marl.utils.trans_tools import _t2n


class OnPolicyHARunner(OnPolicyBaseRunner):
    def train(self):
        actor_train_infos = []
        T, N = self.algo_args["train"]["episode_length"], self.algo_args["train"]["n_rollout_threads"]
        factor = np.ones((T, N, 1), dtype=np.float32)
        advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
            self.critic_buffer.value_preds[:-1]
        )

        for agent_id in list(torch.randperm(self.num_agents).numpy()):
            buffer = self.actor_buffer[agent_id]
            buffer.update_factor(factor)
            flat = lambda arr: arr.reshape(-1, *arr.shape[2:])
            evaluate = lambda: self.actor[agent_id].evaluate_actions(
                flat(buffer.obs[:-1]), flat(buffer.actions), flat(buffer.active_masks[:-1])
            )
            old_actions_logprob, _, _ = evaluate()
            actor_train_infos.append(self.actor[agent_id].train(buffer, advantages.copy()))
            new_actions_logprob, _, _ = evaluate()
            factor = factor * _t2n(
                torch.prod(torch.exp(new_actions_logprob - old_actions_logprob), dim=-1).reshape(T, N, 1)
            )

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        return actor_train_infos, critic_train_info
