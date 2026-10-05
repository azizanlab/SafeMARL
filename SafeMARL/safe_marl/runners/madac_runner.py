from safe_marl.algorithms.madac import MADAC
from safe_marl.runners.off_policy_base_runner import OffPolicyBaseRunner


class MADACRunner(OffPolicyBaseRunner):
    def _build(self):
        self.learner = MADAC(
            {**self.algo_args["model"], **self.algo_args["algo"]},
            self.envs.observation_space,
            self.envs.action_space,
            self.envs.share_observation_space[0],
            self.num_agents,
            self.device,
        )

    def get_actions(self, obs, share_obs, stochastic=True):
        return self.learner.act(obs, share_obs, stochastic=stochastic)

    def train(self):
        return self.learner.update(self.buffer.sample())

    def save(self):
        self.learner.save(self.save_dir)
