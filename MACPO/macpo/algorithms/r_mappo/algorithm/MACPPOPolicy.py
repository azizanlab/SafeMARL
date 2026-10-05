import torch
from macpo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic


class MACPPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.cost_critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps,
                                               weight_decay=self.weight_decay)

    def get_actions(self, cent_obs, obs, deterministic=False):
        actions, action_log_probs = self.actor(obs, deterministic)
        values = self.critic(cent_obs)
        cost_preds = self.cost_critic(cent_obs)
        return values, actions, action_log_probs, cost_preds

    def get_values(self, cent_obs):
        return self.critic(cent_obs)

    def get_cost_values(self, cent_obs):
        return self.cost_critic(cent_obs)

    def evaluate_actions(self, cent_obs, obs, action, active_masks=None):
        action_log_probs, dist_entropy, action_mu, action_std = self.actor.evaluate_actions(obs, action, active_masks)
        values = self.critic(cent_obs)
        cost_values = self.cost_critic(cent_obs)
        values = self.critic(cent_obs)
        return values, action_log_probs, dist_entropy, cost_values, action_mu, action_std

    def act(self, obs, deterministic=False):
        actions, _ = self.actor(obs, deterministic)
        return actions
