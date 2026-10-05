import torch
import torch.nn as nn
from safe_marl.utils.models_tools import huber_loss
from safe_marl.utils.envs_tools import check
from safe_marl.models.value_function_models.v_net import VNet


class VCritic:
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]
        self.critic = VNet(args, cent_obs_space, self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args["critic_lr"], eps=args["opti_eps"], weight_decay=args["weight_decay"]
        )

    def get_values(self, cent_obs):
        return self.critic(cent_obs)

    def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        value_normalizer.update(return_batch)
        error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = value_normalizer.normalize(return_batch) - values
        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)
        return torch.max(value_loss_original, value_loss_clipped).mean()

    def update(self, sample, value_normalizer):
        share_obs_batch, value_preds_batch, return_batch = sample
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        values = self.get_values(share_obs_batch)
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, value_normalizer)
        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer):
        train_info = {"value_loss": 0, "critic_grad_norm": 0}
        for _ in range(self.critic_epoch):
            for sample in critic_buffer.feed_forward_generator_critic(self.critic_num_mini_batch):
                value_loss, critic_grad_norm = self.update(sample, value_normalizer)
                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm
        num_updates = self.critic_epoch * self.critic_num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.critic.train()

    def prep_rollout(self):
        self.critic.eval()
