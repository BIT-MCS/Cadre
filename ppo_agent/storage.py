import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, mini_batch_num, feature_dims, seq_length, hidden_size, use_gae, gamma, tau):
        self.mini_batch_num = mini_batch_num
        self.command = torch.zeros((num_steps + 1, 1), dtype=torch.int)

        self.obs = torch.zeros(num_steps + 1, seq_length, feature_dims)

        self.z_dims = feature_dims

        self.rewards = torch.zeros(num_steps + 1, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps + 1, 1)

        self.action = torch.zeros((num_steps + 1, 1), dtype=torch.long)

        self.seq_length = seq_length
        self.hn = torch.zeros(num_steps + 1, hidden_size)
        self.cn = torch.zeros(num_steps + 1, hidden_size)
        self.hid_size = hidden_size

        self.masks = torch.zeros(num_steps + 1, 1)
        self.num_steps = num_steps
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.step = 0

    def to(self, device):
        self.command = self.command.to(device)
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action = self.action.to(device)
        self.masks = self.masks.to(device)
        self.hn = self.hn.to(device)
        self.cn = self.cn.to(device)

    def insert(self, obs, action, action_log_probs, value_preds, rewards, masks, hidden_state, command):
        self.action[self.step].copy_(action.squeeze())
        self.action_log_probs[self.step].copy_(action_log_probs.squeeze())
        self.value_preds[self.step].copy_(value_preds.squeeze())
        self.rewards[self.step].copy_(rewards.squeeze())
        self.obs[self.step].copy_(obs.squeeze())
        if hidden_state is not None and self.step < self.num_steps:
            hn, cn = hidden_state
            self.hn[self.step + 1].copy_(hn.clone().squeeze())
            self.cn[self.step + 1].copy_(cn.clone().squeeze())
        self.masks[self.step].copy_(masks.squeeze())
        self.command[self.step] = command
        self.step = self.step + 1
        self.step = self.step % (self.num_steps + 1)

    def after_update(self, hidden_state):
        # todo: debug what about obs
        self.step = 0
        if hidden_state is not None:
            hn, cn = hidden_state
            self.hn[0].copy_(hn.squeeze())
            self.cn[0].copy_(cn.squeeze())

    def compute_returns(self, next_value):
        if self.use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step] - \
                        self.value_preds[step]
                gae = delta + self.gamma * self.tau * self.masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            length = self.rewards.size(0)
            for step in range(length):
                gae = 0
                for index in reversed(range(self.step, min(self.step + 10, length))):
                    delta = self.rewards[index] + self.gamma * self.value_preds[index + 1] * self.masks[index] - \
                            self.value_preds[index]
                    gae = delta + self.gamma * self.tau * self.masks[index] * gae
                self.returns[step] = gae + self.value_preds[step]

    def get_last(self):
        obs_batch = self.obs[-1]
        command = self.command[-1].item()
        return obs_batch, command

    def feed_forward_generator(self, advantages):
        mini_batch_size = self.num_steps // self.mini_batch_num
        sampler = BatchSampler(SubsetRandomSampler(range(0, self.num_steps)),
                               mini_batch_size,
                               drop_last=False)
        for indices in sampler:

            obs_batch = self.obs[indices]
            # obs_batch: from [batch_size, sequence_length, feature_dim] to [sequence_length, batch_size, feature_dim]
            obs_batch = obs_batch.permute(1, 0, 2)
            # obs_batch = torch.cat(obs_batch)
            obs_batch = obs_batch.reshape(-1, obs_batch.size(-1))

            hn_batch = [self.hn[indices]]
            cn_batch = [self.cn[indices]]
            hn_batch = torch.cat(hn_batch, dim=0)
            cn_batch = torch.cat(cn_batch, dim=0)

            action_batch = self.action[indices]
            value_pred_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            action_log_probs = self.action_log_probs[indices]
            advantages_batch = advantages[indices]
            masks_batch = self.masks[indices]
            command_batch = self.command[indices]
            hidden_state = [hn_batch, cn_batch]
            yield obs_batch, action_batch, value_pred_batch, return_batch, masks_batch, \
                  action_log_probs, advantages_batch, hidden_state, command_batch
