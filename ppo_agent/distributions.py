import torch
import torch.nn as nn
import torch.distributions
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from ppo_agent.utils import AddBias, init, init_normc_
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
# import torchsnooper

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class Categorical_1d(nn.Module):
    def __init__(self, num_inputs, num_outputs, name="none"):
        super(Categorical_1d, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = nn.Sequential(
            init_(nn.Linear(num_inputs, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, num_outputs))
        )
        self.dis_cat = None
        self.logits = 0
        self.sampler = Categorical
        self.weighted_sampler = WeightedRandomSampler
        # self.action_masks = self.construct_mask(num_outputs)
        self.action_masks = self.construct_mask1(num_outputs)
        self.name = name
        self.train()

    def construct_mask(self, bins):
        a = torch.zeros([bins, bins])
        for i in range(bins):
            for j in range(bins):
                if i + j <= bins - 1:
                    a[i, j] = 1.0
        return a

    def construct_mask1(self, bins):
        a = torch.zeros([bins, bins])
        for i in range(bins):
            for j in range(bins):
                if i >= j:
                    a[i, j] = 1.0
        return a

    def forward(self, x):
        x = self.linear(x)
        # # todo: ordinal policy
        # sigmoid_logits = torch.sigmoid(x)
        # batch_num = sigmoid_logits.size(0)
        # num_outputs = sigmoid_logits.size(1)
        #
        # sigmoid_logits = sigmoid_logits.reshape(batch_num, -1, num_outputs).unsqueeze(-1)
        # sigmoid_logits = sigmoid_logits.repeat(1, 1, 1, num_outputs)
        #
        # logits = torch.log(sigmoid_logits + 1e-8) * self.action_masks + torch.log(1 - sigmoid_logits + 1e-8) * (
        #         1 - self.action_masks)
        # x = torch.sum(logits, dim=-1)
        # x = x.reshape(batch_num, -1)
        self.dis_cat = Categorical(logits=x)
        self.logits = self.dis_cat.logits
        self.probs = self.dis_cat.probs
        return self.dis_cat

    def sample(self):
        return self.dis_cat.sample()

    def mode(self):
        return torch.argmax(self.probs)

    def gumbel_softmax_sample(self, tau):
        dist = F.gumbel_softmax(self.logits, tau=tau, hard=False)
        action = torch.tensor(list(self.weighted_sampler(dist, 1, replacement=False)))
        return action.squeeze(-1)

    def softmax_sample(self):
        probs = F.softmax(self.logits)
        action = Categorical(probs=probs).sample()
        return action

    def log_probs(self, action):
        return self.dis_cat.log_prob(action.squeeze(-1)).unsqueeze(-1)

    def entropy(self):
        return self.dis_cat.entropy()

    def to_device(self, device):
        self.linear.to(device)
        self.action_masks = self.action_masks.to(device)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = nn.Sequential(
            init_(nn.Linear(num_inputs, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 1)),
            nn.Sigmoid()
        )
        self.logstd = AddBias(torch.zeros(1))
        self.dist = None

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)  # TODO new
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.to(self.device)

        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)  # TODO new
        return FixedNormal(action_mean, action_logstd.exp())

    def to_device(self, device):
        self.device = device
        self.fc_mean.to(device)
        self.logstd.to(device)


class DiagGaussianTan(nn.Module):
    def __init__(self, num_inputs):
        super(DiagGaussianTan, self).__init__()
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = nn.Sequential(
            init_(nn.Linear(num_inputs, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 1)),
            nn.Tanh()
        )
        self.logstd = AddBias(torch.zeros(1))
        self.dist = None

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)  # TODO new
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.to(self.device)

        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)  # TODO new
        return FixedNormal(action_mean, action_logstd.exp())

    def to_device(self, device):
        self.device = device
        self.fc_mean.to(device)
        self.logstd.to(device)


class MixDist(nn.Module):
    def __init__(self, num_inputs, device):
        super(MixDist, self).__init__()
        self.device = device
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))
        # todo: hidsize 64 -> 32
        hidsize = 32
        self.steer_mu = nn.Sequential(
            init_(nn.Linear(num_inputs, hidsize)),
            nn.ReLU(),
            init_(nn.Linear(hidsize, 1)),
            nn.Tanh()
        ).to(device)
        self.steer_bias = AddBias(torch.zeros(1).to(device))

        self.speed_mu = nn.Sequential(
            init_(nn.Linear(num_inputs, hidsize)),
            nn.ReLU(),
            init_(nn.Linear(hidsize, 1)),
            nn.Sigmoid()
        ).to(device)
        self.speed_bias = AddBias(torch.zeros(1).to(device))

        # self.brake_logits = nn.Sequential(
        #     init_(nn.Linear(num_inputs, 2)),
        # ).to(device)

    def forward(self, x):
        self.speed_mean = self.speed_mu(x)
        # self.speed_dist = Categorical(logits=self.speed_logits)
        speed_zeros = torch.zeros(self.speed_mean.size()).to(self.device)
        speed_logstd = F.tanh(self.speed_bias(speed_zeros))
        self.speed_dist = Normal(self.speed_mean, speed_logstd.exp())

        self.steer_mean = self.steer_mu(x)
        steer_zeros = torch.zeros(self.steer_mean.size()).to(self.device)
        steer_logstd = F.tanh(self.steer_bias(steer_zeros))
        self.steer_dist = Normal(self.steer_mean, steer_logstd.exp())

        # self.brake_dist = Categorical(logits=self.brake_logits(x))

    def sample(self):
        steer = self.steer_dist.sample().clamp(-1, 1)
        speed = self.speed_dist.sample().clamp(0, 1)
        # brake = self.brake_dist.sample().unsqueeze(-1)
        # return [steer.clone().detach(), speed.clone().detach()]
        return [steer.clone().detach(), speed.clone().detach()]

    def mode(self):
        steer = self.steer_mean.clamp(-1, 1)
        # speed = torch.argmax(self.speed_logits).unsqueeze(-1)
        speed = self.speed_mean.clamp(0, 1)
        return [steer.clone().detach(), speed.clone().detach()]

    def log_probs(self, action):
        steer = action[0]
        steer_log_probs = self.steer_dist.log_prob(steer)

        speed = action[1]
        # speed_log_probs = self.speed_dist.log_prob(speed.squeeze(-1)).unsqueeze(-1)
        speed_log_probs = self.speed_dist.log_prob(speed)
        # brake_log_probs = self.brake_dist.log_prob(brake.squeeze(-1)).unsqueeze(-1)
        # print('log_probs', steer_log_probs.size(), speed_log_probs.size())
        # log_probs = 0.5 * steer_log_probs + 0.5 * speed_log_probs
        log_probs = speed_log_probs
        # log_probs = steer_log_probs
        return log_probs

    def entropy(self):
        steer_entropy = self.steer_dist.entropy()
        speed_entropy = self.speed_dist.entropy()
        # brake_entropy = self.brake_dist.entropy().unsqueeze(-1)
        # entropy = steer_entropy
        # print('entropy:', steer_entropy.size(), speed_entropy.size())
        entropy = speed_entropy
        # entropy = 0.5 * steer_entropy + 0.5 * speed_entropy
        return entropy


class NormDist(nn.Module):
    def __init__(self, num_inputs, constant_speed, trainable):
        super(NormDist, self).__init__()
        # self.steer = DiagGaussian(num_inputs)
        self.steer = DiagGaussianTan(num_inputs)
        self.steer_dist = None
        if not constant_speed:
            self.throttle = DiagGaussian(num_inputs)
            self.throttle_dist = None
        self.constant_speed = constant_speed
        if trainable:
            self.train()
        else:
            self.eval()

    def forward(self, feature):
        self.steer_dist = self.steer(feature)
        if not self.constant_speed:
            self.throttle_dist = self.throttle(feature)

    def sample(self, tau):
        steer = self.steer_dist.sample().clamp(-1, 1)
        action = [steer]
        if not self.constant_speed:
            throttle = self.throttle_dist.sample().clamp(0, 0.75)
            action.append(throttle)
        return action

    def log_probs(self, action):
        steer_log_probs = self.steer_dist.log_probs(action[0])
        probs = [self.steer_dist.mean]

        if not self.constant_speed:
            throttle_log_probs = self.throttle_dist.log_probs(action[1])
            log_probs = steer_log_probs + throttle_log_probs
            probs.append(self.throttle_dist.mean)
        else:
            log_probs = steer_log_probs
        return log_probs, probs

    def entropy(self):
        steer_entropy = self.steer_dist.entropy()
        if not self.constant_speed:
            throttle_entropy = self.throttle_dist.entropy()
            entropy = steer_entropy + throttle_entropy
        else:
            entropy = steer_entropy
        return entropy

    def to_device(self, device):
        self.steer.to_device(device)
        if not self.constant_speed:
            self.throttle.to_device(device)


class CatDist(nn.Module):
    def __init__(self, num_inputs, num_output, ratio, trainable):
        # def __init__(self, num_inputs, num_output, constant_speed, trainable):
        super(CatDist, self).__init__()
        self.dist = Categorical_1d(num_inputs, num_output)
        self.ratio = ratio

        if constant_speed:
            self.steer_weight = 1.0
            self.throttle_weight = 0.0
        else:
            self.steer_weight = 1 / (self.ratio + 1)
            self.throttle_weight = self.ratio / (self.ratio + 1)
        if trainable:
            self.train()
        else:
            self.eval()

    def forward(self, feature):
        return self.dist(feature)

    def sample(self, tau):
        control = self.steer_dist.gumbel_softmax_sample(tau)
        return control.clone().detach()

    def softmax_sample(self, tau):
        control = self.dist.softmax_sample(tau)
        return control.clone().detach()

    def mode(self):
        control = self.dist.mode()
        return [steer.clone().detach()]

    def log_probs(self, action):
        steer_log_probs = self.steer_dist.log_probs(action[0])
        probs = [self.steer_dist.probs]
        if not self.constant_speed:
            throttle_log_probs = self.throttle_dist.log_probs(action[1])
            log_probs = self.steer_weight * steer_log_probs + self.throttle_weight * throttle_log_probs
            self.ratio = torch.mean(steer_log_probs / throttle_log_probs).item()
            probs.append(self.throttle_dist.probs)
        else:
            log_probs = steer_log_probs

        return log_probs, probs

    def update_ratio(self):
        self.steer_weight = 1 / (self.ratio + 1)
        self.throttle_weight = self.ratio / (self.ratio + 1)

    def entropy(self):
        steer_entropy = self.steer_dist.entropy()
        if not self.constant_speed:
            throttle_entropy = self.throttle_dist.entropy()
            entropy = self.steer_weight * steer_entropy + self.throttle_weight * throttle_entropy
            # entropy = 0.5 * steer_entropy + 0.5 * throttle_entropy
        else:
            entropy = steer_entropy
        return entropy

    def to_device(self, device):
        self.steer_dist.to_device(device)
        if not self.constant_speed:
            self.throttle_dist.to_device(device)
