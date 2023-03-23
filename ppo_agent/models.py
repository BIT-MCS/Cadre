from ppo_agent.distributions import MixDist, CatDist, NormDist
from ppo_agent.utils import Counter
import torch.nn as nn
import torch
import torch.multiprocessing as mp
from carla_perception.Networks.danet import DANet
from ppo_agent.utils import init_normc_
from torch.nn import functional as F
from carla_perception.Config.auto_danet import danet_config


# import torchsnooper


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    else:
        print('None bias')
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 64, 16, 30)

def get_vae_output(model_cfg):
    vae_params_cfg = model_cfg['vae_params']
    measurement_dim = model_cfg['measurement_dim']
    vae_params = danet_config()

    if vae_params_cfg == "CoPM" or vae_params_cfg == "CoPM w/o att":
        obs_dim = 2 * vae_params.networks['autoencoder']['z_dims'] + measurement_dim
    else:
        obs_dim = vae_params.networks['autoencoder']['z_dims'] + measurement_dim
    return obs_dim, vae_params

def create_model(model_cfg, load_vae=False):
    obs_dim, vae_params = get_vae_output(model_cfg)
    if load_vae:
        vae_device = model_cfg['vae_device']
        if vae_device == -1:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda:" + str(vae_device))
        # self.lstm_input = self.vae_output
        # load vae model
        vae_model_path = vae_params.networks['autoencoder']['pretrained_path']
        pretrained_model = torch.load(vae_model_path, map_location=vae_device)

        network = DANet(vae_params.networks['autoencoder'])

        key = 'autoencoder'
        if key in pretrained_model.keys():
            if pretrained_model[key].keys() == network.state_dict().keys():
                print('==> network parameters in pre-trained file %s can strictly match' % (vae_model_path))
                network.load_state_dict(pretrained_model[key])
            else:
                pretrained_model_dict = list(pretrained_model[key].keys())
                network_keys = list(network.state_dict().keys())
                for _key in pretrained_model_dict:
                    if _key not in network_keys:
                        print(_key, ' does not exist in model config!')
                print('VAE model load fail in ', vae_model_path)

        vae_model = network
        vae_model.to(vae_device)
        if vae_params.pred_left_camera_seg:
            del vae_model.reverse_left_image
        if vae_params.pred_right_camera_seg:
            del vae_model.reverse_right_image
        if vae_params.pred_light_dist:
            del vae_model.reverse_lightDist
        if vae_params.pred_lidar:
            del vae_model.reverse_lidar
        if vae_params.pred_topdown_rgb:
            del vae_model.reverse_topdown_rgb
        if vae_params.pred_topdown_seg:
            del vae_model.reverse_topdown_seg
        vae_model.eval()
    else:
        vae_model = None
    device = model_cfg['device_num']
    use_lstm = model_cfg['use_lstm']


    if device == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(device))
    model_dict = {}

    command_num = model_cfg.command_num

    for _command in range(command_num):

        steer_output = model_cfg['num_output']['steer']
        ppo_model_steer = Model(obs_dim, steer_output)
        ppo_model_steer.to_device(device)
        ppo_model_steer.train()

        throttle_output = model_cfg['num_output']['throttle']
        ppo_model_throttle = Model(obs_dim, throttle_output)
        ppo_model_throttle.to_device(device)
        ppo_model_throttle.train()
        model_dict['steer_ppo_' + str(_command)] = ppo_model_steer
        model_dict['throttle_ppo_' + str(_command)] = ppo_model_throttle

        if use_lstm:
            for _command in range(command_num):
                lstm_model = LSTM(obs_dim, hid_size=obs_dim)
                lstm_model.to(device)
                lstm_model.train()

                model_dict['steer_lstm_' + str(_command)] = lstm_model
                lstm_model = LSTM(obs_dim, hid_size=obs_dim)
                lstm_model.to(device)
                lstm_model.train()
                model_dict['throttle_lstm_' + str(_command)] = lstm_model
    return vae_model, model_dict



class LSTM(nn.Module):
    def __init__(self, input_size, hid_size=128, num_layers=1):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTMCell(input_size, hid_size)
        nn.init.orthogonal_(self.rnn.weight_ih.data)
        nn.init.orthogonal_(self.rnn.weight_hh.data)
        self.rnn.bias_ih.data.fill_(0)
        self.rnn.bias_hh.data.fill_(0)

    def forward(self, x, hidden_state):
        if x.size(0) == hidden_state[0].size(0):
            # x: [N, -1]
            hidden_state = self.rnn(x, hidden_state)
            x = hidden_state[0]
        else:
            # x: [T * N, -1]
            N = hidden_state[0].size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            for i in range(T):
                hidden_state = self.rnn(x[i], hidden_state)
            x = hidden_state[0]
        return x, hidden_state

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


from ppo_agent.distributions import Categorical_1d


class Model(nn.Module):
    def __init__(self, num_input, num_output, trainable=True, hidsize=128):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.control = Categorical_1d(num_input, num_output)
        # critic
        self.critic = nn.Sequential(
            init_(nn.Linear(num_input, hidsize)),
            nn.ReLU(),
            init_(nn.Linear(hidsize, hidsize)),
            nn.ReLU(),
            init_(nn.Linear(hidsize, 1)),
        )
        if trainable:
            self.train()
        else:
            self.eval()
        self.trainable = trainable

    def act(self, obs_feature):
        with torch.no_grad():
            value = self.critic(obs_feature)
            self.control.forward(obs_feature)
            control = self.control.softmax_sample()
        return value.clone().detach(), control, obs_feature.clone().detach()

    def get_log_probs(self, action):
        action_log_probs = self.control.log_probs(action)
        return action_log_probs

    def to_device(self, device):
        self.critic.to(device)
        self.control.to_device(device)

    def get_value(self, obs_feature):
        value = self.critic(obs_feature)
        return value

    def evaluate_actions(self, obs_feature, action):
        """
        obs_feature: [N, *obs_dim]
        action: [N, 3]
        """
        value = self.critic(obs_feature)
        self.control.forward(obs_feature)
        action_log_probs = self.get_log_probs(action)
        entropy = self.control.entropy().unsqueeze(-1)
        return value, action_log_probs, entropy

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


class Shared_grad_buffers():
    def __init__(self, model_list, device):
        self.grads = {}
        self.counter = Counter()
        self.lock = mp.Lock()
        for model_name in model_list:
            model = model_list[model_name]
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.grads[model_name + '_' + name + '_grad'] = torch.zeros(p.size()).share_memory_().to(device)
        self.device = device

    def add_gradient(self, model_list):
        with self.lock:
            for model_name in model_list:
                model = model_list[model_name]
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.grads[model_name + '_' + name + '_grad'] += p.grad.data.to(self.device)
                        # self.grads[model_name + '_' + name + '_grad'] += p.grad.data
            self.counter.increment()

    def average_gradient(self):
        counter_num = self.counter.get()
        for name, grad in self.grads.items():
            self.grads[name] /= counter_num

    def print_gradient(self):
        # for grad in self.grads:
        #     if 'base.critic' in grad:
        #         # if grad == 'fc1.weight_grad':
        #         print(grad, '  ', self.grads[grad].mean())
        for name, grad in self.grads.items():
            # if 'critic' in name:
            print(name, self.grads[name].mean())

    def reset(self):
        self.counter.reset()
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)


class SmallCNN(nn.Module):
    def __init__(self, channel, frame, z_dims):
        super(SmallCNN, self).__init__()
        self.preprocess = nn.Sequential(
            # image: [B, 4, 144, 256]
            nn.Conv2d(in_channels=channel * frame, out_channels=64, kernel_size=(4, 4), stride=2),
            # image: [B, 64, 71, 127]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2),
            # image: [B, 32, 34, 62]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=2),
            # output: [B, 64, 16, 30]
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=32 * 16 * 30, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=z_dims),
            nn.ReLU(),
        )
        self.frame = frame

    def forward(self, topdown):
        topdown_feature = self.preprocess(topdown)
        state_feature = topdown_feature
        return state_feature

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)
