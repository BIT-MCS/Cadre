import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchsnooper
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, c, h, w):
        super(Reshape, self).__init__()
        self.c = c
        self.h = h
        self.w = w

    def forward(self, x):
        # return x.view(-1, 64, 18, 32)
        return x.view(-1, self.c, self.h, self.w)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self, net_params):
        super(BetaVAE, self).__init__()

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.light_classes_num = net_params['light_classes_num']

        self.z_dims = net_params['z_dims']
        self.beta = net_params['beta']
        # self.gamma = net_params['gamma']
        # self.loss_type = net_params['loss_type']
        # self.max_capacity = net_params['max_capacity']
        # self.C_stop_iter = net_params['Capacity_max_iter']
        # self.C_max = torch.Tensor([self.max_capacity])

        self.pred_light = net_params['pred_light']
        self.pred_lidar = net_params['pred_lidar']
        self.pred_topdown_rgb = net_params['pred_topdown_rgb']
        self.pred_topdown_seg = net_params['pred_topdown_seg']

        modules = []
        # self.hidden_dims = [32, 64, 128, 256, 512]
        # self.hidden_dims = [64, 128, 256, 512, 1024]
        self.hidden_dims = [64, 128, 256, 512]
        # self.hidden_dims = [32, 64, 128, 256]
        # self.hidden_dims = [128, 256, 512, 1024]

        # Build Encoder
        # h: [144, 72, 36, 18, 9, 5]
        # w: [256, 128, 64, 32, 16, 8]
        input_channel = self.input_channel
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channel, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            input_channel = h_dim

        self.encoder = nn.Sequential(*modules)

        output_h = 9
        output_w = 16
    
        self.image_mu = nn.Sequential(
            nn.Linear(in_features=self.hidden_dims[-1] * output_h * output_w, out_features=self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dims[-1], out_features=self.z_dims),
        )

        self.image_logvar = nn.Sequential(
            nn.Linear(in_features=self.hidden_dims[-1] * output_h * output_w, out_features=self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dims[-1], out_features=self.z_dims),
        )

        # self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.z_dims)
        # self.fc_var = nn.Linear(hidden_dims[-1]*4, self.z_dims)


        # Build Decoder
        # self.decoder_input = nn.Linear(self.z_dims, hidden_dims[-1] * 4)

        self.reverse_feature = nn.Sequential(
            nn.Linear(in_features=self.z_dims, out_features=self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dims[-1], out_features=self.hidden_dims[-1] * output_h * output_w),
            # nn.LeakyReLU(),
            Reshape(self.hidden_dims[-1], output_h, output_w),
        )

        # h: [5, 9, 18, 36, 72, 144]
        # w: [8, 16, 32, 64, 128, 256]
        self.reverse_image = self.build_reverse_module(output=3)

        if self.pred_lidar:
            self.reverse_lidar = self.build_reverse_module(output=3)
        
        if self.pred_topdown_rgb:
            self.reverse_topdown_rgb = self.build_reverse_module(output=3)
        
        if self.pred_topdown_seg:
            self.reverse_topdown_seg = self.build_reverse_module(output=1)

        reverse_feature_size = self.hidden_dims[-1] * output_h * output_w

        # if self.pred_curSpeed:
        #     self.reverse_curSpeed = nn.Sequential(
        #         Flatten(),
        #         nn.Linear(in_features=reverse_feature_size, out_features=256),
        #         nn.LeakyReLU(),
        #         nn.Linear(in_features=256, out_features=64),
        #         nn.LeakyReLU(),
        #         nn.Linear(in_features=64, out_features=1),
        #     )

        # if self.pred_tarSpeed:
        #     self.reverse_tarSpeed = nn.Sequential(
        #         Flatten(),
        #         nn.Linear(in_features=reverse_feature_size, out_features=256),
        #         nn.LeakyReLU(),
        #         nn.Linear(in_features=256, out_features=64),
        #         nn.LeakyReLU(),
        #         nn.Linear(in_features=64, out_features=1),
        #     )

        if self.pred_light:
            self.reverse_lightState = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=reverse_feature_size, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.LeakyReLU(),
                nn.Linear(in_features=64, out_features=self.light_classes_num),
            )

            self.reverse_lightDist = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=reverse_feature_size, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.LeakyReLU(),
                nn.Linear(in_features=64, out_features=1),
        )

    def build_reverse_module(self, output=3):
        modules = []
        hidden_dims = self.hidden_dims[::-1]
        # h: [5, 9, 18, 36, 72, 144]
        # w: [8, 16, 32, 64, 128, 256]
        for i in range(len(hidden_dims) - 1):
            # if i == 0:
            #     cur_output_padding = (0, 1)
            # else:
            #     cur_output_padding = 1
            cur_output_padding = 1
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=cur_output_padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                    output,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                nn.Sigmoid())
                # nn.Tanh())
        )

        cur_module = nn.Sequential(*modules)
        return cur_module

    def encode(self, input):
        # encode_input = torch.cat((img, lidar), dim=1)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)
        mu = self.image_mu(result)
        log_var = self.image_logvar(result)

        return [mu, log_var]

    def decode(self, z):
        reverse_lightState = None
        reverse_lightDist = None
        reverse_lidar = None
        reverse_topdown = None
        
        reverse_feature = self.reverse_feature(z)
        reverse_image = self.reverse_image(reverse_feature)
        # if self.pred_curSpeed:
        #     reverse_curSpeed = self.reverse_curSpeed(reverse_feature)
        # if self.pred_tarSpeed:
        #     reverse_tarSpeed = self.reverse_tarSpeed(reverse_feature)
        if self.pred_light:
            reverse_lightState = self.reverse_lightState(reverse_feature)
            reverse_lightDist = self.reverse_lightDist(reverse_feature)

        if self.pred_lidar:
            reverse_lidar = self.reverse_lidar(reverse_feature)
        
        if self.pred_topdown_rgb:
            reverse_topdown = self.reverse_topdown_rgb(reverse_feature)
        
        if self.pred_topdown_seg:
            reverse_topdown = self.reverse_topdown_seg(reverse_feature)

        return reverse_lightState, reverse_lightDist, \
                reverse_image, \
                reverse_lidar, \
                reverse_topdown 


    def reparameterize(self, mu, logvar, training=True):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, input, training=True):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar, training)

        # return  [self.decode(z), input, mu, log_var]

        z_decode = self.decode(z)
        reverse_lightState, reverse_lightDist, \
        reverse_image, \
        reverse_lidar, \
        reverse_topdown = z_decode

        img_pred = reverse_image
        lidar_pred = reverse_lidar
        topdown_pred = reverse_topdown
        lightState_pred = reverse_lightState
        lightDist_pred = reverse_lightDist

        return img_pred, lidar_pred, topdown_pred, \
            lightState_pred, lightDist_pred, mu, logvar

    def test_forward(self, input, training=False):
        results = self.forward(input, training)
        return results

    def get_latent_feature(self, input, traing=False):
        mu, logvar = self.encode(input)

        return mu

    def loss_function(self,
                      *args,
                      **kwargs):
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.z_dims)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def get_model(net_params):
    return BetaVAE(net_params)