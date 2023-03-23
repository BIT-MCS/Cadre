import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchsnooper
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def forward(self, x):
        # return x.view(-1, 64, 18, 32)
        return x.view(-1, 64, 9, 16)


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

class vanillaVAE(nn.Module):
    def __init__(self, net_params):
        super(vanillaVAE, self).__init__()

        self.z_dims = net_params['z_dims']
        self.no_of_samples = net_params['no_of_samples']

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.light_classes_num = net_params['light_classes_num']

        self.encoder = nn.Sequential(
            # F.interpolate([1, 144, 256]),
            nn.Conv2d(in_channels=self.input_channel, out_channels=32, kernel_size=(5, 5), stride=2, padding=(5, 5)),
            # image: [B, 32, 75, 131]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=(3, 3)),
            # image: [B, 64, 40, 68]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=(3, 3)),
            # image: [B, 128, 22, 36]
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=(3, 3)),
            # # image: [B, 64, 13, 20]
            nn.LeakyReLU(),
            Flatten()
        )

        # self.preprocess_lidar = nn.Sequential(
        #     # F.interpolate([1, 144, 256]),
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=(5, 5)),
        #     # image: [B, 32, 75, 131]
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=(3, 3)),
        #     # image: [B, 64, 40, 68]
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=(3, 3)),
        #     # image: [B, 128, 22, 36]
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=(3, 3)),
        #     # # image: [B, 64, 13, 20]
        #     # nn.LeakyReLU(),
        #     Flatten()
        # )

        output_h = 13
        output_w = 20
        # output_h = 22
        # output_w = 36

        self.image_mu = nn.Sequential(
            nn.Linear(in_features=64 * output_h * output_w, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=self.z_dims),
        )

        self.image_logvar = nn.Sequential(
            nn.Linear(in_features=64 * output_h * output_w, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=self.z_dims),
        )
        
        # For decoder
        self.reverse_feature = nn.Sequential(
            nn.Linear(in_features=self.z_dims, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=64 * 9 * 16),
            # nn.Linear(in_features=1024, out_features=64 * 18 * 32),
            nn.LeakyReLU(),
            Reshape(),
        )
        # self.reverse_topdown = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=6, kernel_size=4, stride=2),
        # )

        self.reverse_lidar_mu = nn.Sequential(
            # output: [B, 128, 9, 16]
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 128, 18, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 64, 36, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 32, 72, 128]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 3, 144, 256]
            nn.Sigmoid()
        )

        self.reverse_lidar_logvar = nn.Sequential(
            # output: [B, 128, 9, 16]
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 128, 18, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 64, 36, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 32, 72, 128]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 2, 144, 256]
            nn.Sigmoid()
        )

        self.reverse_image_mu = nn.Sequential(
            # output: [B, 128, 9, 16]
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 128, 18, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 64, 36, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 32, 72, 128]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 3, 144, 256]
            nn.Sigmoid()
        )

        self.reverse_image_logvar = nn.Sequential(
            # output: [B, 128, 9, 16]
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 128, 18, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 64, 36, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 32, 72, 128]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 2, 144, 256]
            nn.Sigmoid()
        )

        self.reverse_topdown_mu = nn.Sequential(
            # output: [B, 128, 9, 16]
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 128, 18, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 64, 36, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 32, 72, 128]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 3, 144, 256]
            nn.Sigmoid()
        )

        self.reverse_topdown_logvar = nn.Sequential(
            # output: [B, 128, 9, 16]
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 128, 18, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 64, 36, 64]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 32, 72, 128]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            # output: [B, 2, 144, 256]
            nn.Sigmoid()
        )

        reverse_feature_size = 64 * 9 * 16
        self.reverse_curSpeed = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=reverse_feature_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

        self.reverse_tarSpeed = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=reverse_feature_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

        self.reverse_lightState = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=reverse_feature_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=self.light_classes_num),
        )

        self.reverse_lightDist = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=reverse_feature_size, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

        # self.cel = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.5, 3, 5, 5, 5]), reduction='mean')
        # self.mse = nn.MSELoss()
        # self.bec = nn.L1Loss()

    def encode(self, x, lidar):
        # image_f = self.preprocess_rgb(x)
        # lidar_f = self.preprocess_lidar(lidar)
        # features = {}
        # features['image'] = image_f
        # features['lidar'] = lidar_f
        # features = sum(features.values())
        encode_input = torch.cat((x, lidar), dim=1)
        features = self.encoder(encode_input)

        mu_z = self.image_mu(features)
        logvar_z = self.image_logvar(features)
        return mu_z, logvar_z

    def reparameterize(self, mu, logvar, training):
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        if training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(self.no_of_samples):
                std = logvar.mul(0.5).exp_()  # type: Variable
                # std = logvar.mul(0.5).exp_()  # type: Variable
                # - std.data is the [128,ZDIMS] tensor that is wrapped by std
                # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
                #   and stddev 1 normal distribution that is 128 samples
                #   of random ZDIMS-float vectors

                eps = std.data.new(std.size()).normal_().clone()
                # eps = std.data.new(std.size())
                # new_tensor.data = new_tensor.normal_()
                # eps.normal_()
                # eps = torch.tensor(std.data.new(std.size()).normal_())
                # - sample from a normal distribution with standard
                #   deviation = std and mean = mu by multiplying mean 0
                #   stddev 1 sample with desired std and mu, see
                #   https://stats.stackexchange.com/a/16338
                # - so we have 128 sets (the batch) of random ZDIMS-float
                #   vectors sampled from normal distribution with learned
                #   std and mu for the current input
                # dist = torch.mul(eps, std)
                # dist = torch.add(dist, mu)
                sample_z.append(eps.mul(std).add_(mu))
                # sample_z.append(dist)

            return sample_z

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            # return [mu]
            sample_z = []
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = std.data.new(std.size()).normal_().clone()
            sample_z.append(eps.mul(std).add_(mu))

            return sample_z
            
    def decode(self, z):
        reverse_feature = self.reverse_feature(z)
        reverse_curSpeed = self.reverse_curSpeed(reverse_feature)
        reverse_tarSpeed = self.reverse_tarSpeed(reverse_feature)
        reverse_lightState = self.reverse_lightState(reverse_feature)
        reverse_lightDist = self.reverse_lightDist(reverse_feature)

        reverse_lidar_mu = self.reverse_lidar_mu(reverse_feature)
        reverse_lidar_logvar = self.reverse_lidar_logvar(reverse_feature)
        reverse_image_mu = self.reverse_image_mu(reverse_feature)
        reverse_image_logvar = self.reverse_image_logvar(reverse_feature)
        reverse_topdown_mu = self.reverse_topdown_mu(reverse_feature)
        reverse_topdown_logvar = self.reverse_topdown_logvar(reverse_feature)

        return reverse_curSpeed, reverse_tarSpeed, \
                reverse_lightState, reverse_lightDist, \
                [reverse_image_mu, reverse_image_logvar], \
                [reverse_lidar_mu, reverse_lidar_logvar], \
                [reverse_topdown_mu, reverse_topdown_logvar] 

    def forward(self, image, lidar, training=True):
        mu, logvar = self.encode(image, lidar)
        
        zs = self.reparameterize(mu, logvar, training)

        img_pred = []
        lidar_pred = []
        topdown_pred = []
        curSpeed_pred = []
        tarSpeed_pred = []
        lightState_pred = []
        lightDist_pred = []
        for z in zs:
            z_decode = self.decode(z)
            reverse_curSpeed, reverse_tarSpeed, \
            reverse_lightState, reverse_lightDist, \
            reverse_image, \
            reverse_lidar, \
            reverse_topdown = z_decode

            img_pred.append(reverse_image)
            lidar_pred.append(reverse_lidar)
            topdown_pred.append(reverse_topdown)
            curSpeed_pred.append(reverse_curSpeed)
            tarSpeed_pred.append(reverse_tarSpeed)
            lightState_pred.append(reverse_lightState)
            lightDist_pred.append(reverse_lightDist)

        return img_pred, lidar_pred, topdown_pred, curSpeed_pred, tarSpeed_pred, \
            lightState_pred, lightDist_pred, mu, logvar

    def get_latent_feature(self, image, lidar):
        mu, logvar = self.encode(image, lidar)

        return mu
        
    def test_forward(self, image, lidar):
        training = False
        results = self.forward(image, lidar, training)
        return results

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


def get_model(net_params):
    return vanillaVAE(net_params)