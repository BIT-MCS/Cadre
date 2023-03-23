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
    def __init__(self, z_dims, no_of_samples, batch_size, training, device_name):
        super(vanillaVAE, self).__init__()

        self.preprocess_rgb = nn.Sequential(
            # F.interpolate([1, 144, 256]),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=(5, 5)),
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
        ).to(device_name)

        self.preprocess_lidar = nn.Sequential(
            # F.interpolate([1, 144, 256]),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=(5, 5)),
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
            # nn.LeakyReLU(),
            Flatten()
        ).to(device_name)

        # self.feature = nn.Sequential(
        #     # input: [B, 64, 34, 62]
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2),
        #     # output: [B, 64, 16, 30]
        #     nn.ReLU(),
        #     Flatten(),
        # ).to(device_name)
        output_h = 13
        output_w = 20
        # output_h = 22
        # output_w = 36
        # u-net
        self.image_mu = nn.Sequential(
            nn.Linear(in_features=64 * output_h * output_w, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=z_dims),
        ).to(device_name)

        self.image_logvar = nn.Sequential(
            nn.Linear(in_features=64 * output_h * output_w, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=z_dims),
        ).to(device_name)

        # For decoder
        self.reverse_feature = nn.Sequential(
            nn.Linear(in_features=z_dims, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=64 * 9 * 16),
            # nn.Linear(in_features=1024, out_features=64 * 18 * 32),
            nn.LeakyReLU(),
            Reshape(),
        ).to(device_name)

        # self.reverse_topdown = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=6, kernel_size=4, stride=2),
        # ).to(device_name)

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
        ).to(device_name)

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
        ).to(device_name)
        self.device = device_name

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
        ).to(device_name)
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
        ).to(device_name)

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
        ).to(device_name)
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
        ).to(device_name)

        # self.cel = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.5, 3, 5, 5, 5]).to(device_name), reduction='mean')
        # self.mse = nn.MSELoss()
        # self.bec = nn.L1Loss()
        self.no_of_samples = no_of_samples
        self.batch_size = batch_size
        self.training = training

    def encode(self, x, lidar, topdown=None):
        image_f = self.preprocess_rgb(x)
        lidar_f = self.preprocess_lidar(lidar)
        features = {}
        features['image'] = image_f
        features['lidar'] = lidar_f
        features = sum(features.values())

        mu_z = self.image_mu(features)
        logvar_z = self.image_logvar(features)
        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):
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

        if self.training:
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
            return mu

    def decode(self, z):
        reverse_feature = self.reverse_feature(z)
        reverse_lidar_mu = self.reverse_lidar_mu(reverse_feature)
        reverse_lidar_logvar = self.reverse_lidar_logvar(reverse_feature)
        reverse_image_mu = self.reverse_image_mu(reverse_feature)
        reverse_image_logvar = self.reverse_image_logvar(reverse_feature)
        reverse_topdown_mu = self.reverse_topdown_mu(reverse_feature)
        reverse_topdown_logvar = self.reverse_topdown_logvar(reverse_feature)

        return [reverse_image_mu, reverse_image_logvar], [reverse_lidar_mu, reverse_lidar_logvar], [reverse_topdown_mu,
                                                                                                    reverse_topdown_logvar]
    def forward(self, image, lidar, topdown):
        mu, logvar = self.encode(image, lidar, topdown)
        z = self.reparameterize(mu, logvar)
        # print('2--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return [self.decode(z)], mu, logvar

    def loss_function(self, recon_x, gth_image, gth_lidar, gth_topdown, mu, logvar):
        GLL = 0
        for recon_x_one in recon_x:
            logits_image, logits_lidar, logits_topdown = recon_x_one

            image_mu = logits_image[0]
            image_sigma = logits_image[1].mul(0.5).exp_()
            part1 = torch.sum(((gth_image - image_mu) / image_sigma) ** 2) / self.batch_size

            lidar_mu = logits_lidar[0]
            lidar_sigma = logits_lidar[1].mul(0.5).exp_()
            part2 = torch.sum(((gth_lidar - lidar_mu) / lidar_sigma) ** 2) / self.batch_size


            topdown_mu = logits_topdown[0]
            topdown_sigma = logits_topdown[1].mul(0.5).exp_()
            part3 = torch.sum(((gth_topdown - topdown_mu) / topdown_sigma) ** 2) / self.batch_size

            # mu_image = mu_image.view(-1, 1)
            # logvar_image = logvar_image.view(-1, 1)

            # part1 = torch.sum(logvar_image) / self.batch_size
            # sigma = logvar_image.mul(0.5).exp_()
            # part2 = torch.sum(((gth_image - mu_image) / sigma) ** 2) / self.batch_size
            # part3 = self.bce(logits_lidar, gth_lidar)
            # part3 = self.bce(logits_lidar, gth_lidar) / self.batch_size
            # part4 = self.cel(prob_topdown, gth_topdown) / self.batch_size
            GLL += (0.33 * part1 + 0.33 * part2 + 0.33 * part3)
            # GLL += (0.33 * 0.5 * (part1 + part2) + 0.33 * part3 + 0.33 * part4)

        GLL /= len(recon_x)

        # # see Appendix B from VAE paper:
        # # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # # https://arxiv.org/abs/1312.6114
        # # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD = KLD / self.batch_size

        # return GLL + KLD, [part1.item(), part2.item(), part3.item(), KLD.item()]
        return GLL + KLD, [part1.item(), part2.item(), part3.item(), KLD.item()]

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


def get_model(net_params):
    return vanillaVAE(net_params)