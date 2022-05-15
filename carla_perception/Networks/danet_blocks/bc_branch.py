import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchsnooper
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BCBranch(nn.Module):

    def __init__(self, net_params):
        super(BCBranch, self).__init__()

        self.output_channel = 2
        # self.input_h = 5
        # self.input_w = 8
        # self.input_channel = net_params['feature_channel']
        # self.conv_layer_nums = 3
        # self.channel_nums = []
        # self.input_feature_size = self.input_h * self.input_w
        self.z_dims = net_params['z_dims']

        
        # for i in range(self.conv_layer_nums + 1):
        #     self.channel_nums.append(self.input_channel // (2**i))

        # self.conv1 = nn.Conv2d(self.channel_nums[0], self.channel_nums[1], 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(self.channel_nums[1])
        # self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(self.channel_nums[1], self.channel_nums[2], 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.channel_nums[2])
        # self.relu2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(self.channel_nums[2], self.channel_nums[3], 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(self.channel_nums[3])
        # self.relu3 = nn.ReLU()

        # fc1_input = self.channel_nums[3] * self.input_feature_size
        # fc1_output = self.channel_nums[3] * self.input_feature_size // 2

        fc1_input = self.z_dims
        fc1_output = self.z_dims // 2

        # self.fc1 = nn.Linear(fc1_input, 
        #                     fc1_output)
        # self.fc_bn1 = nn.BatchNorm1d(fc1_output)       
        
        # self.fc2 = nn.Linear(fc1_output, 
        #                     self.output_channel)

        self.bc_model = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=fc1_input, out_features=fc1_output),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc1_output, out_features=self.output_channel),
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)

        # x = self.fc1(x.view(x.size()[0], -1))
        # x = self.fc_bn1(x)
        # x = self.fc2(x)
        x = self.bc_model(x)

        return x