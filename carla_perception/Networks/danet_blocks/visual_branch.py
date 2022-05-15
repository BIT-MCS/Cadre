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

class VisualBranch(nn.Module):

    def __init__(self, net_params):
        super(VisualBranch, self).__init__()

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.camera_output_channel = net_params['camera_output_channel']
        self.left_camera_output_channel = net_params['left_camera_output_channel']
        self.right_camera_output_channel = net_params['right_camera_output_channel']
        self.light_classes_num = net_params['light_classes_num']

        self.z_dims = net_params['z_dims']
        self.pred_light_state = net_params['pred_light_state']
        self.pred_light_dist = net_params['pred_light_dist']
        self.pred_lidar = net_params['pred_lidar']
        self.pred_topdown_rgb = net_params['pred_topdown_rgb']
        self.pred_topdown_seg = net_params['pred_topdown_seg']
        self.pred_route = net_params['pred_route']
        self.pred_camera_seg = net_params['pred_camera_seg']
        self.pred_left_camera_seg = net_params['pred_left_camera_seg']
        self.pred_right_camera_seg = net_params['pred_right_camera_seg']
        self.pred_bc = net_params['pred_bc']

        output_h = 5
        output_w = 8
        # self.hidden_dims = [64, 128, 256, 256, 512]
        self.hidden_dims = [32, 64, 128, 256, 512]
        # self.input_dims = 128 * output_h * output_w
        self.input_c = net_params['da_feature_channel']
        # self.reverse_c = 512

        # if self.pred_bc:
        #     self.reverse_feature = nn.Sequential(
        #         nn.Conv2d(self.input_c, self.hidden_dims[-1], 3, padding=1, bias=False),
        #         nn.BatchNorm2d(self.hidden_dims[-1]),
        #         nn.LeakyReLU(),
        #         nn.Conv2d(self.hidden_dims[-1], self.hidden_dims[-1], 3, padding=1, bias=False),
        #         Reshape(self.hidden_dims[-1], output_h, output_w)
        #     )
        # else:
        self.reverse_feature = nn.Sequential(
            nn.Linear(in_features=self.z_dims, out_features=self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dims[-1], out_features=self.hidden_dims[-1] * output_h * output_w),
            # nn.LeakyReLU(),
            Reshape(self.hidden_dims[-1], output_h, output_w),
        )

        # h: [5, 9, 18, 36, 72, 144]
        # w: [8, 16, 32, 64, 128, 256]
        if self.pred_camera_seg:
            # semantic seg
            self.reverse_image = self.build_reverse_module(output=self.camera_output_channel, use_sig=False)
        else:
            self.reverse_image = self.build_reverse_module(output=self.camera_output_channel, use_sig=True)

        if self.pred_left_camera_seg:
            # semantic seg
            self.reverse_left_image = self.build_reverse_module(output=self.left_camera_output_channel, use_sig=False)
        # else:
        #     self.reverse_left_image = self.build_reverse_module(output=self.left_camera_output_channel, use_sig=True)

        if self.pred_right_camera_seg:
            # semantic seg
            self.reverse_right_image = self.build_reverse_module(output=self.right_camera_output_channel, use_sig=False)
        # else:
        #     self.reverse_right_image = self.build_reverse_module(output=self.right_camera_output_channel, use_sig=True)

        if self.pred_route:
            self.reverse_route = self.build_reverse_module(output=1, use_sig=True)

        if self.pred_lidar:
            self.reverse_lidar = self.build_reverse_module(output=3)
        
        if self.pred_topdown_rgb:
            self.reverse_topdown_rgb = self.build_reverse_module(output=3)
        
        if self.pred_topdown_seg:
            self.reverse_topdown_seg = self.build_reverse_module(output=1)

        reverse_feature_size = self.hidden_dims[-1] * output_h * output_w

        if self.pred_light_state:
            self.reverse_lightState = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=reverse_feature_size, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.LeakyReLU(),
                nn.Linear(in_features=64, out_features=self.light_classes_num),
            )

        if self.pred_light_dist:
            self.reverse_lightDist = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=reverse_feature_size, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.LeakyReLU(),
                nn.Linear(in_features=64, out_features=1),
        )

    def build_reverse_module(self, output=3, use_sig=False):
        modules = []
        hidden_dims = self.hidden_dims[::-1]
        # h: [5, 9, 18, 36, 72, 144]
        # w: [8, 16, 32, 64, 128, 256]
        for i in range(len(hidden_dims) - 1):
            if i == 0:
                cur_output_padding = (0, 1)
            else:
                cur_output_padding = 1
            modules.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3,
                                    stride=2, padding=1, output_padding=cur_output_padding))
            modules.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
        
        modules.append(nn.ConvTranspose2d(hidden_dims[-1], output, kernel_size=3,
                                stride=2, padding=1, output_padding=1))
        # for camera seg
        if use_sig:
            modules.append(nn.Sigmoid())

        cur_module = nn.Sequential(*modules)
        return cur_module

    def forward(self, x):
        _reverse_lightState = None
        _reverse_lightDist = None
        _reverse_lidar = None
        _reverse_topdown = None
        _reverse_route = None
        _reverse_left_image = None
        _reverse_right_image = None

        _reverse_feature = self.reverse_feature(x)

        _reverse_image = self.reverse_image(_reverse_feature)
        # if self.pred_curSpeed:
        #     reverse_curSpeed = self.reverse_curSpeed(reverse_feature)
        # if self.pred_tarSpeed:
        #     reverse_tarSpeed = self.reverse_tarSpeed(reverse_feature)
        if self.pred_left_camera_seg:
            _reverse_left_image = self.reverse_left_image(_reverse_feature)

        if self.pred_right_camera_seg:
            _reverse_right_image = self.reverse_right_image(_reverse_feature)

        if self.pred_route:
            _reverse_route = self.reverse_route(_reverse_feature)

        if self.pred_light_state:
            _reverse_lightState = self.reverse_lightState(_reverse_feature)

        if self.pred_light_dist:
            _reverse_lightDist = self.reverse_lightDist(_reverse_feature)

        if self.pred_lidar:
            _reverse_lidar = self.reverse_lidar(_reverse_feature)
        
        if self.pred_topdown_rgb:
            _reverse_topdown = self.reverse_topdown_rgb(_reverse_feature)
        
        if self.pred_topdown_seg:
            _reverse_topdown = self.reverse_topdown_seg(_reverse_feature)

        return  _reverse_lightState, \
                _reverse_lightDist, \
                _reverse_image, \
                _reverse_lidar, \
                _reverse_topdown, \
                _reverse_route, \
                _reverse_left_image, \
                _reverse_right_image
