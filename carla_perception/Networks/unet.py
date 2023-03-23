import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch.nn.functional as F
import torch.utils.data

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

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, net_params):
        super(U_Net, self).__init__()

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.output_channel = net_params['output_channel']
        self.light_classes_num = net_params['light_classes_num']

        self.z_dims = net_params['z_dims']
        self.pred_light = net_params['pred_light']
        self.pred_lidar = net_params['pred_lidar']
        self.pred_topdown_rgb = net_params['pred_topdown_rgb']
        self.pred_topdown_seg = net_params['pred_topdown_seg']

        in_ch = self.input_channel
        out_ch = self.output_channel

        n1 = 64
        # [64, 128, 256, 512, 1024]
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        conv5_output_h = 9
        conv5_output_w = 16
        adaptive_pool_size = (3, 4)

        self.fc_encoder = nn.Sequential(
            nn.AdaptiveMaxPool2d(adaptive_pool_size),
            Flatten(),
            nn.Linear(in_features=filters[4] * adaptive_pool_size[0] * adaptive_pool_size[1], out_features=filters[-2]),
            nn.LeakyReLU(),
            nn.Linear(in_features=filters[-2], out_features=filters[-3]),
            nn.LeakyReLU(),
            nn.Linear(in_features=filters[-3], out_features=self.z_dims),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=self.z_dims, out_features=filters[-3]),
            nn.LeakyReLU(),
            nn.Linear(in_features=filters[-3], out_features=filters[-2]),
            nn.LeakyReLU(),
            nn.Linear(in_features=filters[-2], out_features=filters[4] * conv5_output_h * conv5_output_w),
            Reshape(filters[4], conv5_output_h, conv5_output_w),
            # nn.Linear(in_features=64, out_features=filters[4] * adaptive_pool_size[0] * adaptive_pool_size[1]),
            # Reshape(filters[4], adaptive_pool_size[0], adaptive_pool_size[1]),
            # nn.ConvTranspose2d(in_channels=filters[4], out_channels=filters[4], kernel_size=(5, 5), stride=2),
            # nn.LeakyReLU(),
        )
        
        reverse_feature_size = filters[4] * conv5_output_h * conv5_output_w
        # self.reverse_curSpeed = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(in_features=reverse_feature_size, out_features=64),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=64, out_features=64),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=64, out_features=1),
        # )

        # self.reverse_tarSpeed = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(in_features=reverse_feature_size, out_features=64),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=64, out_features=64),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=64, out_features=1),
        # )
        if self.pred_light:
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

        # self.Up5 = up_conv(filters[4], filters[3])
        self.Up5 = up_conv(filters[4], filters[4])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        # self.Up4 = up_conv(filters[3], filters[2])
        self.Up4 = up_conv(filters[3], filters[3])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        # self.Up3 = up_conv(filters[2], filters[1])
        self.Up3 = up_conv(filters[2], filters[2])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        self.Up2 = up_conv(filters[1], filters[1])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, rgbimg_list, lidar_list):
        unet_input = torch.cat((rgbimg_list, lidar_list), dim=1)

        e1 = self.Conv1(unet_input)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        # 1024
        e5 = self.Conv5(e5)
        # [b. 1024, 9, 16]
        # print('e5 size:' + str(e5.size()))

        z = self.fc_encoder(e5)

        reverse_feature = self.fc_decoder(z)

        # reverse_curSpeed = self.reverse_curSpeed(reverse_feature)
        # reverse_tarSpeed = self.reverse_tarSpeed(reverse_feature)
        if self.pred_light:
            reverse_lightState = self.reverse_lightState(reverse_feature)
            reverse_lightDist = self.reverse_lightDist(reverse_feature)

        # d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)

        # d5 = self.Up_conv5(d5)

        # 512
        d5 = self.Up5(reverse_feature)
        d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        # 256
        d4 = self.Up4(d5)
        d4 = self.Up_conv4(d4)

        # d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_conv3(d3)

        # 128
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        # 64
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        # print('out size:' + str(out.size()))

        #d1 = self.active(out)

        lidar_pred = None
        topdown_pred = None
        lightState_pred = None
        lightDist_pred = None

        img_pred = out[:, :3, :, :]
        if self.pred_light:
            lightState_pred = reverse_lightState
            lightDist_pred = reverse_lightDist

        if self.pred_lidar:
            lidar_pred = out[:, 3:3+3, :, :]
        
        if self.pred_topdown_rgb:
            topdown_pred = out[:, -3:, :, :]
        
        if self.pred_topdown_seg:
            topdown_pred = out[:, -1:, :, :]
        # curSpeed_pred = reverse_curSpeed
        # tarSpeed_pred = reverse_tarSpeed
        
        return img_pred, lidar_pred, topdown_pred, \
            lightState_pred, lightDist_pred

    def get_latent_feature(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        # 1024
        e5 = self.Conv5(e5)
        # [b. 1024, 9, 16]
        # print('e5 size:' + str(e5.size()))

        z = self.fc_encoder(e5)
        return z


def get_model(net_params):
    return U_Net(net_params)