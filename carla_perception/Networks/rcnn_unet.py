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


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, net_params):
        super(R2U_Net, self).__init__()

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.output_channel = net_params['output_channel']
        self.light_classes_num = net_params['light_classes_num']
        self.net_name = net_params['net_name']
        self.num_input = net_params['num_input']

        self.z_dims = net_params['z_dims']

        img_ch = self.input_channel
        output_ch = self.output_channel
        t = self.num_input

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        conv5_output_h = 9
        conv5_output_w = 16

        self.fc_encoder = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=filters[4] * conv5_output_h * conv5_output_w, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=self.z_dims),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=self.z_dims, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=filters[4] * conv5_output_h * conv5_output_w),
            Reshape(filters[4], conv5_output_h, conv5_output_w),
        )
        
        reverse_feature_size = filters[4] * conv5_output_h * conv5_output_w
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

        # self.Up5 = up_conv(filters[4], filters[3])
        self.Up5 = up_conv(filters[4], filters[4])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        # self.Up4 = up_conv(filters[3], filters[2])
        self.Up4 = up_conv(filters[3], filters[3])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        # self.Up3 = up_conv(filters[2], filters[1])
        self.Up3 = up_conv(filters[2], filters[2])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        # self.Up2 = up_conv(filters[1], filters[0])
        self.Up2 = up_conv(filters[1], filters[1])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5) 
        
        z = self.fc_encoder(e5)

        reverse_feature = self.fc_decoder(z)

        reverse_curSpeed = self.reverse_curSpeed(reverse_feature)
        reverse_tarSpeed = self.reverse_tarSpeed(reverse_feature)
        reverse_lightState = self.reverse_lightState(reverse_feature)
        reverse_lightDist = self.reverse_lightDist(reverse_feature)


        # d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        # d5 = self.Up_RRCNN5(d5)

        d5 = self.Up5(reverse_feature)
        d5 = self.Up_RRCNN5(d5)

        # d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_RRCNN4(d4)

        d4 = self.Up4(d5)
        d4 = self.Up_RRCNN4(d4)

        # d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_RRCNN3(d3)

        d3 = self.Up3(d4)
        d3 = self.Up_RRCNN3(d3)

        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_RRCNN2(d2)

        d2 = self.Up2(d3)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        img_pred = out[:, :3, :, :]
        lidar_pred = out[:, 3:3+3, :, :]
        topdown_pred = out[:, 6:, :, :]
        curSpeed_pred = reverse_curSpeed
        tarSpeed_pred = reverse_tarSpeed
        lightState_pred = reverse_lightState
        lightDist_pred = reverse_lightDist

        return img_pred, lidar_pred, topdown_pred, curSpeed_pred, \
            tarSpeed_pred, lightState_pred, lightDist_pred

def get_model(net_params):
    return R2U_Net(net_params)