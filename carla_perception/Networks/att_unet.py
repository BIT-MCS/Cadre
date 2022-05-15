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


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, net_params):
        super(AttU_Net, self).__init__()

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.output_channel = net_params['output_channel']
        self.light_classes_num = net_params['light_classes_num']
        self.net_name = net_params['net_name']

        self.z_dims = net_params['z_dims']

        img_ch = self.input_channel
        output_ch = self.output_channel

        n1 = 64
        # [64, 128, 256, 512, 1024]
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

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
        # self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up5 = up_conv(filters[4], filters[4])
        self.Att5 = Attention_block(F_g=filters[4], F_l=filters[4], F_int=filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        # self.Up4 = up_conv(filters[3], filters[2])
        # self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up4 = up_conv(filters[3], filters[3])
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        # self.Up3 = up_conv(filters[2], filters[1])
        # self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up3 = up_conv(filters[2], filters[2])
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up2 = up_conv(filters[1], filters[1])
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        z = self.fc_encoder(e5)

        reverse_feature = self.fc_decoder(z)

        reverse_curSpeed = self.reverse_curSpeed(reverse_feature)
        reverse_tarSpeed = self.reverse_tarSpeed(reverse_feature)
        reverse_lightState = self.reverse_lightState(reverse_feature)
        reverse_lightDist = self.reverse_lightDist(reverse_feature)

        # #print(x5.shape)
        # d5 = self.Up5(e5)
        # #print(d5.shape)
        # x4 = self.Att5(g=d5, x=e4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d5 = self.Up5(reverse_feature)
        d5 = self.Att5(g=d5, x=d5)
        d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)
        # x3 = self.Att4(g=d4, x=e3)
        # d4 = torch.cat((x3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d4 = self.Up4(d5)
        d4 = self.Att4(g=d4, x=d4)
        d4 = self.Up_conv4(d4)

        # d3 = self.Up3(d4)
        # x2 = self.Att3(g=d3, x=e2)
        # d3 = torch.cat((x2, d3), dim=1)
        # d3 = self.Up_conv3(d3)
        d3 = self.Up3(d4)
        d3 = self.Att3(g=d3, x=d3)
        d3 = self.Up_conv3(d3)

        # d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=e1)
        # d2 = torch.cat((x1, d2), dim=1)
        # d2 = self.Up_conv2(d2)
        d2 = self.Up2(d3)
        d2 = self.Att2(g=d2, x=d2)
        d2 = self.Up_conv2(d2)

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

    def get_latent_feature(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        z = self.fc_encoder(e5)
        
        return z

def get_model(net_params):
    return AttU_Net(net_params)