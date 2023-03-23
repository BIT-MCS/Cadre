import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchsnooper
from torch.nn import functional as F
from Networks.danet_blocks.resnet import ResNet, model_funcs
from Networks.danet_blocks.bc_branch import BCBranch
from Networks.danet_blocks.intertask_att import InterTaskAtt
from Networks.danet_blocks.visual_branch import VisualBranch
from Networks.danet_blocks.da_att import PAM_Module, CAM_Module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)
        # print("sa_output size: %s" % str(sa_output.size()))

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)
        # print("sc_output size: %s" % str(sc_output.size()))

        # [b, 128, 5, 8]
        feat_sum = sa_conv+sc_conv
        # print("feat_sum size: %s" % str(feat_sum.size()))
        
        sasc_output = self.conv8(feat_sum)
        # print("sasc_output size: %s" % str(sasc_output.size()))

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)

        # return tuple(output)

        return sasc_output

class DABetaVae(nn.Module):

    def __init__(self, net_params):
        super(DABetaVae, self).__init__()

        self.net_name = net_params['net_name']
        self.model_name = net_params['model_name']
        self.input_channel = net_params['input_channel']
        self.da_feature_channel = net_params['da_feature_channel']
        self.pred_bc = net_params['pred_bc']
        self.z_dims = net_params['z_dims']
        self.in_bc_speed = net_params['in_bc_speed']
        self.inter_att_dims = net_params['inter_att_dims']

        # self.backbone_name = 'resnet34'
        self.backbone_name = 'resnet18'
        block, layers, _ = model_funcs[self.backbone_name]
        self.backbone = ResNet(block, layers, self.input_channel)

        output_c = 512
        output_h = 5
        output_w = 8

        self.da_head = DANetHead(output_c, self.da_feature_channel)

        self.visual_conv = nn.Conv2d(self.da_feature_channel, self.da_feature_channel, 1)
        self.visual_branch = VisualBranch(net_params)

        if self.pred_bc:
            if self.in_bc_speed:
                self.in_bc_speed_fc = nn.Sequential(
                    Flatten(),
                    nn.Linear(in_features=1, out_features=64),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=64, out_features=self.inter_att_dims),
                )

            self.bc_conv = nn.Conv2d(self.da_feature_channel, self.da_feature_channel, 1)
            self.inter_task_att = InterTaskAtt(net_params)
            self.bc_branch = BCBranch(net_params)

            self.visual_mu = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.z_dims, out_features=self.z_dims),
            )

            self.visual_logvar = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.z_dims, out_features=self.z_dims),
            )

            self.bc_mu = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.z_dims, out_features=self.z_dims),
            )

            self.bc_logvar = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.z_dims, out_features=self.z_dims),
            )
        else:
            self.visual_mu = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.da_feature_channel * output_h * output_w, out_features=1024),
                nn.LeakyReLU(),
                nn.Linear(in_features=1024, out_features=self.z_dims),
            )

            self.visual_logvar = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.da_feature_channel * output_h * output_w, out_features=1024),
                nn.LeakyReLU(),
                nn.Linear(in_features=1024, out_features=self.z_dims),
            )

    def decode_with_graph(self, x, graph, training=False):
        # [b, 512, 5, 8]
        layer4_x = self.backbone(x)
        # da_visual_x, da_bc_x = self.da_head(layer4_x)
        da_att_x = self.da_head(layer4_x)
        # [b, self.da_feature_channel, 5, 8]
        da_visual_x = self.visual_conv(da_att_x)
        
        if self.pred_bc:
            # [b, self.da_feature_channel, 5, 8]
            da_bc_x = self.bc_conv(da_att_x)

            # [b, self.da_feature_channel // 4, 5, 8]
            att_visual_x, att_bc_x = self.inter_task_att(da_visual_x, da_bc_x)
            if self.in_bc_speed and bc_speed is not None:
                bc_speed_feature = self.in_bc_speed_fc(bc_speed)
                att_bc_x = att_bc_x + bc_speed_feature
        else:
            att_visual_x = self.visual_fc(da_visual_x)

        att_visual_x = att_visual_x * graph.view_as(att_visual_x)
        visual_decode = self.visual_branch(att_visual_x)
        _reverse_lightState, \
        _reverse_lightDist, \
        _reverse_image, \
        _reverse_lidar, \
        _reverse_topdown, \
        _reverse_route, \
        _reverse_left_image, \
        _reverse_right_image = visual_decode

        _pred_steer = None
        _pred_throttle = None
        if self.pred_bc:
            bc_decode = self.bc_branch(att_bc_x)
            _pred_steer = bc_decode[:, 0]
            _pred_throttle = bc_decode[:, 1]
        
        return  _reverse_lightState, \
                _reverse_lightDist, \
                _reverse_image, \
                _reverse_lidar, \
                _reverse_topdown, \
                _reverse_route, \
                _reverse_left_image, \
                _reverse_right_image, \
                _pred_steer, _pred_throttle

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

    def forward(self, x, bc_speed=None):
        # [b, 512, 5, 8]
        layer4_x = self.backbone(x)
        # da_visual_x, da_bc_x = self.da_head(layer4_x)
        da_att_x = self.da_head(layer4_x)
        # [b, self.da_feature_channel, 5, 8]
        da_visual_x = self.visual_conv(da_att_x)
        
        if self.pred_bc:
            # [b, self.da_feature_channel, 5, 8]
            da_bc_x = self.bc_conv(da_att_x)

            # [b, self.z_dim, 5, 8] for position (invaild now)
            # [b, self.z_dim] for transformer
            att_visual_x, att_bc_x = self.inter_task_att(da_visual_x, da_bc_x)
            if self.in_bc_speed and bc_speed is not None:
                bc_speed_feature = self.in_bc_speed_fc(bc_speed)
                att_bc_x = att_bc_x + bc_speed_feature

            att_visual_mu = self.visual_mu(att_visual_x)
            att_visual_logvar = self.visual_logvar(att_visual_x)
            training = True
            att_visual_z = self.reparameterize(att_visual_mu, att_visual_logvar, training)
            
            att_bc_mu = self.bc_mu(att_bc_x)
            att_bc_logvar = self.bc_logvar(att_bc_x)
            training = True
            att_bc_z = self.reparameterize(att_bc_mu, att_bc_logvar, training)
        else:
            att_bc_mu = None
            att_bc_logvar = None

            att_visual_mu = self.visual_mu(da_visual_x)
            att_visual_logvar = self.visual_logvar(da_visual_x)
            training = True
            att_visual_z = self.reparameterize(att_visual_mu, att_visual_logvar, training)

        visual_decode = self.visual_branch(att_visual_z)
        _reverse_lightState, \
        _reverse_lightDist, \
        _reverse_image, \
        _reverse_lidar, \
        _reverse_topdown, \
        _reverse_route, \
        _reverse_left_image, \
        _reverse_right_image = visual_decode

        _pred_steer = None
        _pred_throttle = None
        if self.pred_bc:
            bc_decode = self.bc_branch(att_bc_z)
            _pred_steer = bc_decode[:, 0]
            _pred_throttle = bc_decode[:, 1]
        
        return  _reverse_lightState, \
                _reverse_lightDist, \
                _reverse_image, \
                _reverse_lidar, \
                _reverse_topdown, \
                _reverse_route, \
                _reverse_left_image, \
                _reverse_right_image, \
                _pred_steer, _pred_throttle, \
                att_visual_mu, att_visual_logvar, \
                att_bc_mu, att_bc_logvar

    def test_forward(self, x):
        results = self.forward(x)
        return results

    def get_latent_feature(self, x):
        # [b, 512, 5, 8]
        layer4_x = self.backbone(x)
        # da_visual_x, da_bc_x = self.da_head(layer4_x)
        da_att_x = self.da_head(layer4_x)
        # [b, self.da_feature_channel, 5, 8]
        da_visual_x = self.visual_conv(da_att_x)
        
        if self.pred_bc:
            # [b, self.da_feature_channel, 5, 8]
            da_bc_x = self.bc_conv(da_att_x)

            # [b, self.da_feature_channel // 4, 5, 8]
            att_visual_x, att_bc_x = self.inter_task_att(da_visual_x, da_bc_x)
            if self.in_bc_speed and bc_speed is not None:
                bc_speed_feature = self.in_bc_speed_fc(bc_speed)
                att_bc_x = att_bc_x + bc_speed_feature

            # ppo_latent_feature = att_visual_x + att_bc_x
            att_visual_mu = self.visual_mu(att_visual_x)
            att_visual_logvar = self.visual_logvar(att_visual_x)
            training = True
            att_visual_z = self.reparameterize(att_visual_mu, att_visual_logvar, training)
            
            att_bc_mu = self.bc_mu(att_bc_x)
            att_bc_logvar = self.bc_logvar(att_bc_x)
            training = True
            att_bc_z = self.reparameterize(att_bc_mu, att_bc_logvar, training)
        else:
            att_bc_mu = None
            att_bc_logvar = None
            att_bc_z = None

            att_visual_mu = self.visual_mu(da_visual_x)
            att_visual_logvar = self.visual_logvar(da_visual_x)
            training = True
            att_visual_z = self.reparameterize(att_visual_mu, att_visual_logvar, training)

        
        # return ppo_latent_feature
        return att_bc_z, att_visual_z

    def get_ppo_input(self, x):
        # [b, 512, 5, 8]
        layer4_x = self.backbone(x)
        # da_visual_x, da_bc_x = self.da_head(layer4_x)
        da_att_x = self.da_head(layer4_x)
        # [b, self.da_feature_channel, 5, 8]
        da_visual_x = self.visual_conv(da_att_x)
        
        if self.pred_bc:
            # [b, self.da_feature_channel, 5, 8]
            da_bc_x = self.bc_conv(da_att_x)

            # [b, self.da_feature_channel // 4, 5, 8]
            att_visual_x, att_bc_x = self.inter_task_att(da_visual_x, da_bc_x)
            if self.in_bc_speed and bc_speed is not None:
                bc_speed_feature = self.in_bc_speed_fc(bc_speed)
                att_bc_x = att_bc_x + bc_speed_feature
                
            # ppo_latent_feature = att_visual_x + att_bc_x
            att_visual_mu = self.visual_mu(att_visual_x)
            att_visual_logvar = self.visual_logvar(att_visual_x)
            training = True
            att_visual_z = self.reparameterize(att_visual_mu, att_visual_logvar, training)
            
            att_bc_mu = self.bc_mu(att_bc_x)
            att_bc_logvar = self.bc_logvar(att_bc_x)
            training = True
            att_bc_z = self.reparameterize(att_bc_mu, att_bc_logvar, training)
        else:
            # att_visual_x = self.visual_fc(da_visual_x)
            # ppo_latent_feature = att_visual_x
            att_bc_mu = None
            att_bc_logvar = None

            att_visual_mu = self.visual_mu(da_visual_x)
            att_visual_logvar = self.visual_logvar(da_visual_x)
            training = True
            att_visual_z = self.reparameterize(att_visual_mu, att_visual_logvar, training)

        
        # visual_decode = self.visual_branch(att_visual_x)
        visual_decode = self.visual_branch(att_visual_z)
        _reverse_lightState, \
        _reverse_lightDist, \
        _reverse_image, \
        _reverse_lidar, \
        _reverse_topdown, \
        _reverse_route, \
        _reverse_left_image, \
        _reverse_right_image = visual_decode

        # return ppo_latent_feature, _reverse_lightState
        return att_bc_z, att_visual_z, _reverse_lightState

def get_model(net_params):
    return DABetaVae(net_params)