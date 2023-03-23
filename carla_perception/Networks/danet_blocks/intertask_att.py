import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchsnooper
from torch.nn import functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class InterTaskAtt(nn.Module):

    def __init__(self, net_params):
        super(InterTaskAtt, self).__init__()
        self.input_h = 5
        self.input_w = 8
        self.da_feature_channel = net_params['da_feature_channel']
        # self.z_dims = net_params['z_dims'] * 4
        self.inter_att_dims = net_params['inter_att_dims']
        self.model_name = net_params['model_name']
        if self.model_name == 'danet':
            self.z_dims = net_params['z_dims']
        elif self.model_name == 'da_beta_vae':
            self.z_dims = net_params['inter_att_dims']

        #['transformer', 'position']
        self.att_type = net_params['att_type']
        if self.att_type == 'transformer':
            self.input_dim = self.da_feature_channel * self.input_h * self.input_w
            # self.att_dim = 1024 #512 #128
            # self.temperature = self.att_dim ** 0.5
            self.temperature = self.z_dims ** 0.5
            # self.temperature = 1024 ** 0.5

            # self.visual_query_layer = nn.Linear(self.input_dim, self.att_dim, bias=False)
            # self.bc_query_layer = nn.Linear(self.input_dim, self.att_dim, bias=False)
            self.visual_query_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )
            self.bc_query_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )
            
            # self.visual_key_layer = nn.Linear(self.input_dim, self.att_dim, bias=False)
            # self.bc_key_layer = nn.Linear(self.input_dim, self.att_dim, bias=False)
            self.visual_key_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )
            self.bc_key_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )

            # self.visual_value_layer = nn.Linear(self.input_dim, self.att_dim, bias=False)
            # self.bc_value_layer = nn.Linear(self.input_dim, self.att_dim, bias=False)
            self.visual_value_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )
            self.bc_value_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )

            # self.visual_fc = nn.Linear(self.att_dim, self.input_dim, bias=False)
            # self.bc_fc = nn.Linear(self.att_dim, self.input_dim, bias=False)
            
            # self.visual_fc = nn.Linear(self.att_dim, self.att_dim, bias=False)
            # self.bc_fc = nn.Linear(self.att_dim, self.att_dim, bias=False)

            self.dropout = nn.Dropout(0.1)
            self.softmax = nn.Softmax(dim=-1)

        elif self.att_type == 'position':
            in_dim = self.input_c
            self.visual_query_layer = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.bc_query_layer = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

            self.visual_key_layer = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.bc_key_layer = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

            self.visual_value_layer = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
            self.bc_value_layer = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

            self.visual_gamma = nn.Parameter(torch.zeros(1))
            self.bc_gamma = nn.Parameter(torch.zeros(1))
            self.softmax = nn.Softmax(dim=-1)

        elif self.att_type == 'invaild':
            self.input_dim = self.da_feature_channel * self.input_h * self.input_w
            self.visual_value_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )
            self.bc_value_layer = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=self.input_dim, out_features=self.inter_att_dims),
                nn.LeakyReLU(),
                nn.Linear(in_features=self.inter_att_dims, out_features=self.z_dims),
            )

    def forward(self, da_visual_x, da_bc_x):
        b, c, h, w = da_visual_x.size()
        if self.att_type == 'transformer':  
            da_visual_x = da_visual_x.view(b, -1)
            da_bc_x = da_bc_x.view(b, -1)

            # [b, 128]
            da_visual_q = self.visual_query_layer(da_visual_x)
            da_visual_k = self.visual_key_layer(da_visual_x)
            da_visual_v = self.visual_value_layer(da_visual_x)

            da_bc_q = self.bc_query_layer(da_bc_x)
            da_bc_k = self.bc_key_layer(da_bc_x)
            da_bc_v = self.bc_value_layer(da_bc_x)

            ############ visual to bc ##############
            da_visual_q = da_visual_q.view(b, 1, self.z_dims).permute(0, 2, 1)
            da_bc_k = da_bc_k.view(b, 1, self.z_dims) 
            visual2bc_energy = torch.bmm(da_visual_q / self.temperature, da_bc_k)
            visual2bc_att = self.dropout(self.softmax(visual2bc_energy))
            
            # visual2bc_att1 = self.softmax(visual2bc_energy)
            # tmp1 = visual2bc_att1[0]
            # tmp11 = tmp1[0]
            # print("atten row sum: %f" % torch.sum(tmp11))
            # tmp12 = tmp1[:, 0]
            # print("atten colum sum: %f" % torch.sum(tmp12))

            da_bc_v = da_bc_v.view(b, 1, self.z_dims)
            att_bc_out = torch.bmm(da_bc_v, visual2bc_att.permute(0, 2, 1))
            att_bc_out = att_bc_out.view(b, -1)

            # att_bc_out = self.bc_fc(att_bc_out)
            # att_bc_out = att_bc_out.view(b, c, h, w)
            # da_bc_x = da_bc_x.view(b, c, h, w)
            # att_bc_out = att_bc_out + da_bc_x

            da_bc_v = da_bc_v.view(b, -1)
            att_bc_out = att_bc_out + da_bc_v

            ############ bc to visual ##############
            da_bc_q = da_bc_q.view(b, 1, self.z_dims).permute(0, 2, 1)
            da_visual_k = da_visual_k.view(b, 1, self.z_dims) 
            bc2visual_energy = torch.bmm(da_bc_q / self.temperature, da_visual_k)
            bc2visual_att = self.dropout(self.softmax(bc2visual_energy))
            da_visual_v = da_visual_v.view(b, 1, self.z_dims)
            att_visual_out = torch.bmm(da_visual_v, bc2visual_att.permute(0, 2, 1))
            att_visual_out = att_visual_out.view(b, -1)

            # att_visual_out = self.visual_fc(att_visual_out)
            # att_visual_out = att_visual_out.view(b, c, h, w)
            # da_visual_x = da_visual_x.view(b, c, h, w)
            # att_visual_out = att_visual_out + da_visual_x

            da_visual_v = da_visual_v.view(b, -1)
            att_visual_out = att_visual_out + da_visual_v

        elif self.att_type == 'position':
            da_visual_q = self.visual_query_layer(da_visual_x)
            da_visual_k = self.visual_key_layer(da_visual_x)
            da_visual_v = self.visual_value_layer(da_visual_x)

            da_bc_q = self.bc_query_layer(da_bc_x)
            da_bc_k = self.bc_key_layer(da_bc_x)
            da_bc_v = self.bc_value_layer(da_bc_x)

            ############ visual to bc ##############
            # [b, 5*8, 128]
            da_visual_q = da_visual_q.view(b, -1, h*w).permute(0, 2, 1)
            # [b, 128, 5*8]
            da_bc_k = da_bc_k.view(b, -1, h*w) 
            # [b, 5*8, 5*8]
            visual2bc_energy = torch.bmm(da_visual_q, da_bc_k)
            visual2bc_att = self.softmax(visual2bc_energy)

            # visual2bc_att1 = self.softmax(visual2bc_energy)
            # tmp1 = visual2bc_att1[0]
            # tmp11 = tmp1[0]
            # print("atten row sum: %f" % torch.sum(tmp11))
            # tmp12 = tmp1[:][0]
            # print("atten colum sum: %f" % torch.sum(tmp12))

            da_bc_v = da_bc_v.view(b, -1, h*w)
            att_bc_out = torch.bmm(da_bc_v, visual2bc_att.permute(0, 2, 1))
            att_bc_out = att_bc_out.view(b, c, h, w)
            att_bc_out = self.bc_gamma * att_bc_out + da_bc_x

            ############ bc to visual ##############
            da_bc_q = da_bc_q.view(b, -1, h*w).permute(0, 2, 1)
            da_visual_k = da_visual_k.view(b, -1, h*w) 
            bc2visual_energy = torch.bmm(da_bc_q, da_visual_k)
            bc2visual_att = self.softmax(bc2visual_energy)

            da_visual_v = da_visual_v.view(b, -1, h*w)
            att_visual_out = torch.bmm(da_visual_v, bc2visual_att.permute(0, 2, 1))
            att_visual_out = att_visual_out.view(b, c, h, w)
            att_visual_out = self.visual_gamma * att_visual_out + da_visual_x
        
        elif self.att_type == 'invaild':
            da_visual_v = self.visual_value_layer(da_visual_x)
            da_visual_v = da_visual_v.view(b, -1)
            att_visual_out = da_visual_v

            da_bc_v = self.bc_value_layer(da_bc_x)
            da_bc_v = da_bc_v.view(b, -1)
            att_bc_out = da_bc_v

        # print("att_visual_out size: %s \n" % str(att_visual_out.size()))
        # print("att_bc_out size: %s \n" % str(att_bc_out.size()))

        return att_visual_out, att_bc_out