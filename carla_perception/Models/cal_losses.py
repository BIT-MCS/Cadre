import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn

def get_imgRecon_loss(img_target, img_pred, model_type):
    # img_pred, lidar_pred, topdown_pred, curSpeed_pred, tarSpeed_pred, \
    #         lightState_pred, lightDist_pred, mu, logvar = results
    b, c, h, w = img_target.size()
    recon_loss = 0.0
    if model_type == 'vae':
        recon_loss = 0
        for img_pred_ele in img_pred:
            image_mu = img_pred_ele[0]
            image_sigma = img_pred_ele[1].mul(0.5).exp_()
            tmp = torch.sum(((img_target - image_mu) / image_sigma) ** 2) / b
            recon_loss += tmp
        recon_loss /= len(img_pred)
    elif model_type == 'beta-vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(img_pred, img_target)
        recon_loss *= c * h * w
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(img_pred, img_target)
        recon_loss *= c * h * w
        # cal_l1_loss = nn.L1Loss()
        # recon_loss = cal_l1_loss(img_pred, img_target)
    elif model_type == 'unet':
        cal_l1_loss = nn.L1Loss()
        # print('img_target size: ' + str(img_target.size()))
        # print('img_pred size: ' + str(img_pred.size()))
        recon_loss = cal_l1_loss(img_pred, img_target)
        recon_loss *= c * h * w
    return recon_loss

def get_imgSeg_loss(img_target, img_pred, model_type, loss_weight):
    b, c, h, w = img_target.size()
    seg_loss = 0.0
    # print('img_target size: ' + str(img_target.size()))
    # print('img_pred size: ' + str(img_pred.size()))
    # print('loss_weight size: ' + str(loss_weight.size()))
    if model_type == 'vae':
        img_target = torch.squeeze(img_target, dim=1).long()
        cal_seg_loss = nn.CrossEntropyLoss(weight=loss_weight)
        seg_loss = cal_seg_loss(img_pred, img_target)
        seg_loss *= c * h * w
    elif model_type == 'beta-vae':
        img_target = torch.squeeze(img_target, dim=1).long()
        cal_seg_loss = nn.CrossEntropyLoss(weight=loss_weight)
        seg_loss = cal_seg_loss(img_pred, img_target)
        seg_loss *= c * h * w
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        img_target = torch.squeeze(img_target, dim=1).long()
        # print('img_target size: ' + str(img_target.size()))
        # print('img_pred size: ' + str(img_pred.size()))
        cal_seg_loss = nn.CrossEntropyLoss(weight=loss_weight)
        seg_loss = cal_seg_loss(img_pred, img_target)
        seg_loss *= c * h * w 
        # cal_l1_loss = nn.L1Loss()
        # recon_loss = cal_l1_loss(img_pred, img_target)
    elif model_type == 'unet':
        img_target = torch.squeeze(img_target, dim=1).long()
        cal_seg_loss = nn.CrossEntropyLoss(weight=loss_weight)
        # print('img_target size: ' + str(img_target.size()))
        # print('img_pred size: ' + str(img_pred.size()))
        seg_loss = cal_seg_loss(img_pred, img_target)
        seg_loss *= c * h * w
    return seg_loss

def get_routeRecon_loss(route_target, route_pred, model_type):
    # img_pred, lidar_pred, topdown_pred, curSpeed_pred, tarSpeed_pred, \
    #         lightState_pred, lightDist_pred, mu, logvar = results
    b, c, h, w = route_target.size()
    recon_loss = 0.0
    if model_type == 'vae':
        recon_loss = 0
        for route_pred_ele in route_pred:
            route_mu = route_pred_ele[0]
            route_sigma = route_pred_ele[1].mul(0.5).exp_()
            tmp = torch.sum(((route_target - route_mu) / route_sigma) ** 2) / b
            recon_loss += tmp
        recon_loss /= len(route_pred)
    elif model_type == 'beta-vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(route_pred, route_target)
        recon_loss *= c * h * w
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(route_pred, route_target)
        recon_loss *= c * h * w
        # cal_l1_loss = nn.L1Loss()
        # recon_loss = cal_l1_loss(route_pred, route_target)
    elif model_type == 'unet':
        cal_l1_loss = nn.L1Loss()
        # print('route_target size: ' + str(route_target.size()))
        # print('route_pred size: ' + str(route_pred.size()))
        recon_loss = cal_l1_loss(route_pred, route_target)
        recon_loss *= c * h * w
    return recon_loss

def get_lidarRecon_loss(lidar_target, lidar_pred, model_type):
    b, c, h, w = lidar_target.size()
    recon_loss = 0.0
    if model_type == 'vae':
        recon_loss = 0
        for lidar_pred_ele in lidar_pred:
            lidar_mu = lidar_pred_ele[0]
            lidar_sigma = lidar_pred_ele[1].mul(0.5).exp_()
            tmp = torch.sum(((lidar_target - lidar_mu) / lidar_sigma) ** 2) / b
            recon_loss += tmp
        recon_loss /= len(lidar_pred)
    elif model_type == 'beta-vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(lidar_pred, lidar_target)
        recon_loss *= c * h * w
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(lidar_pred, lidar_target)
        recon_loss *= c * h * w
    elif model_type == 'unet':
        cal_l1_loss = nn.L1Loss()
        # print('lidar_target size: ' + str(lidar_target.size()))
        # print('lidar_pred size: ' + str(lidar_pred.size()))
        recon_loss = cal_l1_loss(lidar_pred, lidar_target)
        recon_loss *= c * h * w
    return recon_loss

def get_topdownPred_rgb_loss(topdown_target, topdown_pred, model_type):
    b, c, h, w = topdown_target.size()
    recon_loss = 0.0
    if model_type == 'vae':
        recon_loss = 0
        for topdown_pred_ele in topdown_pred:
            topdown_mu = topdown_pred_ele[0]
            topdown_sigma = topdown_pred_ele[1].mul(0.5).exp_()
            tmp = torch.sum(((topdown_target - topdown_mu) / topdown_sigma) ** 2) / b
            recon_loss += tmp
        recon_loss /= len(topdown_pred)
    elif model_type == 'beta-vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(topdown_pred, topdown_target)
        recon_loss *= c * h * w
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        cal_mse_loss = nn.MSELoss()
        recon_loss = cal_mse_loss(topdown_pred, topdown_target)
        recon_loss *= c * h * w
    elif model_type == 'unet':
        cal_l1_loss = nn.L1Loss()
        # print('topdown_target size: ' + str(topdown_target.size()))
        # print('topdown_pred size: ' + str(topdown_pred.size()))
        recon_loss = cal_l1_loss(topdown_pred, topdown_target)
        recon_loss *= c * h * w
    return recon_loss

# def get_curSpeed_loss(cur_speed, curSpeed_pred, model_type):
#     cal_l1_loss = nn.L1Loss()
#     if model_type == 'vae':
#         output = 0
#         for curSpeed_pred_ele in curSpeed_pred:
#             # print('cur_speed size: ' + str(cur_speed.size()))
#             # print('curSpeed_pred size: ' + str(curSpeed_pred_ele.size()))
#             tmp = cal_l1_loss(curSpeed_pred_ele, cur_speed)
#             output += tmp
#         # print('cur_speed size: ' + str(cur_speed.size()))
#         # print('curSpeed_pred size: ' + str(curSpeed_pred_ele.size()))
#         output /= len(curSpeed_pred)
#     elif model_type == 'beta-vae':
#         output = cal_l1_loss(curSpeed_pred, cur_speed)
#     elif model_type == 'vanilla-vae':
#         output = cal_l1_loss(curSpeed_pred, cur_speed)
#     elif model_type == 'unet':
#         # print('cur_speed size: ' + str(cur_speed.size()))
#         # print('curSpeed_pred size: ' + str(curSpeed_pred.size()))
#         output = cal_l1_loss(curSpeed_pred, cur_speed)
#     return output

# def get_targetSpeed_loss(target_speed, tarSpeed_pred, model_type):
#     cal_l1_loss = nn.L1Loss()
#     if model_type == 'vae':
#         output = 0
#         for tarSpeed_pred_ele in tarSpeed_pred:
#             # print('target_speed size: ' + str(target_speed.size()))
#             # print('tarSpeed_pred_ele size: ' + str(tarSpeed_pred_ele.size()))
#             tmp = cal_l1_loss(tarSpeed_pred_ele, target_speed)
#             output += tmp
#         # print('target_speed size: ' + str(target_speed.size()))
#         # print('tarSpeed_pred_ele size: ' + str(tarSpeed_pred_ele.size()))
#         output /= len(tarSpeed_pred)
#     elif model_type == 'beta-vae':
#         output = cal_l1_loss(tarSpeed_pred, target_speed)
#     elif model_type == 'vanilla-vae':
#         output = cal_l1_loss(tarSpeed_pred, target_speed)
#     elif model_type == 'unet':
#         # print('target_speed size: ' + str(target_speed.size()))
#         # print('tarSpeed_pred size: ' + str(tarSpeed_pred.size()))
#         output = cal_l1_loss(tarSpeed_pred, target_speed)
#     return output

def get_lightState_loss(light_state, lightState_pred, model_type, loss_weight):
    cal_crossentropy_loss = nn.CrossEntropyLoss(weight=loss_weight)
    if model_type == 'vae':
        output = 0
        for lightState_pred_ele in lightState_pred:
            # print('light_state size: ' + str(light_state.size()))
            # print('lightState_pred_ele size: ' + str(lightState_pred_ele.size()))
            tmp = cal_crossentropy_loss(lightState_pred_ele, light_state)
            output += tmp
        # print('light_state size: ' + str(light_state.size()))
        # print('lightState_pred_ele size: ' + str(lightState_pred_ele.size()))
        output /= len(lightState_pred)
    elif model_type == 'beta-vae':
        output = cal_crossentropy_loss(lightState_pred, light_state)
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        output = cal_crossentropy_loss(lightState_pred, light_state)
        # output *= 10.0
    elif model_type == 'unet':
        # print('light_state size: ' + str(light_state.size()))
        # print('lightState_pred size: ' + str(lightState_pred.size()))
        output = cal_crossentropy_loss(lightState_pred, light_state)
    return output

def get_lightDist_loss(light_dist, lightDist_pred, model_type):
    cal_l1_loss = nn.L1Loss()
    if model_type == 'vae':
        output = 0
        for lightDist_pred_ele in lightDist_pred:
            # print('light_dist size: ' + str(light_dist.size()))
            # print('lightDist_pred_ele size: ' + str(lightStalightDist_pred_elete_pred_ele.size()))
            tmp = cal_l1_loss(lightDist_pred_ele, light_dist)
            output += tmp
        # print('light_dist size: ' + str(light_dist.size()))
        # print('lightDist_pred_ele size: ' + str(lightDist_pred_ele.size()))
        output /= len(lightDist_pred)
    elif model_type == 'beta-vae':
        output = cal_l1_loss(lightDist_pred, light_dist)
    elif model_type == 'vanilla-vae' or model_type == 'danet' or model_type == 'da_beta_vae':
        output = cal_l1_loss(lightDist_pred, light_dist)
    elif model_type == 'unet':
        # print('light_dist size: ' + str(light_dist.size()))
        # print('lightDist_pred size: ' + str(lightDist_pred.size()))
        output = cal_l1_loss(lightDist_pred, light_dist)
    return output

def get_steer_loss(steer_target, steer_pred):
    cal_mse_loss = nn.MSELoss()
    output = cal_mse_loss(steer_pred, steer_target)
    # output *= 10.0
    return output

def get_throttle_loss(throttle_target, throttle_pred):
    cal_mse_loss = nn.MSELoss()
    output = cal_mse_loss(throttle_pred, throttle_target)
    # output *= 10.0
    return output

def get_kld_loss(mu, logvar):
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # # Normalise by same number of elements as in reconstruction
    # KLD = KLD / batch_size

    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)     
    # kld_loss *= 0.1
    return kld_loss

def get_all_kld_loss(mu, logvar):
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # # Normalise by same number of elements as in reconstruction
    # KLD = KLD / batch_size
    # print("get_all_kld_loss called")
    # print("mu size: %s" % str(mu.size()))
    # print("logvar size: %s" % str(logvar.size()))
    x1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1)
    # print("x1 size: %s" % str(x1.size()))
    x2 = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    # print("x2 size: %s" % str(x2.size()))

    # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)     
    # kld_loss *= 0.1
    all_kld_loss = torch.mean(x2, dim=0)
    # print("all_kld_loss size: %s" % str(all_kld_loss.size()))

    return all_kld_loss