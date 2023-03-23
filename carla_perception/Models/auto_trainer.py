from __future__ import print_function
import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import Models.warmUpLR as warmup
from Models.experiments_builder import Experiments_Builder
from Models.cal_losses import *

from PIL import Image
import torch.nn as nn
# from apex import amp
# todo: debug
from torch.cuda import amp

from skimage import io

cityscapes_valid_classes = [7, 8, 11, 12, 13, 17, 19, 
                    20, 21, 22, 23, 24, 25, 26,
                    27, 28, 31, 32, 33]

def get_cityscapes_labels():
    return np.array([
        #[  0,   0,   0],
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]
        ])

class Auto_Trainer(Experiments_Builder):
    def __init__(self, config):
        Experiments_Builder.__init__(self, config)
    
    def decode_segmap_cv(self, label_mask, dataset, sailent_class_index, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        if dataset == 'pascal':
            print()
        elif dataset == 'cityscapes':
            n_classes = 19
            label_colours = get_cityscapes_labels()
        else:
            raise NotImplementedError

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            if ll in sailent_class_index:
                r[label_mask == ll] = label_colours[ll, 0]
                g[label_mask == ll] = label_colours[ll, 1]
                b[label_mask == ll] = label_colours[ll, 2]
            else:
                r[label_mask == ll] = 0
                g[label_mask == ll] = 0
                b[label_mask == ll] = 0
        
        r[label_mask == 255] = 0
        g[label_mask == 255] = 0
        b[label_mask == 255] = 0
        
        #    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        #   # rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
        #    #replace blue with red as opencv uses bgr
        #    rgb[:, :, 0] = r /255.0     
        #    rgb[:, :, 1] = g /255.0
        #    rgb[:, :, 2] = b /255.0
        #    

        rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
        #replace blue with red as opencv uses bgr
        rgb[:, :, 0] = b #/255.0     
        rgb[:, :, 1] = g #/255.0
        rgb[:, :, 2] = r #/255.0
        #    
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def record_imgSeg_preds(self, img_names, img_seg_list, img_pred):
        b, c, h, w = img_seg_list.size()
        img_seg_list = img_seg_list.detach().cpu().numpy()

        # out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
        seg_predictions = torch.max(img_pred, 1)[1]
        seg_pred = seg_predictions.detach().cpu().numpy()

        sailent_class_index = [0, 1, 2, 3, 4, 5, 6, 7]

        for cur_img_index in range(b):
            cur_img_name = img_names[cur_img_index]

            cur_seg_pred = np.expand_dims(seg_pred[cur_img_index], axis=0)
            pred_color = self.decode_segmap_cv(cur_seg_pred, 'cityscapes', sailent_class_index)
            pred_color = pred_color[...,::-1]

            cur_seg_tar = img_seg_list[cur_img_index]
            target_color = self.decode_segmap_cv(cur_seg_tar, 'cityscapes', sailent_class_index)
            target_color = target_color[...,::-1]

            # _combine_img = np.hstack([pred_color, target_color])
            _combine_img = pred_color
            _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')
            # im.show()
            result_dir = os.path.join(self.exp_dir, 'recon_epoch'+str(self.curr_epoch))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _combine_img.save(os.path.join(result_dir, cur_img_name+'_imgSeg_pred.png'))

            _combine_img = target_color
            _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')
            # im.show()
            result_dir = os.path.join(self.exp_dir, 'recon_epoch'+str(self.curr_epoch))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _combine_img.save(os.path.join(result_dir, cur_img_name+'_imgSeg_tar.png'))

    def record_route_preds(self, img_names, route_list, route_pred):
        b, c, h, w = route_list.size()
        route_list = route_list.detach().cpu().numpy()
        route_pred = route_pred.detach().cpu().numpy()

        for cur_img_index in range(b):
            cur_route_target = route_list[cur_img_index]
            cur_route_pred = route_pred[cur_img_index]
            cur_img_name = img_names[cur_img_index]

            # cur_route_pred = np.clip(cur_route_pred, 0.0, 1.0)
            # cur_img_target = np.transpose(cur_img_target, (1, 2, 0))
            # cur_img_target = cur_img_target * 255.0 

            # cur_route_target = np.clip(cur_route_target, 0.0, 1.0)
            # cur_img_pred = np.transpose(cur_img_pred, (1, 2, 0))
            # cur_img_pred = cur_img_pred * 255.0 
            # print('cur_route_pred size %s'% str(cur_route_pred.shape))
            # print('cur_route_target size %s'% str(cur_route_target.shape))

            # cur_route_pred = cur_route_pred[...,::-1]
            # cur_route_target = cur_route_target[...,::-1]

            # _combine_img = np.hstack([cur_route_pred * 255, cur_route_target * 255])
            _combine_img = cur_route_pred * 255
            _combine_img = np.squeeze(_combine_img, axis=0)
            _combine_img = _combine_img.swapaxes(0, 1)
            # print('_combine_img size %s'% str(_combine_img.shape))
            _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'L')
            # im.show()
            result_dir = os.path.join(self.exp_dir, 'recon_epoch'+str(self.curr_epoch))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _combine_img.save(os.path.join(result_dir, cur_img_name+'_route_pred.png'))

            _combine_img = cur_route_target * 255
            _combine_img = np.squeeze(_combine_img, axis=0)
            _combine_img = _combine_img.swapaxes(0, 1)
            # print('_combine_img size %s'% str(_combine_img.shape))
            _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'L')
            # im.show()
            result_dir = os.path.join(self.exp_dir, 'recon_epoch'+str(self.curr_epoch))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _combine_img.save(os.path.join(result_dir, cur_img_name+'_route_tar.png'))

    def record_preds(self, img_names, img_target, lidar_target, topdown_target, 
                            img_pred, lidar_pred, topdown_pred):

        b, c, h, w = img_target.size()
        img_target = img_target.detach().cpu().numpy()
        img_pred = img_pred.detach().cpu().numpy()

        if self.config.pred_lidar:
            lidar_target = lidar_target.detach().cpu().numpy()
            lidar_pred = lidar_pred.detach().cpu().numpy()

        if self.config.pred_topdown_rgb or self.config.pred_topdown_seg:
            topdown_target = topdown_target.detach().cpu().numpy()
            topdown_pred = topdown_pred.detach().cpu().numpy()

        for cur_img_index in range(b):
            cur_img_target = img_target[cur_img_index]
            cur_img_pred = img_pred[cur_img_index]
            cur_img_name = img_names[cur_img_index]

            cur_img_target = np.clip(cur_img_target, 0.0, 1.0)
            cur_img_target = np.transpose(cur_img_target, (1, 2, 0))
            cur_img_target = cur_img_target * 255.0 

            cur_img_pred = np.clip(cur_img_pred, 0.0, 1.0)
            cur_img_pred = np.transpose(cur_img_pred, (1, 2, 0))
            cur_img_pred = cur_img_pred * 255.0 

            _combine_img = np.hstack([cur_img_pred, cur_img_target])
            _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')
            # im.show()
            result_dir = os.path.join(self.exp_dir, 'recon_epoch'+str(self.curr_epoch))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            _combine_img.save(os.path.join(result_dir, cur_img_name+'_img.png'))

            ################ save lidar images ################
            if self.config.pred_lidar:
                cur_lidar_target = lidar_target[cur_img_index]
                cur_lidar_pred = lidar_pred[cur_img_index]

                cur_lidar_target = np.clip(cur_lidar_target, 0.0, 1.0)
                cur_lidar_target = np.transpose(cur_lidar_target, (1, 2, 0))
                cur_lidar_target = cur_lidar_target * 255.0

                cur_lidar_pred = np.clip(cur_lidar_pred, 0.0, 1.0)
                cur_lidar_pred = np.transpose(cur_lidar_pred, (1, 2, 0))
                cur_lidar_pred = cur_lidar_pred * 255.0 

                _combine_lidar = np.hstack([cur_lidar_pred, cur_lidar_target])
                _combine_lidar = Image.fromarray(_combine_lidar.astype('uint8'), 'RGB')
                _combine_lidar.save(os.path.join(result_dir, cur_img_name+'_lidar.png'))

            ################ save topdown images ################
            if self.config.pred_topdown_rgb or self.config.pred_topdown_seg:
                cur_topdown_target = topdown_target[cur_img_index]
                cur_topdown_pred = topdown_pred[cur_img_index]

                cur_topdown_target = np.clip(cur_topdown_target, 0.0, 1.0)
                cur_topdown_target = np.transpose(cur_topdown_target, (1, 2, 0))
                cur_topdown_target = cur_topdown_target * 255.0 

                cur_topdown_pred = np.clip(cur_topdown_pred, 0.0, 1.0)
                if self.config.pred_topdown_rgb:
                    cur_topdown_pred = np.transpose(cur_topdown_pred, (1, 2, 0))
                cur_topdown_pred = cur_topdown_pred * 255.0 

                _combine_topdown = np.hstack([cur_topdown_pred, cur_topdown_target])
                _combine_topdown = Image.fromarray(_combine_topdown.astype('uint8'), 'RGB')
                _combine_topdown.save(os.path.join(result_dir, cur_img_name+'_topdown.png'))

    def update_Autoencoder(self, total_losses):
        if self.config.pred_camera_seg:
            total_loss = total_losses['imgSeg_loss']
        else:
            total_loss = total_losses['imgRecon_loss']

        if self.config.pred_left_camera_seg:
            total_loss += total_losses['left_imgSeg_loss']
        if self.config.pred_right_camera_seg:
            total_loss += total_losses['right_imgSeg_loss']

        if self.config.pred_route:
            total_loss += 0.5 * total_losses['routeRecon_loss']

        if self.config.pred_light_state:
            total_loss += 0.1 * total_losses['lightState_loss']
        if self.config.pred_light_dist:
            total_loss += total_losses['lightDist_loss']

        if self.config.pred_lidar:
            total_loss += total_losses['lidarRecon_loss']

        if self.config.pred_topdown_rgb:
            total_loss += total_losses['topdownPred_rgb_loss']
        if self.config.pred_topdown_seg:
            total_loss += total_losses['topdownPred_seg_loss']

        if self.config.pred_bc:
            # total_loss += 0.1 * total_losses['steer_loss']
            # total_loss += 0.1 * total_losses['throttle_loss']
            total_loss += total_losses['steer_loss']
            total_loss += total_losses['throttle_loss']

        if self.model_type.find('vae') >= 0:
            total_loss += total_losses['visual_kld_loss']
            if self.config.pred_bc:
                total_loss += total_losses['bc_kld_loss']

        if self.cuda:
            total_loss = total_loss.cuda()

        ########### Clean Grad ###########
        for key in self.optimizers.keys():
                self.optimizers[key].zero_grad()

        optimizer_list = []
        for key in self.optimizers.keys():
            # self.optimizers[key].zero_grad()
            optimizer_list.append(self.optimizers[key])

        ########### Backward G ##############
        if self.cuda:
            if self.mixed_percision:
                with amp.scale_loss(total_loss, optimizer_list) as scaled_loss2:
                    scaled_loss2.backward()
            else:
                total_loss.backward()
        else:
            total_loss.backward()
        
        ########### Update G ##############
        for key in self.optimizers.keys():
            self.optimizers[key].step()

    def train_step(self, batch):
        img_input_list, img_tar_list, img_seg_list, \
        left_img_input_list, left_img_tar_list, left_img_seg_list, \
        right_img_input_list, right_img_tar_list, right_img_seg_list, \
        topdown_rgb_list, topdown_seg_list, \
        lidar_list, route_list, command, \
        speed, target_speed, \
        steer, throttle, \
        light_state, light_dist, img_name = batch

        # img_input_list, img_tar_list, topdown_rgb_list, topdown_seg_list, lidar_list, \
        # command, speed, target_speed, light_state, \
        # light_dist, img_names, loss_weight, img_seg_list, route_list = batch

        command = command - 1
        speed = speed.double()
        steer = steer.double()
        throttle = throttle.double()

        if self.cuda:
            img_input_list = img_input_list.cuda()
            img_tar_list = img_tar_list.cuda()
            if self.config.pred_camera_seg:
                img_seg_list = img_seg_list.cuda()
            
            command = command.cuda()
            speed = speed.cuda()
            target_speed = target_speed.cuda()
            light_state = light_state.cuda()
            light_dist = light_dist.cuda()
            steer = steer.cuda()
            throttle = throttle.cuda()

            ### left camera
            if self.config.in_left_camera:
                left_img_input_list = left_img_input_list.cuda()
                left_img_tar_list = left_img_tar_list.cuda()
            if self.config.pred_left_camera_seg:
                left_img_seg_list = left_img_seg_list.cuda()
            
            ### right camera
            if self.config.in_right_camera:
                right_img_input_list = right_img_input_list.cuda()
                right_img_tar_list = right_img_tar_list.cuda()
            if self.config.pred_right_camera_seg:
                right_img_seg_list = right_img_seg_list.cuda()

            if self.config.pred_topdown_rgb:
                topdown_rgb_list = topdown_rgb_list.cuda()
            if self.config.pred_topdown_seg:
                topdown_seg_list = topdown_seg_list.cuda()

            if self.config.pred_lidar or self.config.in_lidar:
                lidar_list = lidar_list.cuda()

            if self.config.in_route:
                route_list = route_list.cuda()

            self.config.seg_class_weight = self.config.seg_class_weight.cuda()
            self.config.command_class_weight = self.config.command_class_weight.cuda()
            self.config.light_class_weight = self.config.light_class_weight.cuda()

        b, _, c, h, w = img_input_list.size()
        img_input_list = img_input_list.view(b, -1, h, w)
        img_tar_list = img_tar_list.view(b, -1, h, w)
        if self.config.pred_camera_seg:
            img_seg_list = img_seg_list.view(b, -1, h, w)

        cur_speed = speed.view(b, -1)
        target_speed = target_speed.view(b, -1)
        # light_state = light_state.view(b, -1)
        light_dist = light_dist.view(b, -1)
        steer = steer.view(b, -1)
        throttle = throttle.view(b, -1)

        if self.config.in_left_camera:
            left_img_input_list = left_img_input_list.view(b, -1, h, w)
            left_img_tar_list = left_img_tar_list.view(b, -1, h, w)
        if self.config.pred_left_camera_seg:
            left_img_seg_list = left_img_seg_list.view(b, -1, h, w)

        if self.config.in_right_camera:
            right_img_input_list = right_img_input_list.view(b, -1, h, w)
            right_img_tar_list = right_img_tar_list.view(b, -1, h, w)
        if self.config.pred_right_camera_seg:
            right_img_seg_list = right_img_seg_list.view(b, -1, h, w)

        if self.config.pred_topdown_rgb:
            topdown_rgb_list = topdown_rgb_list.view(b, -1, h, w)
        if self.config.pred_topdown_seg:
            topdown_seg_list = topdown_seg_list.view(b, -1, h, w)
        
        if self.config.pred_lidar or self.config.in_lidar:
            lidar_list = lidar_list.view(b, -1, h, w)
        
        if self.config.in_route:
            route_list = route_list.view(b, -1, h, w)
        
        if self.config.in_speed:
            cur_speed = cur_speed.unsqueeze(2)
            cur_speed = cur_speed.unsqueeze(3)
            cur_speed = cur_speed.repeat(1, 1, h, w)

        input_list = img_input_list
        if self.config.in_left_camera:
            input_list = torch.cat((input_list, left_img_input_list), dim=1)
        if self.config.in_right_camera:
            input_list = torch.cat((input_list, right_img_input_list), dim=1)
        if self.config.in_lidar:
            input_list = torch.cat((input_list, lidar_list), dim=1)
        if self.config.in_route:
            input_list = torch.cat((input_list, route_list), dim=1)
        if self.config.in_speed:
            input_list = torch.cat((input_list, cur_speed.float()), dim=1)
        if self.config.in_bc_speed:
            bc_speed = speed.view(b, -1).float()
        else:
            bc_speed = None

        if self.model_type.find('vae') >= 0:
            lightState_pred, \
            lightDist_pred, \
            img_pred, \
            lidar_pred, \
            topdown_pred, \
            route_pred, \
            left_img_pred, \
            right_img_pred,\
            steer_pred, throttle_pred,\
            att_visual_mu, att_visual_logvar, \
            att_bc_mu, att_bc_logvar = self.networks['autoencoder'](input_list, bc_speed)
            # img_pred, lidar_pred, topdown_pred, lightState_pred, \
            # lightDist_pred, route_pred, mu, logvar = self.networks['autoencoder'](input_list)
        else:
            lightState_pred, \
            lightDist_pred, \
            img_pred, \
            lidar_pred, \
            topdown_pred, \
            route_pred, \
            left_img_pred, \
            right_img_pred,\
            steer_pred, throttle_pred = self.networks['autoencoder'](input_list, bc_speed)
            # img_pred, lidar_pred, topdown_pred, lightState_pred, \
            # lightDist_pred = self.networks['autoencoder'](input_list)

        # results = self.networks['autoencoder'](rgbimg_list, lidar_list)

        total_losses = dict()

        # for loss_name, loss_func_name in self.config.loss_funcs.item():
        #     total_losses[loss_name] = eval(loss_func_name)(results, self.model_type)

        if self.model_type.find('vae') >= 0:
            visual_kld_loss = get_kld_loss(att_visual_mu, att_visual_logvar)
            if att_bc_mu is not None and att_bc_logvar is not None:
                bc_kld_loss = get_kld_loss(att_bc_mu, att_bc_logvar)
            # kld_loss = visual_kld_loss + bc_kld_loss

            kld_weight = b * self.config.world_size / self.config.num_data_img

            if kld_weight == 0.0:
                kld_weight = b / self.config.num_data_img
                
            # kld_weight = 1.0
            # if self.model_type == 'vanilla-vae':
            #     kld_loss = kld_weight * kld_loss
            # elif self.model_type == 'old-vae':
            #     kld_loss = kld_weight * kld_loss
            # elif self.model_type == 'oldv2-vae':
            #     kld_loss = kld_weight * kld_loss
            if self.model_type == 'da_beta_vae':
                visual_kld_loss = self.config.networks['autoencoder']['beta'] * kld_weight * visual_kld_loss
                if att_bc_mu is not None and att_bc_logvar is not None:
                    bc_kld_loss = self.config.networks['autoencoder']['beta'] * kld_weight * bc_kld_loss

            total_losses['visual_kld_loss'] = visual_kld_loss
            if att_bc_mu is not None and att_bc_logvar is not None:
                total_losses['bc_kld_loss'] = bc_kld_loss

        if self.config.pred_camera_seg:
            imgSeg_loss = get_imgSeg_loss(img_seg_list, img_pred, self.model_type, 
                                self.config.seg_class_weight)
            total_losses['imgSeg_loss'] = imgSeg_loss
        else:
            img_target = img_tar_list[:, -3:, :, :]
            imgRecon_loss = get_imgRecon_loss(img_target, img_pred, self.model_type)
            total_losses['imgRecon_loss'] = imgRecon_loss

        if self.config.pred_left_camera_seg:
            left_imgSeg_loss = get_imgSeg_loss(left_img_seg_list, left_img_pred, 
                                    self.model_type, self.config.seg_class_weight)
            total_losses['left_imgSeg_loss'] = left_imgSeg_loss

        if self.config.pred_right_camera_seg:
            right_imgSeg_loss = get_imgSeg_loss(right_img_seg_list, right_img_pred,
                                    self.model_type, self.config.seg_class_weight)
            total_losses['right_imgSeg_loss'] = right_imgSeg_loss

        if self.config.pred_route:
            route_target = route_list[:, -1:, :, :]
            routeRecon_loss = get_routeRecon_loss(route_target, route_pred, self.model_type)
            total_losses['routeRecon_loss'] = routeRecon_loss

        if self.config.pred_lidar:
            lidar_target = lidar_list[:, -3:, :, :]
            lidarRecon_loss = get_lidarRecon_loss(lidar_target, lidar_pred, self.model_type)
            total_losses['lidarRecon_loss'] = lidarRecon_loss

        if self.config.pred_topdown_rgb:
            topdown_target = topdown_rgb_list[:, -3:, :, :]
            topdownPred_loss = get_topdownPred_rgb_loss(topdown_target, topdown_pred, self.model_type)
            total_losses['topdownPred_rgb_loss'] = topdownPred_loss

        if self.config.pred_topdown_seg:
            topdown_target = topdown_seg_list[:, -3:, :, :]
            # topdownPred_loss = get_topdownPred_rgb_loss(topdown_target, topdown_pred, self.model_type)
            topdownPred_loss = 0.0
            total_losses['topdownPred_seg_loss'] = topdownPred_loss

        # if self.config.pred_curSpeed:
        #     curSpeed_loss = get_curSpeed_loss(cur_speed, curSpeed_pred, self.model_type)
        #     total_losses['curSpeed_loss'] = curSpeed_loss

        # if self.config.pred_tarSpeed:
        #     targetSpeed_loss = get_targetSpeed_loss(target_speed, tarSpeed_pred, self.model_type)
        #     total_losses['targetSpeed_loss'] = targetSpeed_loss

        if self.config.pred_light_state:
            lightState_loss = get_lightState_loss(light_state, lightState_pred, 
                                    self.model_type, self.config.light_class_weight)
            total_losses['lightState_loss'] = lightState_loss
        
        if self.config.pred_light_dist:
            lightDist_loss = get_lightDist_loss(light_dist, lightDist_pred, self.model_type)
            total_losses['lightDist_loss'] = lightDist_loss

        if self.config.pred_bc:
            steer_pred = steer_pred.double().view(b, -1)
            steer_loss = get_steer_loss(steer, steer_pred)
            total_losses['steer_loss'] = steer_loss

            throttle_pred = throttle_pred.double().view(b, -1)
            throttle_loss = get_throttle_loss(throttle, throttle_pred)
            total_losses['throttle_loss'] = throttle_loss

        ########### Update Networks ##############
        self.update_Autoencoder(total_losses)

        for key, val in total_losses.items():
            total_losses[key] = val.detach().cpu().numpy()

        return total_losses

    def solve(self, train_dataloader, train_sampler):
        self.train_dataloader = train_dataloader

        start_epoch = self.curr_epoch

        for key in self.optimizers.keys():
            iter_per_epoch = len(train_dataloader)
            self.warmup_schedulers[key] = warmup.WarmUpLR(
                self.optimizers[key], iter_per_epoch * self.config.warm)

        #print('3--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

        self.init_record_of_best_model()

        for self.curr_epoch in range(start_epoch, self.max_num_epochs + 1): #120
            self.logger.info('Training epoch [%3d / %3d]' %
                             (self.curr_epoch, self.max_num_epochs))

            #print('4--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))
            if self.cuda and train_sampler is not None:
                train_sampler.set_epoch(self.curr_epoch)

            self.run_train_epoch(train_dataloader)

            if self.curr_epoch > self.config.warm:
                self.adjust_learning_rates(self.curr_epoch)

            #print('23--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

            if self.curr_epoch % self.config.save_interval == 0:
                self.save_checkpoint(self.curr_epoch)

        if self.config.local_rank == 1:
            self.writer.close()

    def run_train_epoch(self, train_dataloader):
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))

        for key, network in self.networks.items():
            network.train()

        #print('5--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

        batch_num = len(train_dataloader)

        for idx, batch in enumerate(train_dataloader):
            #print('6--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

            #print('7--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))
            train_stats_this = self.train_step(batch)

            if self.curr_epoch <= self.config.warm:
                for key in self.warmup_schedulers.keys():
                    self.warmup_schedulers[key].step()
            #print('20--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))
            train_stats_result = self.dict2str(train_stats_this)
            #print('21--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

            self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' %
                             (self.curr_epoch, idx + 1, batch_num,
                              train_stats_result))
            is_Training = True
            if self.config.local_rank == 1:
                cur_idx = (self.curr_epoch - 1) * batch_num + idx + 1
                print("Rank %d writes idx %d loss into writer" % (self.config.local_rank, cur_idx))
                self.dict2writer(train_stats_this,
                                (self.curr_epoch - 1) * batch_num + idx + 1,
                                is_Training)
            #print('22--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))


    def test_route(self, test_dataloader):
        self.logger.info('Testing: %s' % os.path.basename(self.exp_dir))

        for key, network in self.networks.items():
            network.eval()

        num_sample = len(test_dataloader)

        total_losses = dict()
        total_losses['visual_kld_loss'] = 0.0
        total_losses['bc_kld_loss'] = 0.0
        total_losses['imgRecon_loss'] = 0.0
        total_losses['imgSeg_loss'] = 0.0
        total_losses['imgSeg_accuracy'] = 0
        total_losses['left_imgSeg_loss'] = 0.0
        total_losses['left_imgSeg_accuracy'] = 0
        total_losses['right_imgSeg_loss'] = 0.0
        total_losses['right_imgSeg_accuracy'] = 0
        total_losses['routeRecon_loss'] = 0.0
        total_losses['lidarRecon_loss'] = 0.0
        total_losses['topdownPred_rgb_loss'] = 0.0
        total_losses['topdownPred_seg_loss'] = 0.0
        # total_losses['curSpeed_loss'] = 0.0
        # total_losses['targetSpeed_loss'] = 0.0
        total_losses['lightState_loss'] = 0.0
        total_losses['lightState_accuracy'] = 0
        total_losses['lightDist_loss'] = 0.0
        total_losses['steer_loss'] = 0.0
        total_losses['throttle_loss'] = 0.0

        avg_visual_all_kld_loss = np.zeros(self.config.networks['autoencoder']['z_dims'])
        total_losses['avg_visual_all_kld_loss'] = avg_visual_all_kld_loss

        img_target_max = -1
        lidar_target_max = -1
        topdown_rgb_target_max = -1

        img_target_min = 2.0
        lidar_target_min = 2.0
        topdown_rgb_target_min = 2.0

        for idx, batch in enumerate(test_dataloader):
            print("begin %d" % idx)
            img_input_list, img_tar_list, img_seg_list, \
            left_img_input_list, left_img_tar_list, left_img_seg_list, \
            right_img_input_list, right_img_tar_list, right_img_seg_list, \
            topdown_rgb_list, topdown_seg_list, \
            lidar_list, route_list, command, \
            speed, target_speed, \
            steer, throttle, \
            light_state, light_dist, img_names = batch

            command = command - 1
            speed = speed.double()
            steer = steer.double()
            throttle = throttle.double()

            if self.cuda:
                img_input_list = img_input_list.cuda()
                img_tar_list = img_tar_list.cuda()
                if self.config.pred_camera_seg:
                    img_seg_list = img_seg_list.cuda()
                
                command = command.cuda()
                speed = speed.cuda()
                target_speed = target_speed.cuda()
                light_state = light_state.cuda()
                light_dist = light_dist.cuda()
                steer = steer.cuda()
                throttle = throttle.cuda()

                ### left camera
                if self.config.in_left_camera:
                    left_img_input_list = left_img_input_list.cuda()
                    left_img_tar_list = left_img_tar_list.cuda()
                if self.config.pred_left_camera_seg:
                    left_img_seg_list = left_img_seg_list.cuda()
                
                ### right camera
                if self.config.in_right_camera:
                    right_img_input_list = right_img_input_list.cuda()
                    right_img_tar_list = right_img_tar_list.cuda()
                if self.config.pred_right_camera_seg:
                    right_img_seg_list = right_img_seg_list.cuda()

                if self.config.pred_topdown_rgb:
                    topdown_rgb_list = topdown_rgb_list.cuda()
                if self.config.pred_topdown_seg:
                    topdown_seg_list = topdown_seg_list.cuda()

                if self.config.pred_lidar or self.config.in_lidar:
                    lidar_list = lidar_list.cuda()

                if self.config.in_route:
                    route_list = route_list.cuda()

                self.config.seg_class_weight = self.config.seg_class_weight.cuda()
                self.config.command_class_weight = self.config.command_class_weight.cuda()
                self.config.light_class_weight = self.config.light_class_weight.cuda()

            b, _, c, h, w = img_input_list.size()
            img_input_list = img_input_list.view(b, -1, h, w)
            img_tar_list = img_tar_list.view(b, -1, h, w)
            if self.config.pred_camera_seg:
                img_seg_list = img_seg_list.view(b, -1, h, w)

            cur_speed = speed.view(b, -1)
            target_speed = target_speed.view(b, -1)
            # light_state = light_state.view(b, -1)
            light_dist = light_dist.view(b, -1)
            steer = steer.view(b, -1)
            throttle = throttle.view(b, -1)

            if self.config.in_left_camera:
                left_img_input_list = left_img_input_list.view(b, -1, h, w)
                left_img_tar_list = left_img_tar_list.view(b, -1, h, w)
            if self.config.pred_left_camera_seg:
                left_img_seg_list = left_img_seg_list.view(b, -1, h, w)

            if self.config.in_right_camera:
                right_img_input_list = right_img_input_list.view(b, -1, h, w)
                right_img_tar_list = right_img_tar_list.view(b, -1, h, w)
            if self.config.pred_right_camera_seg:
                right_img_seg_list = right_img_seg_list.view(b, -1, h, w)

            if self.config.pred_topdown_rgb:
                topdown_rgb_list = topdown_rgb_list.view(b, -1, h, w)
            if self.config.pred_topdown_seg:
                topdown_seg_list = topdown_seg_list.view(b, -1, h, w)
            
            if self.config.pred_lidar or self.config.in_lidar:
                lidar_list = lidar_list.view(b, -1, h, w)
            
            if self.config.in_route:
                route_list = route_list.view(b, -1, h, w)

            if self.config.in_speed:
                cur_speed = cur_speed.unsqueeze(2)
                cur_speed = cur_speed.unsqueeze(3)
                cur_speed = cur_speed.repeat(1, 1, h, w)

            input_list = img_input_list
            if self.config.in_left_camera:
                input_list = torch.cat((input_list, left_img_input_list), dim=1)
            if self.config.in_right_camera:
                input_list = torch.cat((input_list, right_img_input_list), dim=1)
            if self.config.in_lidar:
                input_list = torch.cat((input_list, lidar_list), dim=1)
            if self.config.in_route:
                input_list = torch.cat((input_list, route_list), dim=1)
            if self.config.in_speed:
                input_list = torch.cat((input_list, cur_speed.float()), dim=1)
            if self.config.in_bc_speed:
                bc_speed = speed.view(b, -1).float()
            else:
                bc_speed = None

            if self.model_type.find('vae') >= 0:
                lightState_pred, \
                lightDist_pred, \
                img_pred, \
                lidar_pred, \
                topdown_pred, \
                route_pred, \
                left_img_pred, \
                right_img_pred,\
                steer_pred, throttle_pred,\
                att_visual_mu, att_visual_logvar, \
                att_bc_mu, att_bc_logvar = self.networks['autoencoder'](input_list, bc_speed)
            else:
                lightState_pred, \
                lightDist_pred, \
                img_pred, \
                lidar_pred, \
                topdown_pred, \
                route_pred, \
                left_img_pred, \
                right_img_pred,\
                steer_pred, throttle_pred = self.networks['autoencoder'](input_list, bc_speed)
            
            if self.model_type.find('vae') >= 0:
                # kld_loss = get_kld_loss(mu, logvar)

                visual_kld_loss = get_kld_loss(att_visual_mu, att_visual_logvar)
                visual_all_kld_loss = get_all_kld_loss(att_visual_mu, att_visual_logvar)
                if att_bc_mu is not None and att_bc_logvar is not None:
                    bc_kld_loss = get_kld_loss(att_bc_mu, att_bc_logvar)

                # kld_weight = b / self.config.num_data_img
                kld_weight = 1.0
                if self.model_type == 'vanilla-vae':
                    visual_kld_loss = kld_weight * visual_kld_loss
                if self.model_type == 'da_beta_vae':
                    visual_kld_loss = self.config.networks['autoencoder']['beta'] * kld_weight * visual_kld_loss
                    if att_bc_mu is not None and att_bc_logvar is not None:
                        bc_kld_loss = self.config.networks['autoencoder']['beta'] * kld_weight * bc_kld_loss

                # elif self.model_type == 'old-vae':
                #     kld_loss = kld_weight * kld_loss
                # elif self.model_type == 'oldv2-vae':
                #     kld_loss = kld_weight * kld_loss
                # elif self.model_type == 'beta-vae':
                #     kld_loss = self.config.networks['autoencoder']['beta'] * kld_weight * kld_loss

                total_losses['avg_visual_all_kld_loss'] += visual_all_kld_loss.detach().cpu().numpy() 
                total_losses['visual_kld_loss'] += visual_kld_loss.detach().cpu().numpy() 
                if att_bc_mu is not None and att_bc_logvar is not None:
                    total_losses['bc_kld_loss'] += bc_kld_loss.detach().cpu().numpy() 
            
            img_target = None
            if self.config.pred_camera_seg:
                imgSeg_loss = get_imgSeg_loss(img_seg_list, img_pred, self.model_type, self.config.seg_class_weight)
                total_losses['imgSeg_loss'] += imgSeg_loss.detach().cpu().numpy()

                _, img_pred_indices = torch.max(img_pred.data, 1)
                total_losses['imgSeg_accuracy'] += (img_pred_indices == img_seg_list).detach().cpu().sum().item() / (h*w)
            else: 
                img_target = img_tar_list[:, -3:, :, :]
                cur_img_target_max = torch.max(img_target).detach().cpu().item()
                if cur_img_target_max > img_target_max:
                    img_target_max = cur_img_target_max
                cur_img_target_min = torch.min(img_target).detach().cpu().item()
                if cur_img_target_min < img_target_min:
                    img_target_min = cur_img_target_min
                imgRecon_loss = get_imgRecon_loss(img_target, img_pred, self.model_type)
                total_losses['imgRecon_loss'] += imgRecon_loss.detach().cpu().numpy()

            if self.config.pred_left_camera_seg:
                left_imgSeg_loss = get_imgSeg_loss(left_img_seg_list, left_img_pred, self.model_type, self.config.seg_class_weight)
                total_losses['left_imgSeg_loss'] += left_imgSeg_loss.detach().cpu().numpy()

                _, left_img_pred_indices = torch.max(left_img_pred.data, 1)
                total_losses['left_imgSeg_accuracy'] += (left_img_pred_indices == left_img_seg_list).detach().cpu().sum().item() / (h*w)

            if self.config.pred_right_camera_seg:
                right_imgSeg_loss = get_imgSeg_loss(right_img_seg_list, right_img_pred, self.model_type, self.config.seg_class_weight)
                total_losses['right_imgSeg_loss'] += right_imgSeg_loss.detach().cpu().numpy()

                _, right_img_pred_indices = torch.max(right_img_pred.data, 1)
                total_losses['right_imgSeg_accuracy'] += (right_img_pred_indices == right_img_seg_list).detach().cpu().sum().item() / (h*w)

            route_target = None
            if self.config.pred_route:
                route_target = route_list[:, -1:, :, :]
                routeRecon_loss = get_routeRecon_loss(route_target, route_pred, self.model_type)
                total_losses['routeRecon_loss'] += routeRecon_loss.detach().cpu().numpy()

            lidar_target = None
            if self.config.pred_lidar:
                lidar_target = lidar_list[:, -3:, :, :]
                cur_lidar_target_max = torch.max(lidar_target).detach().cpu().item()
                if cur_lidar_target_max > lidar_target_max:
                    lidar_target_max = cur_lidar_target_max
                cur_lidar_target_min = torch.min(lidar_target).detach().cpu().item()
                if cur_lidar_target_min < lidar_target_min:
                    lidar_target_min = cur_lidar_target_min
                lidarRecon_loss = get_lidarRecon_loss(lidar_target, lidar_pred, self.model_type)
                total_losses['lidarRecon_loss'] += lidarRecon_loss.detach().cpu().numpy()

            topdown_target = None
            if self.config.pred_topdown_rgb:
                topdown_target = topdown_rgb_list[:, -3:, :, :]
                cur_topdown_target_max = torch.max(topdown_target).detach().cpu().item()
                if cur_topdown_target_max > topdown_rgb_target_max:
                    topdown_rgb_target_max = cur_topdown_target_max
                cur_topdown_target_min = torch.min(topdown_target).detach().cpu().item()
                if cur_topdown_target_min < topdown_rgb_target_min:
                    topdown_rgb_target_min = cur_topdown_target_min
                topdownPred_loss = get_topdownPred_rgb_loss(topdown_target, topdown_pred, self.model_type)
                total_losses['topdownPred_rgb_loss'] += topdownPred_loss.detach().cpu().numpy()

            if self.config.pred_topdown_seg:
                topdown_target = topdown_seg_list[:, -3:, :, :]
                # topdownPred_loss = get_topdownPred_rgb_loss(topdown_target, topdown_pred, self.model_type)
                topdownPred_loss = 0.0
                total_losses['topdownPred_seg_loss'] += topdownPred_loss.detach().cpu().numpy()

            # if self.config.pred_curSpeed:
            #     curSpeed_loss = get_curSpeed_loss(cur_speed, curSpeed_pred, self.model_type)
            #     total_losses['curSpeed_loss'] += curSpeed_loss.detach().cpu().numpy()

            # if self.config.pred_tarSpeed:
            #     targetSpeed_loss = get_targetSpeed_loss(target_speed, tarSpeed_pred, self.model_type)
            #     total_losses['targetSpeed_loss'] += targetSpeed_loss.detach().cpu().numpy()

            if self.config.pred_light_state:
                lightState_loss = get_lightState_loss(light_state, lightState_pred, self.model_type, self.config.light_class_weight)
                total_losses['lightState_loss'] += lightState_loss.detach().cpu().numpy()

                if self.model_type == 'beta-vae':
                    _, lightState_pred_indices = torch.max(lightState_pred.data, 1)
                elif self.model_type == 'vanilla-vae' or self.model_type == 'danet':
                    _, lightState_pred_indices = torch.max(lightState_pred.data, 1)
                elif self.model_type == 'unet':
                    _, lightState_pred_indices = torch.max(lightState_pred.data, 1)
                elif self.model_type == 'da_beta_vae':
                    _, lightState_pred_indices = torch.max(lightState_pred.data, 1)
                # print("lightState_pred size: %s" % str(lightState_pred.size()))
                # print("lightState_pred_indices size: %s" % str(lightState_pred_indices.size()))
                # z = (lightState_pred_indices == light_state)
                # print("z size: %s" % str(z.size()))
                # a = (lightState_pred_indices == light_state).detach().cpu().sum().item()
                # print("a: %d" % a)
                total_losses['lightState_accuracy'] += (lightState_pred_indices == light_state).detach().cpu().sum().item()
            
            if self.config.pred_light_dist:
                lightDist_loss = get_lightDist_loss(light_dist, lightDist_pred, self.model_type)
                total_losses['lightDist_loss'] += lightDist_loss.detach().cpu().numpy()

            if self.config.pred_bc:
                steer_pred = steer_pred.view(b, -1)
                steer_loss = get_steer_loss(steer, steer_pred)
                total_losses['steer_loss'] += steer_loss.detach().cpu().numpy()

                throttle_pred = throttle_pred.view(b, -1)
                throttle_loss = get_throttle_loss(throttle, throttle_pred)
                total_losses['throttle_loss'] += throttle_loss.detach().cpu().numpy()

            # if self.config.local_rank == 0:
            #     self.record_preds(img_names, command, target, refine_pred)
            if idx % 5 == 0:
                if self.model_type == 'beta-vae':
                    self.record_preds(img_names, img_target, lidar_target, topdown_target, 
                                img_pred, lidar_pred, topdown_pred)
                elif self.model_type == 'vanilla-vae' or self.model_type == 'danet':
                    if self.config.pred_camera_seg:
                        self.record_imgSeg_preds(img_names, img_seg_list, img_pred)
                    if self.config.pred_left_camera_seg:
                        self.record_imgSeg_preds(img_names, left_img_seg_list, left_img_pred)
                    if self.config.pred_right_camera_seg:
                        self.record_imgSeg_preds(img_names, right_img_seg_list, right_img_pred)
                    if self.config.pred_route:
                        self.record_route_preds(img_names, route_list, route_pred)
                    # else:
                    #     self.record_preds(img_names, img_target, lidar_target, topdown_target, 
                    #                 img_pred, lidar_pred, topdown_pred)

                elif self.model_type == 'unet':
                    self.record_preds(img_names, img_target, lidar_target, topdown_target, 
                                img_pred, lidar_pred, topdown_pred)

        print("img_target_min: %f" % img_target_min)
        print("img_target_max: %f" % img_target_max)
        print("lidar_target_min: %f" % lidar_target_min)
        print("lidar_target_max: %f" % lidar_target_max)
        print("topdown_rgb_target_min: %f" % topdown_rgb_target_min)
        print("topdown_rgb_target_max: %f" % topdown_rgb_target_max)

        print("Total number of samples: %s" % num_sample)
        for key, val in total_losses.items():
            # total_losses[key] = val.detach().cpu().numpy()
            val /= num_sample
            if key == 'avg_visual_all_kld_loss':
                print(np.sort(val))
                print(np.argsort(val))
                np_name = "./avg_visual_all_kld_loss_" + self.config.exp_suffix + ".npy"
                np.save(np_name, val)
                continue

            if key.find('accuracy') >= 0:
                print("%s has accuracy: %f %%" % (key, val * 100))
            else:
                print("%s has loss: %f" % (key, val))
        