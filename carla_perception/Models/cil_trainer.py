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
from apex import amp
from skimage import io

class Cil_Trainer(Experiments_Builder):
    def __init__(self, config):
        Experiments_Builder.__init__(self, config)
    
    def update_CilNet(self, total_losses):
        total_loss = total_losses['uncertain_loss']

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
        img, speed, target, mask = batch

        if self.cuda:
            img = img.cuda()
            speed = speed.cuda()
            target = target.cuda()
            mask = mask.cuda()
        
        branches_out, pred_speed, log_var_control, log_var_speed = self.networks['cil_net'](img, speed)

        total_losses = dict()

        branch_square = torch.pow((branches_out - target), 2)
        branch_loss = torch.mean((torch.exp(-log_var_control)
                                    * branch_square
                                    + log_var_control) * 0.5 * mask) * 4

        speed_square = torch.pow((pred_speed - speed), 2)
        speed_loss = torch.mean((torch.exp(-log_var_speed)
                                    * speed_square
                                    + log_var_speed) * 0.5)

        uncertain_loss = self.config.branch_weight * branch_loss + self.config.speed_weight * speed_loss

        total_losses['branch_loss'] = branch_loss
        total_losses['speed_loss'] = speed_loss
        total_losses['uncertain_loss'] = uncertain_loss

        ########### Update Networks ##############
        self.update_CilNet(total_losses)

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
        total_losses['branch_loss'] = 0.0
        total_losses['speed_loss'] = 0.0
        total_losses['uncertain_loss'] = 0.0

        for idx, batch in enumerate(test_dataloader):
            print("begin %d" % idx)
            img, speed, target, mask = batch

            if self.cuda:
                img = img.cuda()
                speed = speed.cuda()
                target = target.cuda()
                mask = mask.cuda()
            
            branches_out, pred_speed, log_var_control, log_var_speed = self.networks['cil_net'](img, speed)

            branch_square = torch.pow((branches_out - target), 2)
            branch_loss = torch.mean((torch.exp(-log_var_control)
                                        * branch_square
                                        + log_var_control) * 0.5 * mask) * 4

            speed_square = torch.pow((pred_speed - speed), 2)
            speed_loss = torch.mean((torch.exp(-log_var_speed)
                                        * speed_square
                                        + log_var_speed) * 0.5)

            uncertain_loss = self.config.branch_weight * branch_loss + self.config.speed_weight * speed_loss

            total_losses['branch_loss'] += branch_loss
            total_losses['speed_loss'] += speed_loss
            total_losses['uncertain_loss'] += uncertain_loss

        print("Total number of samples: %s" % num_sample)
        for key, val in total_losses.items():
            # total_losses[key] = val.detach().cpu().numpy()
            val /= num_sample
            if key.find('accuracy') >= 0:
                print("%s has accuracy: %f %%" % (key, val * 100))
            else:
                print("%s has loss: %f" % (key, val))
        