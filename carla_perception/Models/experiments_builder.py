from __future__ import print_function
import os
import os.path

import torch
import torch.nn as nn
import torch.optim
import numpy as np

import datetime
import logging

import sys

from Models.parallel import DataParallelModel, DataParallelCriterion

sys.path.append('/data2/wk/carla')

from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
# todo: debug
from torch.cuda import amp

class Experiments_Builder(object):
    def __init__(self, config):
        # torch.manual_seed(0)

        self.config = config
        self.exp_dir = self.config.exp_dir
        if self.config.local_rank == 1:
            self.writer = SummaryWriter(self.exp_dir)
        self.max_num_epochs = self.config.max_num_epochs

        self.set_experiment_dir()
        self.set_log_file_handler()

        self.logger.info('Algorithm options %s' % self.dict2str(self.config.__dict__))

        self.curr_epoch = 1

        self.cuda = False
        self.multi_gpu = False

        if self.config.phase == 'train':
            self.config.gpu_ids = self.config.train_gpu_ids
        else:
            self.config.gpu_ids = self.config.test_gpu_ids

        if -1 not in self.config.gpu_ids and torch.cuda.is_available():  # enable cuda
            print("Using CUDA: " + str(torch.cuda.is_available()))
            self.cuda = True
            if len(self.config.gpu_ids) > 1:
                self.multi_gpu = True

        if self.cuda:
            if self.multi_gpu:
                torch.cuda.set_device(self.config.local_rank)
        
        self.mixed_percision = self.config.mixed_percision

        self.networks = dict()
        self.optimizers = dict()
        self.criterions = dict()
        self.lr_schedulers = dict()
        self.warmup_schedulers = dict()

        if 'autoencoder' in self.config.networks.keys():
            self.model_type = self.config.networks['autoencoder']['model_name']
        self.init_all_networks()

        for key in self.networks.keys():
            if self.cuda:
                if self.multi_gpu:
                    # if key != 'semantic_seg':
                    self.networks[key] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.networks[key])
                    self.networks[key] = self.networks[key].cuda(self.config.local_rank)
                else:
                    self.networks[key] = self.networks[key].cuda()
            
        self.init_all_optimizers()
        
        if self.cuda and self.multi_gpu:
            for key in self.networks.keys():
                if self.mixed_percision:
                    if key in self.optimizers.keys():
                        self.networks[key], self.optimizers[key] = amp.initialize(self.networks[key], self.optimizers[key],
                                                                    opt_level='O2')
                    else:
                        self.networks[key] = amp.initialize(self.networks[key], opt_level='O2')
            
                if key == 'semantic_seg':
                    continue
                self.networks[key] = nn.parallel.DistributedDataParallel(self.networks[key],
                                        device_ids=[self.config.local_rank],
                                        find_unused_parameters=True)

        # if self.cuda: 
        #     self.load_to_gpu()

        self.init_all_criterions()

        if self.cuda:
            if self.multi_gpu:
                for key in self.criterions.keys():
                        # self.criterions[key] = DataParallelCriterion(self.criterions[key]) 
                        self.criterions[key] = self.criterions[key].cuda(self.config.local_rank)
            else:
                for key in self.criterions.keys():
                        # self.criterions[key] = DataParallelCriterion(self.criterions[key]) 
                        self.criterions[key] = self.criterions[key].cuda()

        self.metric1 = self.config.metric1

        if self.config.phase == 'test' and self.config.load_epoch != -1:
            self.curr_epoch = self.config.load_epoch

    def set_experiment_dir(self):
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        now_str = datetime.datetime.now().__str__().replace(' ','_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

    def init_all_networks(self):
        networks_defs = self.config.networks

        for key, val in networks_defs.items():
            if val is None:
                continue

            self.logger.info('Set network %s' % key)

            self.init_network(key, val)

        if 'semantic_seg' in self.networks.keys():
            for param in self.networks['semantic_seg'].parameters():
                param.requires_grad = False

    def init_weights(self, net):
        """the weights of conv layer and fully connected layers
        are both initilized with Xavier algorithm, In particular,
        we set the parameters to random values uniformly drawn from [-a, a]
        where a = sqrt(6 * (din + dout)), for batch normalization
        layers, y=1, b=0, all bias initialized to 0.
        """
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out',
                # nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return net

    def init_network(self, key, val):
        self.logger.info('==> Initiliaze network %s with opts: %s' %
                         (key, val))

        if val['net_name'] == 'autoencoder':
            if val['model_name'] == 'oldv2-vae':
                from Networks.oldv2_vae import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'old-vae':
                from Networks.old_vae import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'vanilla-vae':
                from Networks.vanilla_vae import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'beta-vae':
                from Networks.beta_vae import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'unet':
                from Networks.unet import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'rcnn-unet':
                from Networks.rcnn_unet import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'att-unet':
                from Networks.att_unet import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'rcnn-attunet':
                from Networks.rcnn_attunet import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'nested-unet':
                from Networks.nested_unet import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'danet':
                from Networks.danet import get_model
                self.networks[key] = get_model(val)

            elif val['model_name'] == 'da_beta_vae':
                from Networks.da_beta_vae import get_model
                self.networks[key] = get_model(val)

        elif val['net_name'] == 'cil_net':
            from Networks.cil_net import get_model
            self.networks[key] = get_model(val)
        elif val['net_name'] == 'cilrs_net':
            from Networks.cilrs_net import get_model
            self.networks[key] = get_model(val)
        else:
            raise ValueError('Not supported or recognized net_name', val['net_name'])

        if key in self.networks.keys():
            if key != 'semantic_seg':
                self.logger.info('==> Initilized %s with Xavier' % key)
                self.networks[key] = self.init_weights(self.networks[key])

            if val['pretrained_path'] is not None:
                self.network_load_pretrained(key, self.networks[key], val['pretrained_path'])

            self.logger.info(self.networks[key])

    def network_load_pretrained(self, key, network, pretrained_path):
        self.logger.info('==> Load pretrained network parameters from file %s:' %
                         pretrained_path)

        assert(os.path.isfile(pretrained_path))
        pretrained_model = torch.load(pretrained_path, map_location='cpu')
        # pretrained_model = torch.load(pretrained_path)
        if key == 'semantic_seg':
            network.load_state_dict(pretrained_model)
            return

        if key in pretrained_model.keys():
            if pretrained_model[key].keys() == network.state_dict().keys():
                self.logger.info('==> network parameters in pre-trained file'
                                 ' %s can strictly match' % (pretrained_path))
                network.load_state_dict(pretrained_model[key])
            else:
                self.logger.info('==> WARNING: network parameters in pre-trained file'
                                 ' %s do not strictly match' % (pretrained_path))
                for pname, param in network.named_parameters():
                    if pname in pretrained_model[key]:
                        self.logger.info('==> Copying parameter %s from file %s' %
                                         (pname, pretrained_path))
                        param.data.copy_(pretrained_model[key][pname])
                    if 'module.'+pname in pretrained_model[key]:
                        self.logger.info('==> Copying parameter %s from file %s' %
                                         ('module.'+pname, pretrained_path))
                        param.data.copy_(pretrained_model[key]['module.'+pname])
        else:
            # network.load_state_dict(pretrained_model)
            self.logger.info('==> WARNING: can not load network %s from file %s:' %
                         (key, pretrained_path))

    def optimizer_load_pretrained(self, key, optimizer, lr_scheduler, pretrained_path):
        self.logger.info('==> Load pretrained optimizer and lr_scheduler '
                         'parameters from file %s:' %
                         pretrained_path)

        assert(os.path.isfile(pretrained_path))
        # pretrained_model = torch.load(pretrained_path, map_location='cpu')
        pretrained_model = torch.load(pretrained_path, map_location='cpu')
        optimizer.load_state_dict(pretrained_model[key+'_optimizer'])
        lr_scheduler.load_state_dict(pretrained_model[key+'_lr_scheduler'])

        epoch = pretrained_model['epoch']
        self.curr_epoch = max(self.curr_epoch, epoch + 1)

    def init_all_optimizers(self):
        optimizers_defs = self.config.optimizers

        for key, val in optimizers_defs.items():
            if self.config.phase == 'test':
                continue

            if val is None or key not in self.networks.keys():
                continue

            self.logger.info('Set optimizers %s' % key)

            self.init_optimizer(key, val)

    def trainable_parameters(self, net):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in net.parameters():
            if param.requires_grad:
                yield param

    def split_weights(self, net):
        """split network weights into to categlories,
        one are weights in conv layer and linear layer,
        others are other learnable paramters(conv bias,
        bn weights, bn bias, linear bias)

        Args:
            net: network architecture

        Returns:
            a dictionary of params splite into to categlories
        """

        decay = []
        no_decay = []

        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    decay.append(m.weight)

                if m.bias is not None and m.weight.requires_grad:
                    no_decay.append(m.bias)
            elif isinstance(m, nn.ParameterDict):
                for name, param in m.items():
                    if param.requires_grad:
                        decay.append(param)
            else:
                if hasattr(m, 'weight') and m.weight.requires_grad:
                    no_decay.append(m.weight)
                if hasattr(m, 'bias') and m.bias.requires_grad:
                    no_decay.append(m.bias)
                if hasattr(m, 'scale_cls') and m.scale_cls.requires_grad:
                    decay.append(m.scale_cls)
                if hasattr(m, 'weight_cos') and m.weight_cos.requires_grad:
                    decay.append(m.weight_cos)

        #x = list(self.trainable_parameters(net))

        assert len(list(self.trainable_parameters(net))) == len(decay) + len(no_decay)

        return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

    def init_optimizer(self, key, val):
        #x = self.networks[key].parameters()
        # if len(self.config['gpu_ids']) > 1:
        #     parameters = filter(lambda p: p.requires_grad,
        #                         self.networks[key].module.parameters())
        # else:
        parameters = filter(lambda p: p.requires_grad,
                            self.networks[key].parameters())

        # parameters = self.split_weights(self.networks[key])

        # parameters = filter(lambda p: p.requires_grad,
        #                     parameters)

        self.logger.info('Initialize optimizer: %s with params: %s for netwotk: %s'
                % (val['type'], val, key))
        if val['type'] == 'sgd':
            self.optimizers[key] = torch.optim.SGD(parameters,
                                        lr=val['lr'],
                                        momentum=val['momentum'],
                                        nesterov=val['nesterov'],
                                        weight_decay=val['weight_decay'])
        elif val['type'] == 'adam':
            self.optimizers[key] = torch.optim.Adam(parameters,
                                         lr=val['lr'],
                                         betas=val['beta'],
                                         weight_decay=val['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optimizer_type', val['type'])

        if val['lr_scheduler'] == 'MultiStepLR':
            self.lr_schedulers[key] = torch.optim.lr_scheduler.MultiStepLR(self.optimizers[key],
                                                         milestones=val['milestones'],
                                                         gamma=val['gamma'])
        elif val['lr_scheduler'] == 'CosineLR':
            self.lr_schedulers[key] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers[key],
                                                         T_max=val['t_max'])
        else:
            raise ValueError('Not supported or recognized lr_scheduler_type', val['lr_scheduler'])

        if val['pretrained_path'] is not None:
            self.optimizer_load_pretrained(key, self.optimizers[key],
                                           self.lr_schedulers[key], val['pretrained_path'])

    def init_all_criterions(self):
        criterions_defs = self.config.criterions

        for key, val in criterions_defs.items():
            self.init_criterion(key, val)

    def init_criterion(self, key, val):
        self.logger.info('Initialize criterion[%s]: %s' %
                         (key, val))

        if val == 'CrossEntropyLoss':
            self.gan_class_weight = self.class_weight.repeat(2)
            self.criterions[key] = getattr(nn, val)(weight=self.gan_class_weight)
        else:
            self.criterions[key] = getattr(nn, val)()

    # def load_to_gpu(self):
    #     if len(self.config.train_gpu_ids') > 1:
    #         for key, net in self.networks.items():
    #             self.networks[key] = nn.DataParallel(net)
    #             self.networks[key] = self.networks[key].cuda()
    #     else:
    #         for key, net in self.networks.items():
    #             self.networks[key] = net.cuda()

    #     # for key, criterion in self.criterions.items():
    #     #         self.criterions[key] = criterion.cuda()

    def save_checkpoint(self, epoch, suffix='', metric=None):
        if self.config.local_rank != 0:
            return 
        ####### save all networks ######
        net_state = dict()
        net_state['epoch'] = epoch
        net_state['metric'] = metric
        filename = self._get_net_checkpoint_filename(epoch)+suffix
        for key, net in self.networks.items():
            # self.save_network(key, epoch, suffix=suffix, metric=metric)
            # if len(self.config['gpu_ids']) > 1:
            #     net_state[key] = self.networks[key].module.state_dict()
            # else:
            #     net_state[key] = self.networks[key].state_dict()

            ## DDP
            if self.config.world_size > 0:
                net_state[key] = self.networks[key].module.state_dict()
            else:
                net_state[key] = self.networks[key].state_dict()
        torch.save(net_state, filename)

        ####### save optimizers ######
        optimizer_state = dict()
        optimizer_state['epoch'] = epoch
        filename = self._get_optim_checkpoint_filename(epoch)+suffix
        for key, net in self.optimizers.items():
            # self.save_optimizer(key, epoch, suffix=suffix)
            optimizer_state[key+'_optimizer'] = self.optimizers[key].state_dict()
            optimizer_state[key+'_lr_scheduler'] = self.lr_schedulers[key].state_dict()
        torch.save(optimizer_state, filename)

    def delete_checkpoint(self, epoch, suffix=''):
        filename_net = self._get_net_checkpoint_filename(epoch)+suffix
        if os.path.isfile(filename_net):
            os.remove(filename_net)

        filename_optim = self._get_optim_checkpoint_filename(epoch)+suffix
        if os.path.isfile(filename_optim):
            os.remove(filename_optim)

    # def save_network(self, net_key, epoch, suffix='', metric=None):
    #     assert(net_key in self.networks)
    #     filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
    #     if len(self.config['gpu_ids']) > 1:
    #         state = {
    #             'epoch': epoch,
    #             'network': self.networks[net_key].module.state_dict(),
    #             'metric': metric
    #         }
    #     else:
    #         state = {
    #             'epoch': epoch,
    #             'network': self.networks[net_key].state_dict(),
    #             'metric': metric
    #         }
    #     torch.save(state, filename)

    # def save_optimizer(self, net_key, epoch, suffix=''):
    #     assert(net_key in self.optimizers)
    #     filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
    #     state = {
    #         'epoch': epoch,
    #         'optimizer': self.optimizers[net_key].state_dict(),
    #         'lr_scheduler': self.lr_schedulers[net_key].state_dict()
    #     }
    #     torch.save(state, filename)

    def _get_net_checkpoint_filename(self, epoch):
        return os.path.join(self.exp_dir, 'net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, epoch):
        return os.path.join(self.exp_dir, 'optim_epoch'+str(epoch))


    def adjust_learning_rates(self, epoch):
        for key in self.lr_schedulers.keys():
            self.lr_schedulers[key].step(epoch)

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.metric1 is not None:
            metric_name = self.metric1
            if metric_name not in eval_stats:
                raise ValueError('The provided metric {0} for keeping the best '
                                 'model is not computed by the evaluation routine.'
                                 .format(metric_name))
            metric_val = eval_stats[metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch, suffix='.best')
                self.save_checkpoint(
                    current_epoch, suffix='.best', metric=self.max_metric_val)
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.metric1
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s'
                             % (metric_name, self.best_epoch, self.best_stats))

    def dict2str(self, inputs):
        result = ''
        for key, val in inputs.items():
            if isinstance(val, dict):
                result += key + ': ' + '\n' + self.dict2str(val) + '\n'
            else:
                result += key + ': ' + str(val) + '\n'

        return result

    def dict2writer(self, inputs, n_iter, is_Training):
        if is_Training:
            phase = 'Train/'
        else:
            phase = 'Eval/'
        for key, val in inputs.items():
            self.writer.add_scalar(phase + key, val, n_iter)
            # if 'refine' in key:
            #     self.writer.add_scalar(phase + 'refine/' + key, val, n_iter)
            # elif 'G' in key:
            #     self.writer.add_scalar(phase + 'G_loss/' + key, val, n_iter)
            # elif 'D' in key:
            #     self.writer.add_scalar(phase + 'D_loss/' + key, val, n_iter)

    def img2writer(self, img, n_iter, key):
        self.writer.add_image(key, img, n_iter)
        # self.writer.add_image('Static/syn', s_syn, n_iter)
        # self.writer.add_image('Motion/real', m_real, n_iter)
        # self.writer.add_image('Motion/syn', m_syn, n_iter)

    # Converts a Tensor into a Numpy array
    # |imtype|: the desired type of the converted numpy array
    def tensor2im(self, image_tensor, normalize=True):
        imtype = np.uint8
        # if isinstance(image_tensor, list):
        #     image_numpy = []
        #     for i in range(len(image_tensor)):
        #         image_numpy.append(self.tensor2im(image_tensor[i], imtype, normalize))
        #     return image_numpy

        if isinstance(image_tensor, torch.autograd.Variable):
            image_tensor = image_tensor.data
        # if len(image_tensor.size()) == 5:
        #     image_tensor = image_tensor[0, -1]
        # if len(image_tensor.size()) == 4:
        #     image_tensor = image_tensor[0]
        # image_tensor = image_tensor[:3]
        image_numpy = image_tensor.cpu().float().numpy()
        if normalize:
            # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            image_numpy = (image_numpy + 1) / 2.0 * 255.0
        else:
            # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            image_numpy = image_numpy * 255.0
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1:
            image_numpy = image_numpy[:, :, 0]
        return image_numpy.astype(imtype)
