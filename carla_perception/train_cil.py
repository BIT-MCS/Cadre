from __future__ import print_function
import argparse
import os
import torch
from importlib import util
from Data.dataloaders import DataloaderFactory
from Models.cil_trainer import Cil_Trainer

from datetime import datetime
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDPw
from apex import amp

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cil_net_config',
                        help='config file with parameters of the experiment.')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args_opt = parser.parse_args()

    exp_config_file = os.path.join('.', 'Config', args_opt.config + '.py')
    # exp_directory = os.path.join('.', 'Experiments', args_opt.config)
    # print('Launching experiment: %s' % exp_config_file)

    # Load the configuration params of the experiment
    config_module_child_name = args_opt.config
    config_spec = util.spec_from_file_location(config_module_child_name, exp_config_file)
    config_module = util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    config = config_module.config

    # the place where logs, models, and other stuff will be stored
    # config['exp_dir'] = os.path.join('.', 'Experiments', config['exp_dir'])
    config.exp_dir = os.path.join('.', 'Experiments'+config.exp_suffix, config.exp_dir)

    print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
    # print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))
    print("Generated logs, snapshots, and model files will be stored on %s" % (config.exp_dir))

    config.local_rank = args_opt.local_rank

    return config

if __name__ == "__main__":
    config = load_config()
    # torch.manual_seed(0)

    # config['world_size'] = config['gpu_num'] * config['node_num']
    # rank = config['local_rank']
    rank = config.local_rank
    # os.environ['MASTER_ADDR'] = config['MASTER_ADDR']        
    # os.environ['MASTER_PORT'] = config['MASTER_PORT']                   
    # mp.spawn(train_process, nprocs=config['gpu_num'], args=(config,))         

    CUDA_VISIBLE_DEVICES = str(config.train_gpu_ids[0])
    for gpu_index in range(1, len(config.train_gpu_ids)):
        CUDA_VISIBLE_DEVICES += ", " + str(config.train_gpu_ids[gpu_index])

    print('Current GPU index: ' + CUDA_VISIBLE_DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    # config['world_size'] = 0
    if config.world_size > 1:
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://'                                            
        )  

    train_opt = config.train_opt
    train_opt['rank'] = rank
    train_opt['world_size'] = config.world_size
    train_dataloader, train_sampler, train_dataset = DataloaderFactory(train_opt).load_dataloader()

    #print('1--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

    # test_opt = config['test_opt']
    # test_dataloader = DataloaderFactory(test_opt).load_dataloader()

    # transfer_opt = config['transfer_opt']
    # transfer_dataloader = DataloaderFactory(transfer_opt).load_dataloader()

    config.class_weight = train_dataset.class_weight
    config.num_data_img = len(train_dataset)
    cil_trainer = Cil_Trainer(config)

    #print('2--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

    cil_trainer.solve(train_dataloader, train_sampler)