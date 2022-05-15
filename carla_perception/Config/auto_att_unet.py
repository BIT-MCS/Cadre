from torch.nn import CrossEntropyLoss, BCELoss
import os

config = dict()

config['gpu_ids'] = [0]#[0, 1, 2, 3] #default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
config['gpu_num'] = 0 if -1 in config['gpu_ids'] else len(config['gpu_ids'])
config['node_num'] = 1 #1

config['test_gpu_ids'] = [0]
config['test_gpu_num'] = 0 if -1 in config['test_gpu_ids'] else len(config['test_gpu_ids'])

config['save_interval'] = 2

config['phase'] = 'train' #'test' or 'train'
config['load_epoch'] = 30 #-1

# ['seg-base', 'noseg-base', 'seg-gan', 'noseg-gan', 'seg-salient', 'noseg-salient', 'seg-all', 'noseg-all']
config['version'] = 'att-unet' 
config['train_town'] = "town01" # "all"
config['exp_dir'] = 'carla_' + config['version'] + "_" + config['train_town']

config['in_backbone'] = 1
config['max_num_epochs'] = 30
config['warm'] = 0 #5

config['mixed_percision'] = True

town_route_dict = dict()
# remain one route to test
town_route_dict['all'] = [index for index in range(49+1)] #[0-49]
town_route_dict['town01'] = [index for index in range(9)] #[0-8 for train, 9 for test]
town_route_dict['town02'] = [index for index in range(50, 55)] #[50-55, 55 for test]
town_route_dict['town03'] = [index for index in range(10, 29)] #[10-29, 29 for test]
town_route_dict['town04'] = [index for index in range(30, 39)] #[30-39, 56-65, 39 for test]
town_route_dict['town05'] = [index for index in range(66, 75)] #[66-75, 75 for test]
town_route_dict['town06'] = [index for index in range(40, 49)] #[40-49, 49 for test]

##################################
img_transform_optionsF = dict()
img_transform_optionsF['scale_type'] = 'resize'
img_transform_optionsF['load_width'] = 256#256#1024#512
img_transform_optionsF['load_height'] = 144#128#512#288
img_transform_optionsF['normalize'] = True
img_transform_optionsF['toTensor'] = True

##################################
topdown_transform_optionsF = dict()
topdown_transform_optionsF['scale_type'] = 'resize'
topdown_transform_optionsF['load_width'] = 256#256#1024#512
topdown_transform_optionsF['load_height'] = 144#128#512#288
topdown_transform_optionsF['normalize'] = True
topdown_transform_optionsF['toTensor'] = True

##################################
lidar_transform_optionsF = dict()
lidar_transform_optionsF['scale_type'] = 'resize'
lidar_transform_optionsF['load_width'] = 256#256#1024#512
lidar_transform_optionsF['load_height'] = 144#128#512#288
lidar_transform_optionsF['normalize'] = True
lidar_transform_optionsF['toTensor'] = True

############ train opt ################
train_opt = dict()
train_opt['data_name'] = 'carla_percept'
train_opt['route_index'] = town_route_dict[config['train_town']]
train_opt['batch_size'] = 48
train_opt['num_workers'] = 4
train_opt['in_num_frames'] = config['in_backbone']
train_opt['phase'] = config['phase']
train_opt['root_dir'] = '/data2/wk/datasets/carla_own_collect_vae/train'
train_opt['img_transform_params'] = img_transform_optionsF
train_opt['topdown_transform_params'] = topdown_transform_optionsF
train_opt['lidar_transform_params'] = lidar_transform_optionsF
config['train_opt'] = train_opt

############ test opt ################
test_opt = dict()
test_opt['data_name'] = 'carla_percept_test'
test_opt['route_index'] = [9]
test_opt['batch_size'] = 1
test_opt['num_workers'] = 0
test_opt['in_num_frames'] = config['in_backbone']
test_opt['phase'] = 'test'
test_opt['root_dir'] = '/data2/wk/datasets/carla_own_collect_vae/train'
test_opt['route_name'] = 'route_' + str(test_opt['route_index'][0]).zfill(2)
test_opt['img_transform_params'] = img_transform_optionsF
test_opt['topdown_transform_params'] = topdown_transform_optionsF
test_opt['lidar_transform_params'] = lidar_transform_optionsF
config['test_opt'] = test_opt

###############################

networks = dict()
optimizers = dict()

###########  AutoEncoder ################
networks['autoencoder'] = dict()
networks['autoencoder']['net_name'] = 'autoencoder'
networks['autoencoder']['model_name'] = 'att-unet'
networks['autoencoder']['input_channel'] = config['in_backbone'] * (3 + 3)
networks['autoencoder']['output_channel'] = 3 + 3 + 3
networks['autoencoder']['light_classes_num'] = 4
networks['autoencoder']['z_dims'] = 128
# networks['autoencoder']['no_of_samples'] = 10
if config['load_epoch'] == -1:
    networks['autoencoder']['pretrained_path'] = None
    networks['autoencoder']['pretrained'] = False
else:
    networks['autoencoder']['pretrained'] = True
    file_name = 'net_epoch' + str(config['load_epoch'])
    networks['autoencoder']['pretrained_path'] = os.path.join('../Carla_Perception/Experiments',
                                                           config['exp_dir'],
                                                           file_name)

optimizers['autoencoder'] = dict()
optimizers['autoencoder']['type'] = 'adam'
optimizers['autoencoder']['lr'] = 0.00004
optimizers['autoencoder']['beta'] = (0.5, 0.999)
optimizers['autoencoder']['weight_decay'] = 5e-4
optimizers['autoencoder']['lr_scheduler'] = 'CosineLR'
optimizers['autoencoder']['t_max'] = config['max_num_epochs']
if config['load_epoch'] == -1:
    optimizers['autoencoder']['pretrained_path'] = None
else:
    file_name = 'optim_epoch' + str(config['load_epoch'])
    optimizers['autoencoder']['pretrained_path'] = os.path.join('../Carla_Perception/Experiments',
                                                             config['exp_dir'],
                                                             file_name)

############# Criterions ###################
criterions = dict()

# criterions['action_reg'] = 'L1Loss'
# criterions['command_D'] = 'CrossEntropyLoss' 
# criterions['command_G'] = 'CrossEntropyLoss'

config['metric1'] = 'L1Loss'

config['criterions'] = criterions
config['networks'] = networks
config['optimizers'] = optimizers



