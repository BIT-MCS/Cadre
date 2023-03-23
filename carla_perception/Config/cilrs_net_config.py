from torch.nn import CrossEntropyLoss, BCELoss
import os

class cilrs_net_config(object):
    def __init__(self):
        self.train_gpu_ids = [0, 1, 2, 3]
        self.train_gpu_num = 0 if -1 in self.train_gpu_ids else len(self.train_gpu_ids)
        self.node_num = 1 #1

        self.test_gpu_ids = [0]
        self.test_gpu_num = 0 if -1 in self.test_gpu_ids else len(self.test_gpu_ids)
        
        self.save_interval = 5

        # input: img, speed
        self.input_mode = 0
        # output: steer, acc, brake
        self.output_mode = 0
        self.in_backbone = 1

        self.branch_weight = [0.95, 0.95, 0.95, 0.95, 0.05]
        self.variable_weight = dict()
        self.variable_weight['steer'] = 0.5
        self.variable_weight['throttle'] = 0.45
        self.variable_weight['brake'] = 0.05

        self.augment_lateral_steer = 6
        self.speed_factor = 12.0
        
        self.max_num_epochs = 100
        self.warm = 0
        self.mixed_percision = False

        self.local_rank = -1

        self.phase = 'train' #'test' or 'train'
        self.load_epoch = -1

        self.version = 'cilrs'
        self.train_town = "town01" # "all"

        version_suffix = str(self.input_mode) + str(self.output_mode)

        self.exp_suffix = '_cilrs1'
        self.exp_dir = self.version + version_suffix + "_" + self.train_town

        ##################################
        self.town_route_dict = dict()
        # remain one route to test
        self.town_route_dict['all'] = [index for index in range(49+1)] #[0-49]
        self.town_route_dict['town01'] = [index for index in range(9)] #[0-8 for train, 9 for test]
        self.town_route_dict['town02'] = [index for index in range(50, 55)] #[50-55, 55 for test]
        self.town_route_dict['town03'] = [index for index in range(10, 29)] #[10-29, 29 for test]
        self.town_route_dict['town04'] = [index for index in range(30, 39)] #[30-39, 56-65, 39 for test]
        self.town_route_dict['town05'] = [index for index in range(66, 75)] #[66-75, 75 for test]
        self.town_route_dict['town06'] = [index for index in range(40, 49)] #[40-49, 49 for test]

        ##################################
        self.img_transform_optionsF = dict()
        self.img_transform_optionsF['scale_type'] = 'resize'
        self.img_transform_optionsF['load_width'] = 256 #256#1024#512
        self.img_transform_optionsF['load_height'] = 144 #128#512#288
        self.img_transform_optionsF['normalize'] = False
        self.img_transform_optionsF['GaussianBlur'] = True
        self.img_transform_optionsF['AdditiveGaussianNoise'] = True
        self.img_transform_optionsF['Dropout'] = True
        self.img_transform_optionsF['CoarseDropout'] = True
        self.img_transform_optionsF['Add'] = True
        self.img_transform_optionsF['Multiply'] = True
        self.img_transform_optionsF['ContrastNormalization'] = True
        self.img_transform_optionsF['toTensor'] = True

        ############ train opt ################
        self.train_opt = dict()
        self.train_opt['data_name'] = 'carla_cilrs'
        self.train_opt['route_index'] = self.town_route_dict[self.train_town]
        self.train_opt['batch_size'] = 32
        self.train_opt['num_workers'] = 4
        self.train_opt['in_num_frames'] = self.in_backbone
        self.train_opt['phase'] = 'train'
        self.train_opt['root_dir'] = '/data2/wk/datasets/cil_cilrs_dataset/train'
        self.train_opt['img_transform_params'] = self.img_transform_optionsF

        ############ test opt ################
        self.test_opt = dict()
        self.test_opt['data_name'] = 'carla_cilrs_test'
        self.test_opt['route_index'] = [9] #[1]
        self.test_opt['batch_size'] = 1
        self.test_opt['num_workers'] = 0
        self.test_opt['in_num_frames'] = self.in_backbone
        self.test_opt['phase'] = 'test'
        self.test_opt['root_dir'] = '/data2/wk/datasets/cil_cilrs_dataset/test'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/3camera_seg10/train'
        self.test_opt['route_name'] = 'route_' + str(self.test_opt['route_index'][0]).zfill(2)
        self.test_opt['img_transform_params'] = self.img_transform_optionsF

        #######################
        self.networks = dict()
        self.optimizers = dict()

        ###########  AutoEncoder ################
        self.networks['cilrs_net'] = dict()
        self.networks['cilrs_net']['net_name'] = 'cilrs_net'
        self.networks['cilrs_net']['model_name'] = 'cilrs'
        self.networks['cilrs_net']['sensor_input_shape'] = [self.in_backbone * 3, 144, 256] 

        self.networks['cilrs_net']['perception'] = dict()
        self.networks['cilrs_net']['perception']['res'] = dict()
        self.networks['cilrs_net']['perception']['res']['name'] = 'resnet34'
        self.networks['cilrs_net']['perception']['res']['num_classes'] = 512

        self.networks['cilrs_net']['measurements'] = dict()
        self.networks['cilrs_net']['measurements']['fc'] = dict()
        self.networks['cilrs_net']['measurements']['fc']['neurons'] = [128, 128]
        self.networks['cilrs_net']['measurements']['fc']['dropouts'] = [0.0, 0.0]

        self.networks['cilrs_net']['join'] = dict()
        self.networks['cilrs_net']['join']['fc'] = dict()
        self.networks['cilrs_net']['join']['fc']['neurons'] = [512]
        self.networks['cilrs_net']['join']['fc']['dropouts'] = [0.0]

        self.networks['cilrs_net']['speed_branch'] = dict()
        self.networks['cilrs_net']['speed_branch']['fc'] = dict()
        self.networks['cilrs_net']['speed_branch']['fc']['neurons'] = [256, 256]
        self.networks['cilrs_net']['speed_branch']['fc']['dropouts'] = [0.0, 0.5]

        self.networks['cilrs_net']['branches'] = dict()
        self.networks['cilrs_net']['branches']['number_of_branches'] = 4
        self.networks['cilrs_net']['branches']['fc'] = dict()
        self.networks['cilrs_net']['branches']['fc']['neurons'] = [256, 256]
        self.networks['cilrs_net']['branches']['fc']['dropouts'] = [0.0, 0.5]

        if self.load_epoch == -1:
            self.networks['cilrs_net']['pretrained_path'] = None
            self.networks['cilrs_net']['pretrained'] = False
        else:
            self.networks['cilrs_net']['pretrained'] = True
            file_name = 'net_epoch' + str(self.load_epoch)
            self.networks['cilrs_net']['pretrained_path'] = os.path.join('../carla_perception/Experiments'+self.exp_suffix,
                                                                self.exp_dir,
                                                                file_name)

        self.optimizers['cilrs_net'] = dict()
        self.optimizers['cilrs_net']['type'] = 'adam'
        self.optimizers['cilrs_net']['lr'] = 0.0002
        self.optimizers['cilrs_net']['beta'] = (0.9, 0.999)
        self.optimizers['cilrs_net']['weight_decay'] = 5e-4
        self.optimizers['cilrs_net']['lr_scheduler'] = 'CosineLR'
        self.optimizers['cilrs_net']['t_max'] = self.max_num_epochs
        if self.load_epoch == -1:
            self.optimizers['cilrs_net']['pretrained_path'] = None
        else:
            file_name = 'optim_epoch' + str(self.load_epoch)
            self.optimizers['cilrs_net']['pretrained_path'] = os.path.join('../carla_perception/Experiments'+self.exp_suffix,
                                                                    self.exp_dir,
                                                                    file_name)
        
        self.metric1 = 'L1Loss'
        self.criterions = dict()

config = cilrs_net_config()