from torch.nn import CrossEntropyLoss, BCELoss
import os

class basic_config(object):
    def __init__(self):
        self.train_gpu_ids = [0]
        self.train_gpu_num = 0 if -1 in self.train_gpu_ids else len(self.train_gpu_ids)
        self.node_num = 1 #1
        self.world_size = self.train_gpu_num * self.node_num

        self.test_gpu_ids = [0]
        self.test_gpu_num = 0 if -1 in self.test_gpu_ids else len(self.test_gpu_ids)
        
        self.save_interval = 5 #3

        # 1: 1 center rgb
        # 2: 4 center rgb
        # 3: 1 center rgb, lidar
        # 4: 4 center rgb, lidar
        self.input_mode = -1

        # 0: last rgb
        # 1: last rgb, light status, light dist
        # 2: last rgb, topdown rgb, light status, light dist
        # 3: last rgb, topdown semantic, light status, light dist
        # 4: last rgb, lidar, light status, light dist
        # 5: last rgb, lidar, topdown rgb, light status, light dist
        # 6: last rgb, lidar, topdown semantic, light status, light dist
        self.output_mode = -1
        
        self.max_num_epochs = 100
        self.warm = 0
        self.mixed_percision = False

        ##################################
        self.town_route_dict = dict()
        # remain one route to test
        self.town_route_dict['all'] = [index for index in range(49+1)] #[0-49]
        self.town_route_dict['town01'] = [index for index in range(9)] #[0-8 for train, 9 for test]
        self.town_route_dict['nocrash01'] = [index for index in range(24)] #[0-23 for train, 24 for test]
        self.town_route_dict['nocrashBa'] = [index for index in range(24)] + [index for index in range(25, 49)] #[0-23, 25-48 for train, 24, 49 for test]
        self.town_route_dict['nocrashBaNew'] = [index for index in range(24)]
        self.town_route_dict['challenge75'] = [index for index in range(76)]
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
        self.img_transform_optionsF['GaussianBlur'] = True #False
        self.img_transform_optionsF['AdditiveGaussianNoise'] = True #False
        self.img_transform_optionsF['Dropout'] = True #False
        self.img_transform_optionsF['CoarseDropout'] = True #False
        self.img_transform_optionsF['Add'] = True #False
        self.img_transform_optionsF['Multiply'] = True #False
        self.img_transform_optionsF['ContrastNormalization'] = False
        self.img_transform_optionsF['toTensor'] = True

        ##################################
        self.topdown_rgb_transform_optionsF = dict()
        self.topdown_rgb_transform_optionsF['scale_type'] = 'resize'
        self.topdown_rgb_transform_optionsF['load_width'] = 256#256#1024#512
        self.topdown_rgb_transform_optionsF['load_height'] = 144#128#512#288
        self.topdown_rgb_transform_optionsF['normalize'] = False
        self.topdown_rgb_transform_optionsF['toTensor'] = True

        ##################################
        self.lidar_transform_optionsF = dict()
        self.lidar_transform_optionsF['scale_type'] = 'resize'
        self.lidar_transform_optionsF['load_width'] = 256#256#1024#512
        self.lidar_transform_optionsF['load_height'] = 144#128#512#288
        self.lidar_transform_optionsF['normalize'] = False
        self.lidar_transform_optionsF['toTensor'] = True

    def change_input_mode(self, input_mode):
        self.in_left_camera = False
        self.in_right_camera = False
        self.in_speed = False
        self.in_bc_speed = False

        if input_mode == 1:
            self.in_backbone = 1
            self.in_lidar = False
            self.in_route = False
        elif input_mode == 2:
            self.in_backbone = 4
            self.in_lidar = False
            self.in_route = False
        elif input_mode == 3:
            self.in_backbone = 1
            self.in_lidar = True
            self.in_route = False
        elif input_mode == 4:
            self.in_backbone = 4
            self.in_lidar = True
            self.in_route = False
        elif input_mode == 5:
            self.in_backbone = 1
            self.in_lidar = False
            self.in_route = True
        elif input_mode == 6:
            self.in_backbone = 1
            self.in_lidar = False
            self.in_route = True
            self.in_left_camera = True
            self.in_right_camera = True
        elif input_mode == 7:
            self.in_backbone = 1
            self.in_lidar = False
            self.in_route = True
            self.in_left_camera = False
            self.in_right_camera = False
            self.in_speed = True
        elif input_mode == 8:
            self.in_backbone = 1
            self.in_lidar = True
            self.in_route = True
            self.in_left_camera = False
            self.in_right_camera = False
            self.in_speed = True
        elif input_mode == 9:
            self.in_backbone = 1
            self.in_lidar = False
            self.in_route = True
            self.in_left_camera = False
            self.in_right_camera = False
            self.in_bc_speed = True
        elif input_mode == 10:
            self.in_backbone = 1
            self.in_lidar = False
            self.in_route = False
            self.in_left_camera = False
            self.in_right_camera = False
            self.in_bc_speed = True
    
    def change_output_mode(self, output_mode):
        self.pred_light_state = False
        self.pred_light_dist = False
        self.pred_camera_seg = False
        self.pred_left_camera_seg = False
        self.pred_right_camera_seg = False
        self.pred_route = False
        self.pred_bc = False
        self.pred_lidar = False
        self.pred_topdown_rgb = False
        self.pred_topdown_seg = False
    
        if output_mode == 1:
            self.pred_light_state = True
            self.pred_light_dist = True
        elif output_mode == 2:
            self.pred_topdown_rgb = True
            self.pred_light_state = True
            self.pred_light_dist = True
        elif output_mode == 3:
            self.pred_light_state = True
            self.pred_light_dist = True
            self.pred_topdown_seg = True
        elif output_mode == 4:
            self.pred_lidar = True
            self.pred_light_state = True
            self.pred_light_dist = True
        elif output_mode == 5:
            self.pred_lidar = True
            self.pred_topdown_rgb = True
            self.pred_light_state = True
            self.pred_light_dist = True
        elif output_mode == 6:
            self.pred_lidar = True
            self.pred_topdown_seg = True
            self.pred_light_state = True
            self.pred_light_dist = True
        elif output_mode == 7:
            self.pred_camera_seg = True
        elif output_mode == 8:
            self.pred_camera_seg = True
            self.pred_route = True
        elif output_mode == 9:
            self.pred_camera_seg = True
            self.pred_route = True
            self.pred_light_state = True
        elif output_mode == 10:
            self.pred_camera_seg = True
            self.pred_left_camera_seg = True
            self.pred_right_camera_seg = True
            self.pred_route = True
        elif output_mode == 11:
            self.pred_camera_seg = True
            self.pred_left_camera_seg = True
            self.pred_right_camera_seg = True
            self.pred_route = True
            self.pred_light_state = True
        elif output_mode == 12:
            self.pred_camera_seg = True
            self.pred_route = True
            self.pred_light_state = True
            self.pred_bc = True
        elif output_mode == 13:
            self.pred_camera_seg = True
            self.pred_route = False
            self.pred_light_state = True
            self.pred_bc = False
        elif output_mode == 14:
            self.pred_camera_seg = True
            self.pred_route = False
            self.pred_light_state = True
            self.pred_bc = True